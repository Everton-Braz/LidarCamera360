#include "GraphExecutor.h"
#include "VulkanCompute.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>

namespace rvk {
namespace {
using Shape = std::vector<int>;
void require(bool ok, const std::string &error) {
  if (!ok)
    throw std::runtime_error(error);
}
int count(const Shape &s) {
  int64_t n = 1;
  for (int d : s) {
    require(d > 0 && d <= 16777216, "Invalid dimension");
    n *= d;
    require(n <= INT32_MAX / 4, "Tensor too large");
  }
  return int(n);
}
int axis(int x, int rank) {
  if (x < 0)
    x += rank;
  require(x >= 0 && x < rank, "Axis out of range");
  return x;
}
int bits(float x) {
  int r;
  std::memcpy(&r, &x, 4);
  return r;
}
bool alias(const std::string &op) {
  return op == "Reshape" || op == "Unsqueeze" || op == "Squeeze" ||
         op == "Identity";
}
struct Reader {
  std::ifstream f;
  uint64_t remaining;
  explicit Reader(const std::filesystem::path &p)
      : f(p, std::ios::binary | std::ios::ate) {
    require(bool(f), "Cannot open RVK model");
    auto n = f.tellg();
    require(n >= 20 && n <= int64_t(2) * 1024 * 1024 * 1024,
            "Invalid RVK file size");
    remaining = uint64_t(n);
    f.seekg(0);
  }
  void read(void *p, size_t n) {
    require(n <= remaining, "Truncated RVK model");
    require(bool(f.read(static_cast<char *>(p), n)), "Cannot read RVK model");
    remaining -= n;
  }
  uint32_t u() {
    uint32_t v;
    read(&v, 4);
    return v;
  }
  std::string str() {
    auto n = u();
    require(n <= 1048576, "RVK string too long");
    std::string s(n, '\0');
    read(s.data(), n);
    return s;
  }
  std::vector<int> ints(size_t n) {
    require(n <= 100000, "RVK list too long");
    std::vector<int> v(n);
    read(v.data(), n * 4);
    return v;
  }
};
struct Attr {
  std::vector<int> ints;
  std::vector<float> floats;
  std::string str;
};
struct Node {
  std::string op, name;
  std::vector<int> in, out;
  std::map<std::string, Attr> attrs;
  int integer(const std::string &key, int def = 0) const {
    auto i = attrs.find(key);
    if (i == attrs.end())
      return def;
    require(i->second.ints.size() == 1, "Invalid integer attribute " + key);
    return i->second.ints[0];
  }
  float real(const std::string &key, float def = 0) const {
    auto i = attrs.find(key);
    if (i == attrs.end())
      return def;
    require(i->second.floats.size() == 1, "Invalid float attribute " + key);
    return i->second.floats[0];
  }
  Shape integers(const std::string &key, Shape def = {}) const {
    auto i = attrs.find(key);
    return i == attrs.end() ? def : i->second.ints;
  }
  std::string text(const std::string &key, std::string def = {}) const {
    auto i = attrs.find(key);
    return i == attrs.end() ? def : i->second.str;
  }
};
struct Tensor {
  std::string name;
  int dtype;
  Shape shape;
  std::vector<float> values;
  int root = -1, offset = -1, last = -1;
};
Shape broadcast(const Shape &x, const Shape &y) {
  Shape s(std::max(x.size(), y.size()), 1);
  for (size_t i = 0; i < s.size(); ++i) {
    int a = i < x.size() ? x[x.size() - 1 - i] : 1,
        b = i < y.size() ? y[y.size() - 1 - i] : 1;
    require(a == b || a == 1 || b == 1, "Incompatible broadcast");
    s[s.size() - 1 - i] = std::max(a, b);
  }
  return s;
}
void layout(std::array<int, 256> &r, int start, const Tensor &t) {
  r[start] = t.offset;
  r[start + 1] = int(t.shape.size());
  r[start + 2] = count(t.shape);
  r[start + 3] = t.dtype;
  int stride = 1;
  for (int i = int(t.shape.size()) - 1; i >= 0; --i) {
    r[start + 4 + i] = t.shape[i];
    r[start + 12 + i] = stride;
    stride *= t.shape[i];
  }
}
const std::map<std::string, int> codes = {
    {"Identity", 0},    {"Add", 1},
    {"Sub", 2},         {"Mul", 3},
    {"Div", 4},         {"Exp", 5},
    {"Erf", 6},         {"Sqrt", 7},
    {"Relu", 8},        {"Sigmoid", 9},
    {"Sin", 10},        {"Cos", 11},
    {"Equal", 12},      {"Greater", 13},
    {"Less", 14},       {"And", 15},
    {"Not", 16},        {"Where", 17},
    {"Cast", 18},       {"Transpose", 19},
    {"Concat", 20},     {"Slice", 21},
    {"Expand", 22},     {"Tile", 23},
    {"Gather", 24},     {"GatherElements", 25},
    {"Split", 26},      {"Conv", 27},
    {"GridSample", 28}, {"Resize", 29},
    {"ReduceSum", 30},  {"ReduceMax", 31},
    {"TopK", 32},       {"Einsum", 33},
    {"MatMul", 34},     {"Gemm", 35},
    {"Softmax", 36},    {"LayerNormalization", 37}};
} // namespace
struct GraphExecutor::Impl {
  VulkanCompute gpu;
  std::vector<Tensor> tensors;
  std::vector<Node> nodes;
  std::vector<int> inputs, outputs;
  std::vector<int32_t> records;
  std::vector<Dispatch> dispatches;
  size_t arena_size = 0;
  explicit Impl(int device) : gpu(device) {}
  Shape constants(int id) {
    require(id >= 0 && !tensors[id].values.empty(),
            "Expected constant shape/axis input");
    Shape v;
    for (float f : tensors[id].values) {
      require(std::isfinite(f) && std::floor(f) == f &&
                  std::abs(double(f)) <= 16777216,
              "Invalid integer metadata");
      v.push_back(int(f));
    }
    return v;
  }
  void load(const std::filesystem::path &path,
            const std::filesystem::path &shaders) {
    Reader f(path);
    char magic[4];
    f.read(magic, 4);
    require(std::memcmp(magic, "RVK1", 4) == 0, "Not an RVK1 model");
    auto nt = f.u(), nn = f.u(), ni = f.u(), no = f.u();
    require(nt > 0 && nt < 100000 && nn < 100000 && ni == 1 && no > 0 &&
                no < 32,
            "Invalid RVK graph counts");
    tensors.resize(nt);
    for (auto &t : tensors) {
      t.name = f.str();
      t.dtype = int(f.u());
      require(t.dtype == 1 || t.dtype == 6 || t.dtype == 7 || t.dtype == 9,
              "Unsupported tensor dtype");
      auto rank = f.u();
      require(rank <= 8, "Tensor rank exceeds 8");
      t.shape = f.ints(rank);
      int size = count(t.shape);
      auto n = f.u();
      require(n == 0 || n == uint32_t(size), "Constant size mismatch");
      require(uint64_t(n) * 4 <= f.remaining, "Truncated constant");
      t.values.resize(n);
      f.read(t.values.data(), size_t(n) * 4);
      for (float x : t.values)
        require(std::isfinite(x), "Nonfinite constant");
    }
    inputs = f.ints(ni);
    outputs = f.ints(no);
    auto valid = [&](int i) { return i >= 0 && size_t(i) < tensors.size(); };
    for (int i : inputs)
      require(valid(i) && tensors[i].values.empty() && tensors[i].dtype == 1,
              "Invalid graph input");
    for (int i : outputs)
      require(valid(i), "Invalid graph output");
    for (uint32_t k = 0; k < nn; ++k) {
      Node n;
      n.op = f.str();
      n.name = f.str();
      auto ic = f.u();
      require(ic > 0 && ic <= 8, "Invalid node input count");
      n.in = f.ints(ic);
      auto oc = f.u();
      require(oc > 0 && oc <= 8, "Invalid node output count");
      n.out = f.ints(oc);
      for (int i : n.in)
        require(i == -1 || valid(i), "Invalid input tensor ID");
      for (int i : n.out)
        require(valid(i), "Invalid output tensor ID");
      auto ac = f.u();
      require(ac < 64, "Too many attributes");
      for (uint32_t j = 0; j < ac; ++j) {
        auto key = f.str();
        auto type = f.u();
        Attr a;
        if (type == 2)
          a.str = f.str();
        else {
          auto size = f.u();
          require(size <= 256, "Attribute too large");
          if (type == 0)
            a.ints = f.ints(size);
          else if (type == 1) {
            a.floats.resize(size);
            f.read(a.floats.data(), size * 4);
            for (float x : a.floats)
              require(std::isfinite(x), "Nonfinite attribute");
          } else
            throw std::runtime_error("Unknown attribute type");
        }
        require(n.attrs.emplace(key, std::move(a)).second,
                "Duplicate attribute");
      }
      nodes.push_back(std::move(n));
    }
    require(f.remaining == 0, "Trailing RVK data");
    std::vector<bool> defined(nt, false);
    for (size_t i = 0; i < nt; ++i) {
      tensors[i].root = int(i);
      defined[i] = !tensors[i].values.empty();
    }
    defined[inputs[0]] = true;
    for (auto &n : nodes) {
      require(alias(n.op) || codes.count(n.op),
              "Unsupported operator: " + n.op);
      for (int i : n.in)
        require(i == -1 || defined[i],
                "Graph is not topologically ordered: " + n.name);
      require(n.in[0] >= 0, "Missing first input");
      for (int o : n.out) {
        require(!defined[o], "Tensor has multiple producers");
        defined[o] = true;
        if (alias(n.op)) {
          require(n.out.size() == 1 &&
                      count(tensors[o].shape) == count(tensors[n.in[0]].shape),
                  "Invalid alias shape");
          tensors[o].root = tensors[n.in[0]].root;
        }
      }
    }
    for (int o : outputs)
      require(defined[o], "Undefined graph output");
    for (size_t k = 0; k < nodes.size(); ++k)
      for (int i : nodes[k].in)
        if (i >= 0)
          tensors[tensors[i].root].last =
              std::max(tensors[tensors[i].root].last, int(k));
    for (int i : outputs)
      tensors[tensors[i].root].last = int(nodes.size());
    int end = 0;
    std::vector<std::pair<int, int>> free;
    auto allocate = [&](int id) {
      auto &t = tensors[id];
      int n = count(t.shape);
      for (auto it = free.begin(); it != free.end(); ++it)
        if (it->second >= n) {
          t.offset = it->first;
          it->first += n;
          it->second -= n;
          if (!it->second)
            free.erase(it);
          return;
        }
      require(int64_t(end) + n <= INT32_MAX / 4, "Arena too large");
      t.offset = end;
      end += n;
    };
    for (size_t i = 0; i < nt; ++i)
      if (!tensors[i].values.empty())
        allocate(int(i));
    allocate(inputs[0]);
    for (size_t k = 0; k < nodes.size(); ++k) {
      auto &n = nodes[k];
      for (int id : n.out) {
        auto &t = tensors[id];
        if (t.root == id)
          allocate(id);
        else
          t.offset = tensors[t.root].offset;
      }
      if (!alias(n.op))
        emit(n);
      for (size_t i = 0; i < nt; ++i) {
        auto &t = tensors[i];
        if (t.root == int(i) && t.values.empty() && t.offset >= 0 &&
            t.last == int(k))
          free.emplace_back(t.offset, count(t.shape));
      }
      std::sort(free.begin(), free.end());
      for (size_t i = 1; i < free.size();)
        if (free[i - 1].first + free[i - 1].second == free[i].first) {
          free[i - 1].second += free[i].second;
          free.erase(free.begin() + i);
        } else
          ++i;
    }
    arena_size = size_t(end);
    std::vector<float> initial(arena_size, 0);
    for (auto &t : tensors)
      if (!t.values.empty())
        std::copy(t.values.begin(), t.values.end(), initial.begin() + t.offset);
    gpu.prepare(arena_size, initial, records, dispatches, shaders);
  }
  void emit(const Node &n) {
    const auto &x = tensors[n.in[0]];
    const auto &xs = x.shape;
    auto input = [&](size_t i) -> const Tensor & {
      require(i < n.in.size() && n.in[i] >= 0, "Missing input for " + n.op);
      return tensors[n.in[i]];
    };
    int op = codes.at(n.op);
    require(n.out.size() == 1 || op == 26 || op == 32,
            "Multiple outputs unsupported for " + n.op);
    int splitStart = 0;
    for (size_t oi = 0; oi < (op == 32 ? 1 : n.out.size()); ++oi) {
      const auto &out = tensors[n.out[oi]];
      const auto &os = out.shape;
      int rank = int(os.size());
      std::array<int, 256> r{};
      r[0] = op;
      r[1] = out.offset;
      r[2] = count(os);
      r[3] = rank;
      int stride = 1;
      for (int d = rank - 1; d >= 0; --d) {
        r[4 + d] = os[d];
        r[12 + d] = stride;
        stride *= os[d];
      }
      r[22] = int(n.in.size());
      for (size_t i = 0; i < n.in.size(); ++i)
        if (n.in[i] >= 0)
          layout(r, 32 + 24 * int(i), input(i));
      auto set = [&](int j, int v) { r[224 + j] = v; };
      uint32_t pipeline = 0, gx = uint32_t((r[2] + 63) / 64), gy = 1, gz = 1;
      if (op <= 18) {
        Shape expected = xs;
        if ((op >= 1 && op <= 4) || (op >= 12 && op <= 15))
          expected = broadcast(xs, input(1).shape);
        if (op == 17)
          expected = broadcast(broadcast(xs, input(1).shape), input(2).shape);
        require(expected == os, "Elementwise shape mismatch: " + n.name);
        if (op == 18) {
          int to = n.integer("to");
          require(to == 1 || to == 6 || to == 7 || to == 9,
                  "Unsupported Cast target");
          set(0, to);
        }
      } else if (op == 19) {
        auto perm = n.integers("perm");
        if (perm.empty()) {
          perm.resize(rank);
          std::iota(perm.rbegin(), perm.rend(), 0);
        }
        require(perm.size() == os.size() && xs.size() == os.size(),
                "Invalid permutation rank");
        std::set<int> used;
        for (int d = 0; d < rank; ++d) {
          require(perm[d] >= 0 && perm[d] < rank &&
                      used.insert(perm[d]).second && os[d] == xs[perm[d]],
                  "Invalid permutation");
          set(d, perm[d]);
        }
      } else if (op == 20) {
        int ax = axis(n.integer("axis"), rank), total = 0;
        set(0, ax);
        for (size_t i = 0; i < n.in.size(); ++i) {
          auto s = input(i).shape;
          require(s.size() == os.size(), "Concat rank mismatch");
          for (int d = 0; d < rank; ++d)
            require(d == ax || s[d] == os[d], "Concat dimensions mismatch");
          total += s[ax];
        }
        require(total == os[ax], "Concat size mismatch");
      } else if (op == 21 || op == 26) {
        require(xs.size() == os.size(), "Slice rank mismatch");
        Shape starts(rank, 0), steps(rank, 1);
        if (op == 26) {
          int ax = axis(n.integer("axis"), rank);
          starts[ax] = splitStart;
          splitStart += os[ax];
          if (oi + 1 == n.out.size())
            require(splitStart == xs[ax], "Split sizes mismatch");
        } else {
          auto st = constants(n.in.at(1));
          auto axes =
              n.in.size() > 3 && n.in[3] >= 0 ? constants(n.in[3]) : Shape{};
          if (axes.empty()) {
            axes.resize(st.size());
            std::iota(axes.begin(), axes.end(), 0);
          }
          auto sp = n.in.size() > 4 && n.in[4] >= 0 ? constants(n.in[4])
                                                    : Shape(st.size(), 1);
          require(st.size() == axes.size() && sp.size() == st.size(),
                  "Slice metadata mismatch");
          for (size_t i = 0; i < st.size(); ++i) {
            int ax = axis(axes[i], rank);
            require(sp[i] != 0, "Zero Slice step");
            int start = st[i];
            if (start < 0)
              start += xs[ax];
            starts[ax] = std::clamp(start, sp[i] > 0 ? 0 : -1,
                                    sp[i] > 0 ? xs[ax] : xs[ax] - 1);
            steps[ax] = sp[i];
          }
        }
        for (int d = 0; d < rank; ++d) {
          int64_t last = int64_t(starts[d]) + int64_t(os[d] - 1) * steps[d];
          require(starts[d] >= 0 && starts[d] < xs[d] && last >= 0 &&
                      last < xs[d],
                  "Slice exceeds source");
          set(d, starts[d]);
          set(8 + d, steps[d]);
        }
      } else if (op == 22)
        require(broadcast(xs, os) == os, "Invalid Expand shape");
      else if (op == 23) {
        require(xs.size() == os.size(), "Tile rank mismatch");
        for (int d = 0; d < rank; ++d)
          require(os[d] % xs[d] == 0, "Tile dimensions mismatch");
      } else if (op == 24 || op == 25) {
        int ax = axis(n.integer("axis"), int(xs.size()));
        set(0, ax);
        auto is = input(1).shape;
        if (op == 24) {
          Shape expected(xs.begin(), xs.begin() + ax);
          expected.insert(expected.end(), is.begin(), is.end());
          expected.insert(expected.end(), xs.begin() + ax + 1, xs.end());
          require(os == expected, "Gather shape mismatch");
        } else {
          require(os == is && os.size() == xs.size(),
                  "GatherElements shape mismatch");
          for (int d = 0; d < rank; ++d)
            require(d == ax || os[d] <= xs[d], "GatherElements bounds");
        }
      } else if (op == 27) {
        const auto &w = input(1).shape;
        require(xs.size() == 4 && w.size() == 4 && os.size() == 4,
                "Conv requires NCHW");
        auto st = n.integers("strides", {1, 1}),
             pd = n.integers("pads", {0, 0, 0, 0}),
             di = n.integers("dilations", {1, 1});
        int gr = n.integer("group", 1);
        require(st.size() == 2 && pd.size() == 4 && di.size() == 2 && gr > 0 &&
                    xs[1] == w[1] * gr && os[1] == w[0] && os[1] % gr == 0 &&
                    xs[0] == os[0],
                "Unsupported Conv dimensions");
        require(n.text("auto_pad", "NOTSET") == "NOTSET",
                "Unsupported Conv auto_pad");
        for (int d = 0; d < 2; ++d) {
          require(st[d] > 0 && di[d] > 0 && pd[d] >= 0 && pd[d + 2] >= 0,
                  "Invalid Conv attributes");
          require(os[d + 2] ==
                      (xs[d + 2] + pd[d] + pd[d + 2] - di[d] * (w[d + 2] - 1) -
                       1) / st[d] +
                          1,
                  "Conv output size mismatch");
          set(d, st[d]);
          set(2 + d, pd[d]);
          set(4 + d, di[d]);
        }
        set(6, gr);
        if (n.in.size() > 2)
          require(count(input(2).shape) == os[1], "Conv bias mismatch");
        if (gr == 1) {
          Tensor left = input(1), right = input(0);
          int m = os[1], cols = os[2] * os[3], k = w[1] * w[2] * w[3];
          left.shape = {1, m, k};
          right.shape = {xs[0], xs[1], xs[2] * xs[3]};
          layout(r, 32, left);
          layout(r, 56, right);
          r[3] = 3;
          r[4] = os[0];
          r[5] = m;
          r[6] = cols;
          r[12] = m * cols;
          r[13] = cols;
          r[14] = 1;
          set(0, m);
          set(1, cols);
          set(2, k);
          set(3, 0);
          set(4, 0);
          set(5, bits(1));
          set(6, bits(1));
          set(7, n.in.size() > 2 ? 2 : 0);
          set(8, 1);
          set(9, xs[2]);
          set(10, xs[3]);
          set(11, os[3]);
          set(12, w[3]);
          set(13, w[2]);
          set(14, st[0]);
          set(15, st[1]);
          set(16, pd[0]);
          set(17, pd[1]);
          set(18, di[0]);
          set(19, di[1]);
          pipeline = 1;
          gx = (cols + 15) / 16;
          gy = (m + 15) / 16;
          gz = os[0];
        }
      } else if (op == 28) {
        const auto &g = input(1).shape;
        require(xs.size() == 4 && g.size() == 4 && g[3] == 2 &&
                    os == Shape({xs[0], xs[1], g[1], g[2]}) && g[0] == xs[0],
                "GridSample shape mismatch");
        require(n.integer("align_corners") == 0 &&
                    n.text("mode", "bilinear") == "bilinear" &&
                    n.text("padding_mode", "zeros") == "zeros",
                "Unsupported GridSample mode");
      } else if (op == 29) {
        require(xs.size() == 4 && os.size() == 4 && xs[0] == os[0] &&
                    xs[1] == os[1],
                "Resize requires NCHW spatial resize");
        require(n.text("mode") == "linear" &&
                    n.text("coordinate_transformation_mode", "half_pixel") ==
                        "half_pixel",
                "Unsupported Resize mode");
      } else if (op == 30 || op == 31) {
        auto axes = n.integers("axes");
        if (n.in.size() > 1 && n.in[1] >= 0)
          axes = constants(n.in[1]);
        if (axes.empty()) {
          axes.resize(xs.size());
          std::iota(axes.begin(), axes.end(), 0);
        }
        for (auto &ax : axes)
          ax = axis(ax, int(xs.size()));
        std::sort(axes.begin(), axes.end());
        int first = int(xs.size() - axes.size());
        for (size_t i = 0; i < axes.size(); ++i)
          require(axes[i] == first + int(i),
                  "Only trailing reduction axes supported");
        Shape expected(xs.begin(), xs.begin() + first);
        if (n.integer("keepdims", 1))
          expected.resize(xs.size(), 1);
        require(expected == os, "Reduction shape mismatch");
        set(0, count(xs) / count(os));
      } else if (op == 32) {
        int ax = axis(n.integer("axis", -1), int(xs.size()));
        auto k = constants(n.in.at(1));
        require(ax == int(xs.size()) - 1 && k.size() == 1 && k[0] > 0 &&
                    k[0] <= xs.back() && n.out.size() == 2,
                "Unsupported TopK dimensions");
        require(n.integer("largest", 1) == 1 && n.integer("sorted", 1) == 1,
                "Unsupported TopK order");
        Shape expected = xs;
        expected.back() = k[0];
        require(os == expected && tensors[n.out[1]].shape == expected,
                "TopK output mismatch");
        set(0, xs.back());
        set(1, k[0]);
        r[20] = tensors[n.out[1]].offset;
        gx = uint32_t((count(xs) + 63) / 64);
      } else if (op >= 33 && op <= 35) {
        pipeline = 1;
        Tensor left = input(0), right = input(1);
        Shape expected;
        int ta = 0, tb = 0, m, k, cols;
        if (op == 33) {
          require(n.text("equation") == "bchw,bnc->bnhw" && xs.size() == 4 &&
                      right.shape.size() == 3,
                  "Unsupported Einsum equation");
          require(xs[0] == right.shape[0] && xs[1] == right.shape[2],
                  "Einsum shape mismatch");
          expected = {xs[0], right.shape[1], xs[2], xs[3]};
          left = input(1);
          right = input(0);
          right.shape = {xs[0], xs[1], xs[2] * xs[3]};
          m = left.shape[1];
          k = xs[1];
          cols = right.shape[2];
          r[3] = 3;
          r[4] = xs[0];
          r[5] = m;
          r[6] = cols;
          r[12] = m * cols;
          r[13] = cols;
          r[14] = 1;
        } else {
          require(left.shape.size() >= 2 && right.shape.size() >= 2,
                  "MatMul requires rank >=2");
          ta = n.integer("transA");
          tb = n.integer("transB");
          require((ta == 0 || ta == 1) && (tb == 0 || tb == 1),
                  "Invalid matrix transpose");
          if (op == 35)
            require(left.shape.size() == 2 && right.shape.size() == 2,
                    "Gemm requires matrices");
          int lr = int(left.shape.size()), rr = int(right.shape.size());
          m = left.shape[lr - 2 + ta];
          k = left.shape[lr - 1 - ta];
          cols = right.shape[rr - 1 - tb];
          require(k == right.shape[rr - 2 + tb],
                  "MatMul inner dimensions mismatch");
          Shape lb(left.shape.begin(), left.shape.end() - 2),
              rb(right.shape.begin(), right.shape.end() - 2);
          expected = broadcast(lb, rb);
          expected.push_back(m);
          expected.push_back(cols);
        }
        require(expected == os, "Matrix output shape mismatch");
        layout(r, 32, left);
        layout(r, 56, right);
        set(0, m);
        set(1, cols);
        set(2, k);
        set(3, ta);
        set(4, tb);
        set(5, bits(n.real("alpha", 1)));
        set(6, bits(n.real("beta", 1)));
        bool bias = op == 35 && n.in.size() > 2 && n.in[2] >= 0;
        if (bias)
          require(broadcast(input(2).shape, os) == os, "Gemm bias mismatch");
        set(7, bias);
        gx = (cols + 15) / 16;
        gy = (m + 15) / 16;
        gz = count(os) / (m * cols);
      } else if (op == 36 || op == 37) {
        require(xs == os && !xs.empty() &&
                    axis(n.integer("axis", -1), int(xs.size())) ==
                        int(xs.size()) - 1,
                "Only last-axis normalization supported");
        if (op == 37) {
          require(count(input(1).shape) == xs.back(),
                  "LayerNorm scale mismatch");
          if (n.in.size() > 2)
            require(count(input(2).shape) == xs.back(),
                    "LayerNorm bias mismatch");
          require(n.integer("stash_type", 1) == 1,
                  "Unsupported LayerNorm precision");
          float eps = n.real("epsilon", 1e-5f);
          require(eps > 0, "Invalid LayerNorm epsilon");
          set(1, bits(eps));
        }
        set(0, xs.back());
        pipeline = 2;
        gx = count(xs) / xs.back();
      } else
        throw std::runtime_error("Unsupported operator " + n.op);
      dispatches.push_back(
          {pipeline, uint32_t(records.size() / 256), gx, gy, gz});
      records.insert(records.end(), r.begin(), r.end());
    }
  }
};
GraphExecutor::GraphExecutor(const std::filesystem::path &m,
                             const std::filesystem::path &s, int d)
    : impl_(std::make_unique<Impl>(d)) {
  impl_->load(m, s);
}
GraphExecutor::~GraphExecutor() = default;
const Shape &GraphExecutor::input_shape() const {
  return impl_->tensors[impl_->inputs[0]].shape;
}
const std::string &GraphExecutor::device_name() const {
  return impl_->gpu.device_name();
}
size_t GraphExecutor::arena_bytes() const { return impl_->arena_size * 4; }
std::vector<Output> GraphExecutor::run(const std::vector<float> &input) {
  auto &p = *impl_;
  auto &input_tensor = p.tensors[p.inputs[0]];
  require(input.size() == size_t(count(input_tensor.shape)), "Input tensor size mismatch");
  for (float x : input)
    require(std::isfinite(x), "Nonfinite model input");
  p.gpu.write(input_tensor.offset, input.data(), input.size());
  p.gpu.run();
  std::vector<Output> out;
  for (int id : p.outputs) {
    auto &output_tensor = p.tensors[id];
    auto data = p.gpu.read(output_tensor.offset, count(output_tensor.shape));
    for (float x : data)
      require(std::isfinite(x), "Nonfinite model output: " + output_tensor.name);
    out.push_back({output_tensor.name, output_tensor.shape, std::move(data)});
  }
  return out;
}
} // namespace rvk
