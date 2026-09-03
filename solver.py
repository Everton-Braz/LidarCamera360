# -*- coding: utf-8 -*-
"""Multi-set solving: joint search, cross-validation and least-squares merge."""

import math

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from core import (DataError, DataSet, QUALITY, SOLO_STAGES, R_for, ang_between,
                  build_json, mkR, optimise, optimise_zenith, to_matrices,
                  yaw_scan, zenith_basis)

SCORE_FLOOR = 0.035     # absolute alignment quality floor
SCORE_RATIO = 0.45      # relative to the median over the sets
ZENITH_TOL_DEG = 6.0    # image verticals vs IMU gravity
PAIR_MARGIN_MIN = 0.55  # heading peak, sigma over the rest of the circle
PAIR_SCORE_MIN = 0.030  # heading peak, absolute score
HEADING_TOL_DEG = 25.0  # a heading that disagrees with the other sets
CROSS_KEEP = 0.65       # a foreign answer must keep this much of the own score


class Cancelled(Exception):
    pass


class Solver:
    def __init__(self, pairs, prior, quality="Normal", log=print,
                 progress=lambda f: None, should_stop=lambda: False, frame_pos=0.5):
        self.pairs = pairs
        self.prior = prior
        self.quality = quality
        self.stages = QUALITY[quality]
        self.log = log
        self.progress = progress
        self.should_stop = should_stop
        self.frame_pos = frame_pos
        self.sets = []
        self.stats = {}
        self.contradiction = False
        self.cross = None
        self.cross_best = None
        self.cross_ratio = {}
        self.anchor = None
        self.joint_margin = 0.0
        self.R = self.C = self.T = None

    def _tick(self):
        if self.should_stop():
            raise Cancelled()

    # ------------------------------------------------------------ loading --
    def load(self):
        n = len(self.pairs)
        for i, (bag, video) in enumerate(self.pairs):
            self._tick()
            d = DataSet(i + 1, bag, video)
            self.log("[set %d] %s" % (i + 1, d.name))
            try:
                d.prepare(log=self.log,
                          progress=lambda f, i=i: self.progress(0.05 + 0.40 * (i + f) / n),
                          frame_pos=self.frame_pos)
            except DataError as e:
                d.rejected = True
                d.reject_reason = str(e)
                self.log("   rejected: %s" % e)
                self.sets.append(d)
                continue
            for w in d.warnings:
                self.log("   warning: %s" % w)
            self.sets.append(d)
        if not [d for d in self.sets if not d.rejected]:
            raise DataError("no usable data set")

    # ------------------------------------------------------------ heading --
    def scan_headings(self):
        live = [d for d in self.sets if not d.rejected]
        for i, d in enumerate(live):
            self._tick()
            self.log("[set %d] searching heading ..." % d.idx)
            psi, curve = yaw_scan(d, self.prior)
            s = curve[:, 1]
            far = np.abs(((curve[:, 0] - psi + 180) % 360) - 180) > 20
            d.pair_margin = float((s.max() - s[far].max()) / (s.std() + 1e-9))
            d.pair_score = float(s.max())
            d.heading = float(psi)
            self.log("   heading %+.0f deg | peak %.3f | margin %.2f sigma"
                     % (psi, d.pair_score, d.pair_margin))
            d.pair_suspect = (d.pair_margin < PAIR_MARGIN_MIN
                              or d.pair_score < PAIR_SCORE_MIN)
            if d.pair_suspect:
                d.warnings.append(
                    "the cloud does not lock onto this video (peak %.3f, margin %.2f "
                    "sigma) - bag and mp4 may not be from the same capture"
                    % (d.pair_score, d.pair_margin))
                self.log("   warning: %s" % d.warnings[-1])
            self.progress(0.45 + 0.08 * (i + 1) / max(len(live), 1))

    def check_headings(self):
        """A heading that disagrees with the rest points at a mismatched pair."""
        live = [d for d in self.sets if not d.rejected and hasattr(d, "heading")]
        if len(live) < 2:
            return
        h = np.array([d.heading for d in live])
        med = float(np.median(h))
        for d, dv in zip(live, np.abs(((h - med + 180) % 360) - 180)):
            if dv > HEADING_TOL_DEG:
                d.pair_suspect = True
                d.warnings.append(
                    "heading %+.0f deg disagrees with the other sets (%+.0f deg) - bag "
                    "and mp4 are probably not from the same capture" % (d.heading, med))
                self.log("[set %d] warning: %s" % (d.idx, d.warnings[-1]))

    def joint_heading(self, live):
        """Sweep one shared extrinsic over the full circle against every set.
        A single set can have an ambiguous peak; summed it stands out."""
        with_z = [d for d in live if d.img_zenith is not None]
        anchor = max(with_z or live, key=lambda d: len(d.pts))
        W, H, sg, sub = self.stages[1] if len(self.stages) > 1 else self.stages[0]
        costs = [(d, d.cost(W, H, sg, min(sub, 150000))) for d in live]
        curve = []
        for psi in np.arange(-180, 180, 1.0):
            self._tick()
            rv = np.array([0.0, math.radians(psi), 0.0])
            C0 = self.prior.start_C(anchor.up, mkR(rv, anchor.Rlev))
            curve.append((psi, sum(c.exact(R_for(rv, d, anchor), C0)
                                   for d, c in costs) / len(costs)))
        curve = np.array(curve)
        best = curve[np.argmax(curve[:, 1])]
        s = curve[:, 1]
        far = np.abs(((curve[:, 0] - best[0] + 180) % 360) - 180) > 20
        self.joint_margin = float((s.max() - s[far].max()) / (s.std() + 1e-9))
        self.log("[joint] heading %+.0f deg over %d set(s): score %.4f, margin %.2f sigma"
                 % (best[0], len(live), best[1], self.joint_margin))
        return anchor, float(best[0])

    # --------------------------------------------------- cross-validation --
    @staticmethod
    def _as_rv(x_src, d_src, d_dst):
        """Carry a solution between sets: the extrinsic is one lidar->camera
        matrix, but each set parametrises it against its own levelled frame."""
        R_cl = mkR(np.asarray(x_src[:3]), d_src.Rlev)
        return np.r_[Rot.from_matrix(R_cl @ d_dst.Rlev.T).as_rotvec(),
                     np.asarray(x_src[3:6])]

    def cross_validate(self):
        """M[i][j] = how well set i's answer explains set j."""
        live = [d for d in self.sets if not d.rejected and d.solo is not None]
        if len(live) < 2:
            return None
        W, H, sg, sub = self.stages[min(2, len(self.stages) - 1)]
        costs = {d.idx: d.cost(W, H, sg, sub) for d in live}
        M = np.zeros((len(live), len(live)))
        for a, src in enumerate(live):
            for b, dst in enumerate(live):
                self._tick()
                x = self._as_rv(src.solo, src, dst)
                M[a, b] = costs[dst.idx].exact(mkR(x[:3], dst.Rlev), x[3:6])
        self.cross = dict(idx=[d.idx for d in live], matrix=M.tolist())
        self.log("[cross-check] each answer scored on every set:")
        self.log("        " + "".join("  set%-6d" % d.idx for d in live))
        for a, src in enumerate(live):
            self.log("   set%-3d " % src.idx +
                     "".join("  %8.4f" % M[a, b] for b in range(len(live))))

        own = np.diag(M).copy()
        own[own <= 0] = 1e-6
        ratio = M / own[None, :]
        universal = (ratio >= CROSS_KEEP).all(axis=1)
        if universal.any():
            best = int(np.argmax(np.where(universal, M.sum(1), -np.inf)))
            self.log("   set %d explains every set (worst ratio %.2f)"
                     % (live[best].idx, ratio[best].min()))
            self.cross_best = live[best]
            self.cross_ratio = {d.idx: float(ratio[best, b]) for b, d in enumerate(live)}
            return live[best]

        counts = (ratio >= CROSS_KEEP).sum(1)
        best = int(np.argmax(counts))
        self.cross_best = live[best] if counts[best] >= 2 else None
        self.cross_ratio = {d.idx: float(ratio[best, b]) for b, d in enumerate(live)}
        self.log("   no single answer explains all sets; best (set %d) explains %d of %d"
                 % (live[best].idx, counts[best], len(live)))
        return None

    # -------------------------------------------------------- screening ---
    def screen(self, live):
        for d in live:
            why = []
            med = float(np.median([q.solo_score for q in live]))
            if d.solo_score < SCORE_FLOOR:
                why.append("alignment score %.3f below the usable floor %.2f"
                           % (d.solo_score, SCORE_FLOOR))
            elif med > 0 and d.solo_score < SCORE_RATIO * med:
                why.append("alignment score %.3f far below the other sets (median %.3f)"
                           % (d.solo_score, med))
            if d.zenith_resid is not None and d.zenith_resid > ZENITH_TOL_DEG:
                why.append("image verticals disagree with the IMU by %.1f deg - "
                           "stabilised clip, or not the matching scan" % d.zenith_resid)
            v = self.prior.violation(d.solo[3:6], mkR(d.solo[:3], d.Rlev), d.up)
            if v > 0.05:
                why.append("camera position %.0f cm outside the initial-guess box"
                           % (100 * v))
            if why:
                d.rejected = True
                d.reject_reason = "; ".join(why)
                self.log("[set %d] rejected: %s" % (d.idx, d.reject_reason))

    # ------------------------------------------------------------- merge --
    def lsq_merge(self, live):
        """Rotation by the weighted chordal mean, lever by the weighted mean:
        minimises the sum of squared differences to the accepted sets."""
        w = np.array([max(d.solo_score, 1e-6) for d in live])
        w /= w.sum()
        Rm = Rot.from_matrix(np.stack([mkR(d.solo[:3], d.Rlev) for d in live])
                             ).mean(weights=w).as_matrix()
        U, _, Vt = np.linalg.svd(Rm)
        Rm = U @ Vt
        if np.linalg.det(Rm) < 0:
            U[:, -1] *= -1
            Rm = U @ Vt
        C = np.sum([wi * np.asarray(d.solo[3:6]) for wi, d in zip(w, live)], axis=0)
        ra = [ang_between(mkR(d.solo[:3], d.Rlev), Rm) for d in live]
        rl = [float(np.linalg.norm(np.asarray(d.solo[3:6]) - C)) for d in live]
        self.log("[least squares] merged %d sets: residual %.2f deg / %.0f mm (rms)"
                 % (len(live), math.sqrt(np.mean(np.square(ra))),
                    1000 * math.sqrt(np.mean(np.square(rl)))))
        return Rm, C, ra, rl

    # ----------------------------------------------------------- consensus --
    def consensus(self):
        live = [d for d in self.sets if not d.rejected]

        anchor, psi = self.joint_heading(live)
        self.anchor = anchor
        rv0 = np.array([0.0, math.radians(psi), 0.0])
        x0 = np.r_[rv0, self.prior.start_C(anchor.up, mkR(rv0, anchor.Rlev))]

        # tilt pinned by the scene verticals: four parameters instead of six
        self.log("[joint] fitting the shared extrinsic over %d set(s) ..." % len(live))
        xz, scz = (None, -1.0)
        if zenith_basis(anchor) is not None:
            xz, scz = optimise_zenith(live, math.radians(psi), x0[3:6], self.prior,
                                      self.stages[:max(2, len(self.stages) - 1)],
                                      anchor, log=self.log, tag="tilt ")
        if xz is not None:
            x0 = xz
            self.log("   tilt-constrained fit: score %.4f" % scz)

        x, sc = optimise(live, x0, self.prior, self.stages, log=self.log, tag="joint ",
                         anchor=anchor, progress=lambda f: self.progress(0.55 + 0.25 * f))
        if scz > sc:
            self.log("   the tilt-constrained answer scores higher (%.4f > %.4f), kept"
                     % (scz, sc))
            x, sc = xz, scz
        self.x, self.score = x, sc
        self.R, self.C, self.T = to_matrices(x, live, anchor=anchor)

        # per-set refinement, used to measure agreement and to cross-validate
        for d in live:
            self._tick()
            xd, scd = optimise([d], self._as_rv(x, anchor, d), self.prior,
                               self.stages[1:SOLO_STAGES + 1])
            d.solo, d.solo_score = xd, scd
            d.zenith_resid = d.zenith_residual(xd)
            d.dev_ang = ang_between(mkR(xd[:3], d.Rlev), self.R)
            d.dev_lin = float(np.linalg.norm(xd[3:6] - self.C))
            self.log("   set %d: own score %.3f | %.2f deg, %.0f mm from the joint answer"
                     % (d.idx, scd, d.dev_ang, 1000 * d.dev_lin))

        self.screen(live)
        live = [d for d in live if not d.rejected]
        if not live:
            raise DataError("every data set failed the quality checks")

        if len(live) > 1:
            self.cross_validate()
            if self.cross_best is not None:
                # drop only what the winning answer fails to explain; rejecting
                # by deviation can discard the very sets the check validated
                drop = [d for d in live if self.cross_ratio.get(d.idx, 1.0) < CROSS_KEEP]
                for d in drop:
                    d.rejected = True
                    d.reject_reason = ("its scene is not explained by the answer that "
                                       "fits the others (ratio %.2f)"
                                       % self.cross_ratio[d.idx])
                    self.log("[set %d] rejected: %s" % (d.idx, d.reject_reason))
                if drop:
                    live = [d for d in live if not d.rejected]
                    if not live:
                        raise DataError("no consistent group could be formed")
                    self.log("[joint] refitting over the %d surviving set(s)" % len(live))
                    if anchor not in live:
                        wz = [d for d in live if d.img_zenith is not None]
                        anchor = max(wz or live, key=lambda d: len(d.pts))
                    x, sc = optimise(live, self._as_rv(x, self.anchor, anchor), self.prior,
                                     self.stages, log=self.log, tag="joint ", anchor=anchor)
                    self.anchor, self.x, self.score = anchor, x, sc
                    self.R, self.C, self.T = to_matrices(x, live, anchor=anchor)
                    for d in live:
                        d.dev_ang = ang_between(mkR(d.solo[:3], d.Rlev), self.R)
                        d.dev_lin = float(np.linalg.norm(d.solo[3:6] - self.C))
            else:
                self.contradiction = True
                self.log("[contradiction] no answer explains all sets, "
                         "reporting each separately")

        if len(live) > 1 and not self.contradiction:
            Rm, Cm, ra, rl = self.lsq_merge(live)
            self.stats["lsq_residual_deg"] = float(math.sqrt(np.mean(np.square(ra))))
            self.stats["lsq_residual_m"] = float(math.sqrt(np.mean(np.square(rl))))
            if ang_between(Rm, self.R) < 1.0 and np.linalg.norm(Cm - self.C) < 0.03:
                self.R, self.C = Rm, Cm
                self.T = np.eye(4)
                self.T[:3, :3] = Rm
                self.T[:3, 3] = -Rm @ Cm
                self.log("   least-squares merge adopted")

        ok = live
        bad = [d for d in self.sets if d.rejected]
        spread = max([d.dev_ang for d in ok], default=0.0)
        spread_lin = max([d.dev_lin for d in ok], default=0.0)
        zr = [d.zenith_resid for d in ok if d.zenith_resid is not None]
        h, b, r = self.prior.decompose(self.C, self.R, ok[0].up)
        if self.contradiction:
            unc = "sets contradict each other, see the per-set table"
        elif len(ok) == 1:
            unc = "single set, no cross-check; treat as ~1 deg / ~3 cm"
        else:
            unc = ("angular ~%.1f deg, linear ~%.0f mm (spread of %d sets)"
                   % (max(spread, 0.3), max(1000 * spread_lin, 10), len(ok)))
        self.stats.update(n_ok=len(ok), n_bad=len(bad), spread_deg=spread,
                          spread_lin=spread_lin, joint_margin=self.joint_margin,
                          zenith_resid_deg=float(np.mean(zr)) if zr else 0.0,
                          rig={"up": round(h, 4), "back": round(b, 4), "right": round(r, 4)},
                          uncertainty=unc, score=self.score,
                          contradiction=bool(self.contradiction), cross_check=self.cross)
        return ok, bad

    # --------------------------------------------------------------- run --
    def run(self):
        self.progress(0.02)
        self.load()
        self.scan_headings()
        self.check_headings()
        ok, bad = self.consensus()
        self.progress(1.0)
        return ok, bad

    def json(self):
        return build_json(self.R, self.C, self.T, self.sets, self.stats, self.prior)

    def summary_rows(self):
        rows = []
        for d in self.sets:
            if d.solo is None:
                rows.append((d.name, "FAILED", "-", "-", "-", "-", d.reject_reason))
                continue
            R = mkR(d.solo[:3], d.Rlev)
            e = Rot.from_matrix(R).as_euler("ZYX", degrees=True)
            h, b, r = self.prior.decompose(d.solo[3:6], R, d.up)
            note = d.reject_reason if d.rejected else (
                "%.2f deg, %.0f mm from consensus" % (d.dev_ang, 1000 * d.dev_lin))
            if getattr(d, "pair_suspect", False):
                note = "PAIRING SUSPECT: " + note
            rows.append((d.name,
                         "REJECTED" if d.rejected else "OK",
                         "%.2f / %.2f / %.2f" % (e[0], e[1], e[2]),
                         "%.3f, %.3f, %.3f" % tuple(d.solo[3:6]),
                         "%.1f / %.1f / %.1f" % (100 * h, 100 * b, 100 * r),
                         "%.3f" % d.solo_score, note))
        return rows
