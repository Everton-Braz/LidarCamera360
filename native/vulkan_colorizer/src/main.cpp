// Vulkan compute host. All GPU allocations are owned by Engine; failed jobs never publish RGB.
#include <vulkan/vulkan.h>
#include <opencv2/opencv.hpp>
#include <array>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <future>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <limits>
namespace fs = std::filesystem;
void check(VkResult r) { if(r != VK_SUCCESS) throw std::runtime_error("Vulkan error " + std::to_string(r)); }
struct Buffer { VkBuffer buffer{}; VkDeviceMemory memory{}; VkDeviceSize size{}; void* mapped{}; };
struct Texture { VkImage image{}; VkDeviceMemory memory{}; VkImageView view{}; Buffer staging; uint32_t w{},h{}; bool initialized{}; };
struct Camera { float R[16]{}, C[4]{}, intrinsics[4]{}, radial[4]{}, prism[4]{}; int32_t grid[2]{960,960}; float scale[2]{}; float minZ{.15f},radius{1620}; uint32_t count{},pad{}; };
static_assert(sizeof(Camera)==160);
struct View { fs::path path; std::array<float,24> values; };
struct Pipeline { VkDescriptorSetLayout layout{}; VkPipelineLayout pipelineLayout{}; VkPipeline pipeline{}; };
struct Engine {
 VkInstance instance{}; VkPhysicalDevice physical{}; VkDevice device{}; VkQueue queue{}; uint32_t family{};
 VkCommandPool pool{}; VkCommandBuffer command{}; VkFence fence{}; VkDescriptorPool descriptors{}; VkSampler sampler{};
 VkPhysicalDeviceProperties properties{}; std::vector<Buffer> buffers; std::array<Texture,2> textures{}; std::vector<Pipeline> pipelines;
 ~Engine() {
  if(device) {
   vkDeviceWaitIdle(device);
   for(auto p:pipelines) { if(p.pipeline)vkDestroyPipeline(device,p.pipeline,nullptr); if(p.pipelineLayout)vkDestroyPipelineLayout(device,p.pipelineLayout,nullptr); if(p.layout)vkDestroyDescriptorSetLayout(device,p.layout,nullptr); }
   for(auto& t:textures) { if(t.view)vkDestroyImageView(device,t.view,nullptr); if(t.image)vkDestroyImage(device,t.image,nullptr); if(t.memory)vkFreeMemory(device,t.memory,nullptr); }
   for(auto b:buffers) { if(b.mapped)vkUnmapMemory(device,b.memory); vkDestroyBuffer(device,b.buffer,nullptr); vkFreeMemory(device,b.memory,nullptr); }
   if(sampler)vkDestroySampler(device,sampler,nullptr); if(descriptors)vkDestroyDescriptorPool(device,descriptors,nullptr);
   if(fence)vkDestroyFence(device,fence,nullptr); if(pool)vkDestroyCommandPool(device,pool,nullptr); vkDestroyDevice(device,nullptr);
  }
  if(instance)vkDestroyInstance(instance,nullptr);
 }
 void init(int requested) {
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.pApplicationName="Raven Colorizer"; app.apiVersion=VK_API_VERSION_1_1;
  VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ci.pApplicationInfo=&app; check(vkCreateInstance(&ci,nullptr,&instance));
  uint32_t n=0; check(vkEnumeratePhysicalDevices(instance,&n,nullptr)); std::vector<VkPhysicalDevice> devices(n); check(vkEnumeratePhysicalDevices(instance,&n,devices.data()));
  int best=-1;
  for(uint32_t i=0;i<n;++i) {
   if(requested>=0 && i!=uint32_t(requested))continue;
   VkPhysicalDeviceProperties prop; vkGetPhysicalDeviceProperties(devices[i],&prop);
   uint32_t qn=0; vkGetPhysicalDeviceQueueFamilyProperties(devices[i],&qn,nullptr); std::vector<VkQueueFamilyProperties> qs(qn); vkGetPhysicalDeviceQueueFamilyProperties(devices[i],&qn,qs.data());
   for(uint32_t q=0;q<qn;++q) if(qs[q].queueFlags & VK_QUEUE_COMPUTE_BIT) {
    int rank=prop.deviceType==VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU?100:1;
    if(rank>best) { best=rank; physical=devices[i]; family=q; properties=prop; } break;
   }
  }
  if(!physical)throw std::runtime_error("No compatible Vulkan compute device");
  std::cout<<"Vulkan device: "<<properties.deviceName<<std::endl;
  float priority=1; VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qi.queueFamilyIndex=family; qi.queueCount=1; qi.pQueuePriorities=&priority;
  VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; di.queueCreateInfoCount=1; di.pQueueCreateInfos=&qi; check(vkCreateDevice(physical,&di,nullptr,&device)); vkGetDeviceQueue(device,family,0,&queue);
  VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; pi.queueFamilyIndex=family; pi.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; check(vkCreateCommandPool(device,&pi,nullptr,&pool));
  VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; ai.commandPool=pool; ai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount=1; check(vkAllocateCommandBuffers(device,&ai,&command));
  VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; check(vkCreateFence(device,&fi,nullptr,&fence));
  VkDescriptorPoolSize sizes[]={{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,8},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,32},{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,8}};
  VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; dpi.maxSets=8; dpi.poolSizeCount=3; dpi.pPoolSizes=sizes; check(vkCreateDescriptorPool(device,&dpi,nullptr,&descriptors));
  VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO}; si.magFilter=si.minFilter=VK_FILTER_LINEAR; si.mipmapMode=VK_SAMPLER_MIPMAP_MODE_NEAREST; si.addressModeU=si.addressModeV=si.addressModeW=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; check(vkCreateSampler(device,&si,nullptr,&sampler));
 }
 uint32_t memoryType(uint32_t bits,VkMemoryPropertyFlags flags) {
  VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(physical,&mp);
  for(uint32_t i=0;i<mp.memoryTypeCount;++i)if((bits&(1u<<i)) && (mp.memoryTypes[i].propertyFlags&flags)==flags)return i;
  throw std::runtime_error("No compatible memory type");
 }
 Buffer buffer(VkDeviceSize size,VkBufferUsageFlags usage,bool host=false) {
  if(!size)throw std::runtime_error("Empty buffer");
  if((usage&VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) && size>properties.limits.maxStorageBufferRange)throw std::runtime_error("Cloud exceeds GPU storage buffer limit; use CPU fallback");
  Buffer b; b.size=size; VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; ci.size=size; ci.usage=usage; check(vkCreateBuffer(device,&ci,nullptr,&b.buffer));
  buffers.push_back(b); auto& owned=buffers.back();
  VkMemoryRequirements req; vkGetBufferMemoryRequirements(device,b.buffer,&req); VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize=req.size; ai.memoryTypeIndex=memoryType(req.memoryTypeBits,host?(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT):VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT); check(vkAllocateMemory(device,&ai,nullptr,&owned.memory)); check(vkBindBufferMemory(device,b.buffer,owned.memory,0));
  if(host)check(vkMapMemory(device,owned.memory,0,size,0,&owned.mapped)); return owned;
 }
 void begin() { check(vkResetCommandBuffer(command,0)); VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; bi.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; check(vkBeginCommandBuffer(command,&bi)); }
 void submit() { check(vkEndCommandBuffer(command)); check(vkResetFences(device,1,&fence)); VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&command; check(vkQueueSubmit(queue,1,&si,fence)); }
 void wait() { check(vkWaitForFences(device,1,&fence,VK_TRUE,120000000000ull)); }
 void barrier(VkPipelineStageFlags src,VkAccessFlags sa,VkPipelineStageFlags dst,VkAccessFlags da) {
  VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; b.srcAccessMask=sa; b.dstAccessMask=da; vkCmdPipelineBarrier(command,src,dst,0,1,&b,0,nullptr,0,nullptr);
 }
 void copy(Buffer src,Buffer dst) { VkBufferCopy c{0,0,dst.size}; vkCmdCopyBuffer(command,src.buffer,dst.buffer,1,&c); }
 void texture(Texture& t,uint32_t w,uint32_t h) {
  if(w>properties.limits.maxImageDimension2D||h>properties.limits.maxImageDimension2D)throw std::runtime_error("Image exceeds GPU texture limit");
  t.w=w;t.h=h;t.staging=buffer(VkDeviceSize(w)*h*4,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,true);
  VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO}; ci.imageType=VK_IMAGE_TYPE_2D; ci.format=VK_FORMAT_R8G8B8A8_UNORM; ci.extent={w,h,1}; ci.mipLevels=ci.arrayLayers=1; ci.samples=VK_SAMPLE_COUNT_1_BIT; ci.tiling=VK_IMAGE_TILING_OPTIMAL; ci.usage=VK_IMAGE_USAGE_TRANSFER_DST_BIT|VK_IMAGE_USAGE_SAMPLED_BIT; check(vkCreateImage(device,&ci,nullptr,&t.image));
  VkMemoryRequirements req; vkGetImageMemoryRequirements(device,t.image,&req); VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize=req.size; ai.memoryTypeIndex=memoryType(req.memoryTypeBits,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT); check(vkAllocateMemory(device,&ai,nullptr,&t.memory)); check(vkBindImageMemory(device,t.image,t.memory,0));
  VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO}; vi.image=t.image; vi.viewType=VK_IMAGE_VIEW_TYPE_2D; vi.format=ci.format; vi.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1}; check(vkCreateImageView(device,&vi,nullptr,&t.view));
 }
 void upload(Texture& t) {
  VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}; b.image=t.image; b.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1}; b.srcQueueFamilyIndex=b.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
  b.oldLayout=t.initialized?VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:VK_IMAGE_LAYOUT_UNDEFINED; b.newLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; b.srcAccessMask=t.initialized?VK_ACCESS_SHADER_READ_BIT:0; b.dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;
  vkCmdPipelineBarrier(command,t.initialized?VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT:VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,nullptr,0,nullptr,1,&b);
  VkBufferImageCopy c{}; c.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1}; c.imageExtent={t.w,t.h,1}; vkCmdCopyBufferToImage(command,t.staging.buffer,t.image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&c);
  b.oldLayout=b.newLayout;b.newLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;b.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;b.dstAccessMask=VK_ACCESS_SHADER_READ_BIT; vkCmdPipelineBarrier(command,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,0,nullptr,0,nullptr,1,&b);t.initialized=true;
 }
 Pipeline pipeline(const fs::path& path,std::vector<VkDescriptorType> types,bool push=false) {
  pipelines.emplace_back(); auto& p=pipelines.back();
  std::vector<VkDescriptorSetLayoutBinding> bindings;
  for(uint32_t i=0;i<types.size();++i)bindings.push_back({i,types[i],1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr});
  VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};li.bindingCount=uint32_t(bindings.size());li.pBindings=bindings.data();check(vkCreateDescriptorSetLayout(device,&li,nullptr,&p.layout));
  VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT,0,8}; VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};pli.setLayoutCount=1;pli.pSetLayouts=&p.layout;if(push){pli.pushConstantRangeCount=1;pli.pPushConstantRanges=&range;}check(vkCreatePipelineLayout(device,&pli,nullptr,&p.pipelineLayout));
  std::ifstream file(path,std::ios::binary|std::ios::ate); if(!file)throw std::runtime_error("Missing shader: "+path.string()); auto size=file.tellg(); if(size<=0||size%4!=0)throw std::runtime_error("Invalid shader");std::vector<uint32_t> code(size_t(size)/4);file.seekg(0);file.read(reinterpret_cast<char*>(code.data()),size);
  VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};mi.codeSize=size_t(size);mi.pCode=code.data();VkShaderModule module;check(vkCreateShaderModule(device,&mi,nullptr,&module));
  VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};ci.layout=p.pipelineLayout;ci.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};ci.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;ci.stage.module=module;ci.stage.pName="main";auto result=vkCreateComputePipelines(device,VK_NULL_HANDLE,1,&ci,nullptr,&p.pipeline);vkDestroyShaderModule(device,module,nullptr);check(result);return p;
 }
 VkDescriptorSet set(Pipeline p,std::vector<Buffer> bs,int textureSlot=-1) {
  VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};ai.descriptorPool=descriptors;ai.descriptorSetCount=1;ai.pSetLayouts=&p.layout;VkDescriptorSet ds;check(vkAllocateDescriptorSets(device,&ai,&ds));
  for(uint32_t i=0;i<bs.size();++i) {
   VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};w.dstSet=ds;w.dstBinding=i;w.descriptorCount=1;
   VkDescriptorBufferInfo bi{bs[i].buffer,0,bs[i].size};VkDescriptorImageInfo ii{};
   if(textureSlot>=0 && i==3) {ii={sampler,textures[textureSlot].view,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};w.descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;w.pImageInfo=&ii;}
   else {w.descriptorType=(textureSlot>=0&&i==0)?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;w.pBufferInfo=&bi;}
   vkUpdateDescriptorSets(device,1,&w,0,nullptr);
  }
  return ds;
 }
 void dispatch(Pipeline p,VkDescriptorSet ds,uint32_t count) {
  uint32_t groups=(count+255)/256;
  if(groups>properties.limits.maxComputeWorkGroupCount[0])throw std::runtime_error("Cloud exceeds dispatch limit");
  vkCmdBindPipeline(command,VK_PIPELINE_BIND_POINT_COMPUTE,p.pipeline);vkCmdBindDescriptorSets(command,VK_PIPELINE_BIND_POINT_COMPUTE,p.pipelineLayout,0,1,&ds,0,nullptr);vkCmdDispatch(command,groups,1,1);
 }
};
void read(std::ifstream& f,void* p,size_t n) {if(!f.read(static_cast<char*>(p),n))throw std::runtime_error("Truncated job");}
cv::Mat decode(const fs::path& path) {
 std::ifstream f(path,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error("Missing JPEG: "+path.u8string());auto size=f.tellg();if(size<=0)throw std::runtime_error("Empty JPEG");std::vector<unsigned char> bytes(static_cast<size_t>(size));f.seekg(0);read(f,bytes.data(),bytes.size());
 auto bgr=cv::imdecode(bytes,cv::IMREAD_COLOR);if(bgr.empty())throw std::runtime_error("Invalid JPEG");cv::Mat rgba;cv::cvtColor(bgr,rgba,cv::COLOR_BGR2RGBA);return rgba;
}
int run(int argc,char** argv) {
 fs::path job,out,shaders=fs::absolute(fs::u8path(argv[0])).parent_path()/"shaders";int dev=-1;bool probe=false;
 for(int i=1;i<argc;++i){std::string arg=argv[i];if(arg=="--help"){std::cout<<"vulkan_colorizer --job RVC1.bin --out RGB8.bin [--device N] [--shaders DIR]\n--probe: verify compute device and shaders\n";return 0;}if(arg=="--probe"){probe=true;continue;}if(i+1>=argc)throw std::runtime_error("Missing argument");std::string value=argv[++i];if(arg=="--job")job=fs::u8path(value);else if(arg=="--out")out=fs::u8path(value);else if(arg=="--shaders")shaders=fs::u8path(value);else if(arg=="--device")dev=std::stoi(value);else throw std::runtime_error("Unknown argument: "+arg);}
 if(!probe&&(job.empty()||out.empty()))throw std::runtime_error("--job and --out are required");
 Engine e;e.init(dev);auto U=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,S=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,T=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
 auto z=e.pipeline(shaders/"occlusion_zbuf.spv",{U,S,S,T,S,S});auto color=e.pipeline(shaders/"colorize_consensus.spv",{U,S,S,T,S,S});auto resolve=e.pipeline(shaders/"resolve_consensus.spv",{S,S,S},true);if(probe)return 0;
 std::ifstream f(job,std::ios::binary);char magic[4];uint32_t n,nviews;read(f,magic,4);read(f,&n,4);read(f,&nviews,4);if(std::memcmp(magic,"RVC1",4)||!n||!nviews||nviews>1000000)throw std::runtime_error("Invalid job header");
 auto points=e.buffer(VkDeviceSize(n)*16,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT);auto staging=e.buffer(points.size,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,true);read(f,staging.mapped,size_t(points.size));
 std::vector<View> views(nviews);for(auto& v:views){uint32_t length;read(f,&length,4);if(!length||length>32768)throw std::runtime_error("Invalid path length");std::string s(length,'\0');read(f,s.data(),length);v.path=fs::u8path(s);read(f,v.values.data(),96);}
 auto scores=e.buffer(VkDeviceSize(n)*16,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT);auto colors=e.buffer(scores.size,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT);auto depth=e.buffer(960*960*4,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT);auto output=e.buffer(VkDeviceSize(n)*4,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT);auto readback=e.buffer(output.size,VK_BUFFER_USAGE_TRANSFER_DST_BIT,true);
 std::array<Buffer,2> ubos{e.buffer(sizeof(Camera),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,true),e.buffer(sizeof(Camera),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,true)};
 cv::setNumThreads(2);auto first=decode(views[0].path);for(auto& t:e.textures)e.texture(t,first.cols,first.rows);
 std::array<VkDescriptorSet,2> zs,cs;for(int i=0;i<2;++i){std::vector<Buffer> b{ubos[i],points,depth,{},scores,colors};zs[i]=e.set(z,b,i);cs[i]=e.set(color,b,i);}auto rs=e.set(resolve,{scores,colors,output});
 e.begin();e.copy(staging,points);vkCmdFillBuffer(e.command,scores.buffer,0,scores.size,0);vkCmdFillBuffer(e.command,colors.buffer,0,colors.size,0);e.barrier(VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_TRANSFER_WRITE_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT);e.submit();e.wait();
 std::future<cv::Mat> pending;bool inFlight=false;
 for(uint32_t i=0;i<nviews;++i) {
  auto rgba=i==0?first:pending.get();first.release();auto& tex=e.textures[i%2];if(uint32_t(rgba.cols)!=tex.w||uint32_t(rgba.rows)!=tex.h)throw std::runtime_error("Mixed image dimensions unsupported");
  std::memcpy(tex.staging.mapped,rgba.data,size_t(tex.staging.size));
  if(i+1<nviews)pending=std::async(std::launch::async,[&,i]{return decode(views[i+1].path);});
  Camera camera;auto& v=views[i].values;for(int r=0;r<3;++r)for(int c=0;c<3;++c)camera.R[c*4+r]=v[r*3+c];camera.R[15]=1;std::copy_n(v.data()+9,3,camera.C);std::copy_n(v.data()+12,4,camera.intrinsics);
  camera.radial[0]=v[16];camera.radial[1]=v[17];camera.radial[2]=v[20];camera.radial[3]=v[21];camera.prism[0]=v[18];camera.prism[1]=v[19];camera.prism[2]=v[22];camera.prism[3]=v[23];camera.scale[0]=tex.w/960.f;camera.scale[1]=tex.h/960.f;camera.radius=1620.f*std::min(tex.w,tex.h)/3840.f;camera.count=n;std::memcpy(ubos[i%2].mapped,&camera,sizeof(camera));
  if(inFlight)e.wait();e.begin();
  e.barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_SHADER_READ_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT|VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_ACCESS_TRANSFER_WRITE_BIT|VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT);
  e.upload(tex);vkCmdFillBuffer(e.command,depth.buffer,0,depth.size,0xffffffff);e.barrier(VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_TRANSFER_WRITE_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT);
  e.dispatch(z,zs[i%2],n);e.barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_ACCESS_SHADER_WRITE_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_ACCESS_SHADER_READ_BIT);e.dispatch(color,cs[i%2],n);e.submit();inFlight=true;
  if((i+1)%25==0||i+1==nviews)std::cout<<"Vulkan frames "<<i+1<<"/"<<nviews<<std::endl;
 }
 e.wait();e.begin();e.barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_ACCESS_SHADER_WRITE_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_ACCESS_SHADER_READ_BIT);struct {uint32_t count;float threshold;} push{n,45};vkCmdPushConstants(e.command,resolve.pipelineLayout,VK_SHADER_STAGE_COMPUTE_BIT,0,8,&push);e.dispatch(resolve,rs,n);e.barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_ACCESS_SHADER_WRITE_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_TRANSFER_READ_BIT);e.copy(output,readback);e.barrier(VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_TRANSFER_WRITE_BIT,VK_PIPELINE_STAGE_HOST_BIT,VK_ACCESS_HOST_READ_BIT);e.submit();e.wait();
 std::ofstream result(out,std::ios::binary);result.write(static_cast<char*>(readback.mapped),readback.size);if(!result)throw std::runtime_error("Cannot write output");return 0;
}
#ifdef _WIN32
#include <windows.h>
int wmain(int argc,wchar_t** wargv){std::vector<std::string> args;for(int i=0;i<argc;++i){int n=WideCharToMultiByte(CP_UTF8,0,wargv[i],-1,nullptr,0,nullptr,nullptr);std::string s(n,'\0');WideCharToMultiByte(CP_UTF8,0,wargv[i],-1,s.data(),n,nullptr,nullptr);s.pop_back();args.push_back(s);}std::vector<char*> argv;for(auto& s:args)argv.push_back(s.data());try{return run(argc,argv.data());}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
#else
int main(int argc,char** argv){try{return run(argc,argv);}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
#endif
