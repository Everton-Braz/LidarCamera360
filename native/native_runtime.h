#pragma once
// Value types and parameter adapter for the offline build. There is no ROS
// runtime, transport, master, serialization or background callback thread.
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl/io/pcd_io.h>
#include <yaml-cpp/yaml.h>
#include <vikit/pinhole_camera.h>
#include <vikit/equidistant_camera.h>
#ifdef RGB
#undef RGB
#endif
using uint = unsigned int;
namespace native {
inline std::string output_root;
inline int threads=4;
inline YAML::Node config, camera;
inline YAML::Node parameter(const std::string& key) {
  YAML::Node n = YAML::Clone(config);
  size_t begin=0;
  for (;;) {
    auto end=key.find('/',begin);
    n.reset(n[key.substr(begin,end-begin)]);
    if (!n || end==std::string::npos) return n;
    begin=end+1;
  }
}
inline bool load_camera(vk::AbstractCamera*& cam) {
  auto c=camera;
  const auto model=c["cam_model"].as<std::string>();
  if(model=="EquidistantCamera")
    cam=new vk::EquidistantCamera(c["cam_width"].as<int>(), c["cam_height"].as<int>(),c["scale"].as<double>(1),
      c["cam_fx"].as<double>(),c["cam_fy"].as<double>(),c["cam_cx"].as<double>(),c["cam_cy"].as<double>(),
      c["k1"].as<double>(),c["k2"].as<double>(),c["k3"].as<double>(),c["k4"].as<double>());
  else if(model=="Pinhole")
    cam=new vk::PinholeCamera(c["cam_width"].as<int>(),c["cam_height"].as<int>(),c["scale"].as<double>(1),
      c["cam_fx"].as<double>(),c["cam_fy"].as<double>(),c["cam_cx"].as<double>(),c["cam_cy"].as<double>(),
      c["cam_d0"].as<double>(0),c["cam_d1"].as<double>(0),c["cam_d2"].as<double>(0),c["cam_d3"].as<double>(0),c["cam_d4"].as<double>(0));
  else throw std::runtime_error("Unsupported camera model: " + model);
  return cam!=nullptr;
}
}
namespace ros {
struct Time { double value=0; Time()=default; explicit Time(double v):value(v){} double toSec() const{return value;}
 Time& fromSec(double v){value=v;return *this;} static Time now(){return Time(std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count());} };
struct Duration { explicit Duration(double=0){} };
struct TimerEvent {};
struct Publisher { template<class T> void publish(const T&) const {} };
struct Subscriber {}; struct Timer {};
struct NodeHandle { template<class T> void param(const std::string& key,T& out,const T& fallback) const {auto n=native::parameter(key);out=n?n.as<T>():fallback;} };
struct Rate { explicit Rate(double){} void sleep() const{} };
namespace param { template<class T> void param(const std::string& key,T& out,const T& fallback) {
 auto n=native::camera[key.substr(key.find_last_of('/')+1)];out=n?n.as<T>():fallback;
} }
}
namespace std_msgs { struct Header { ros::Time stamp; std::string frame_id; }; }
namespace geometry_msgs {
struct Vector3 { double x=0,y=0,z=0; }; using Point=Vector3;
struct Quaternion { double x=0,y=0,z=0,w=1; };
struct Pose { Point position; Quaternion orientation; };
struct PoseStamped { std_msgs::Header header; Pose pose; };
struct Twist { Vector3 linear,angular; };
}
namespace sensor_msgs {
struct Imu { using Ptr=std::shared_ptr<Imu>; using ConstPtr=std::shared_ptr<const Imu>;
 std_msgs::Header header; geometry_msgs::Vector3 angular_velocity,linear_acceleration; geometry_msgs::Quaternion orientation; };
using ImuConstPtr=Imu::ConstPtr;
struct Image { using Ptr=std::shared_ptr<Image>; using ConstPtr=std::shared_ptr<const Image>; std_msgs::Header header; cv::Mat image; };
using ImageConstPtr=Image::ConstPtr;
struct PointCloud2 { using ConstPtr=std::shared_ptr<const PointCloud2>; std_msgs::Header header; };
namespace image_encodings { inline const std::string BGR8="bgr8"; }
}
namespace livox_ros_driver { struct CustomMsg {using ConstPtr=std::shared_ptr<const CustomMsg>;}; }
namespace nav_msgs {
struct Odometry { std_msgs::Header header; std::string child_frame_id;
 struct {geometry_msgs::Pose pose;std::array<double,36> covariance{};} pose;
 struct {geometry_msgs::Twist twist;std::array<double,36> covariance{};} twist; };
struct Path { std_msgs::Header header; std::vector<geometry_msgs::PoseStamped> poses; };
}
namespace visualization_msgs {
struct Marker {enum {CYLINDER=3,ADD=0};std_msgs::Header header;std::string ns;int id=0,type=0,action=0;
 geometry_msgs::Pose pose;geometry_msgs::Vector3 scale;struct {float a=0,r=0,g=0,b=0;} color;ros::Duration lifetime;};
struct MarkerArray {std::vector<Marker> markers;};
}
namespace tf {
inline geometry_msgs::Quaternion createQuaternionMsgFromRollPitchYaw(double r,double p,double y){
 Eigen::Quaterniond q=Eigen::AngleAxisd(y,Eigen::Vector3d::UnitZ())*Eigen::AngleAxisd(p,Eigen::Vector3d::UnitY())*Eigen::AngleAxisd(r,Eigen::Vector3d::UnitX());return {q.x(),q.y(),q.z(),q.w()};}
}
namespace image_transport {using Publisher=ros::Publisher;struct ImageTransport{};}
namespace cv_bridge {
struct CvImage {std_msgs::Header header;std::string encoding;cv::Mat image;
 sensor_msgs::Image::Ptr toImageMsg() const {auto p=std::make_shared<sensor_msgs::Image>();p->header=header;p->image=image;return p;} };
inline std::shared_ptr<CvImage> toCvCopy(const sensor_msgs::ImageConstPtr& p,const std::string&){auto c=std::make_shared<CvImage>();c->image=p->image.clone();return c;}
}
namespace pcl {template<class T> void toROSMsg(const pcl::PointCloud<T>&,sensor_msgs::PointCloud2&){} }
#define ROS_ERROR(...) do { std::fprintf(stderr,__VA_ARGS__); std::fputc('\n',stderr); } while(0)
#define ROS_WARN(...) ROS_ERROR(__VA_ARGS__)
#define ROS_INFO(...) do {} while(0)
#define ROS_ASSERT(condition) do { if(!(condition)) throw std::runtime_error("Estimator assertion failed: " #condition); } while(0)
