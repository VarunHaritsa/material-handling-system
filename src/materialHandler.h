// General C++ headers
#include <cmath>
#include <climits>
#include <vector>
#include <algorithm>
#include <mutex>

// OpenCV headers
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// ROS include files
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Vector3.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <image_transport/image_transport.h>

// Kinova driver specific include files
#include "kinova_driver/kinova_api.h"
#include "kinova_driver/kinova_arm.h"
#include "kinova_driver/kinova_tool_pose_action.h"
#include "kinova_driver/kinova_joint_angles_action.h"
#include "kinova_driver/kinova_fingers_action.h"
#include "kinova_driver/kinova_joint_trajectory_controller.h"
#include "kinova_driver/kinova_ros_types.h"
#include <actionlib/client/simple_action_client.h>

// Setting up the actionlib
typedef actionlib::SimpleActionClient<kinova_msgs::ArmPoseAction> armPose;
typedef actionlib::SimpleActionClient<kinova_msgs::SetFingersPositionAction> fingerPose;

// Converts euler angles (in rad) to normalized quaternions
geometry_msgs::Quaternion euler_to_quaterion(double roll, double pitch, double yaw){
	float cy = cos(yaw*0.5);
    float sy = sin(yaw*0.5);
    float cp = cos(pitch*0.5);
    float sp = sin(pitch*0.5);
    float cr = cos(roll*0.5);
    float sr = sin(roll*0.5);

    geometry_msgs::Quaternion quat;

    quat.w = cy * cp * cr + sy * sp * sr;
    quat.x = cy * cp * sr - sy * sp * cr;
    quat.y = sy * cp * sr + cy * sp * cr;
    quat.z = sy * cp * cr - cy * sp * sr;

	return std::move(quat);
}

struct Object {
    cv::Point2f centroid;
    double yaw;
};

class MaterialHandler {
public:
    MaterialHandler();
    ~MaterialHandler() = default;
    void runLoop(const ros::Rate& loopRate);

    /* ROS subsciber callback functions */
    void getCurrentPoseCb(const geometry_msgs::PoseStamped& pose);
    void getDepthImageCb(sensor_msgs::Image::ConstPtr depth);
    void getPointCloudCb(sensor_msgs::PointCloud::ConstPtr cloud);
    void getSegmetedImageCb(sensor_msgs::Image::ConstPtr segmented);

    /* Command functions for the arm and fingers */
    void sendGoalToArm(const geometry_msgs::PoseStamped &goal_pose);
    void sendGraspGoal(float fingerClosure);
    void goHome();

private:
    geometry_msgs::Point convertPixelToPointCloud(int u, int v) ;
    cv::Mat binarizeColorImage(const cv::Mat& image);
    bool findObject(const cv::Mat& image);
    bool computeObjectMidpoint(); 
    geomertry_msgs::Pose generateNextWaypoint(double yaw);
    void runMission(const geometry_msgs::Pose& firstPose,
                    const geometry_msgs::Pose& preGraspPose,
                    const geometry_msgs::Pose& graspPose, 
                    const geometry_msgs::Pose& dropPose,
                    const ros::Rate& loopRate);

private:
    image_transport::ImageTransport it;
	ros::NodeHandle nh;
	
	// The next pose is the pre-grasp pose (0.2 m above the detected object)
	geometry_msgs::PoseStamped home, currentPose;
	geometry_msgs::Point midpoint;
	cv_bridge::CvImagePtr segmentedImgPtr, depthImagePtr;
	
    Object object;
    std::unique_ptr<sensor_msgs::PointCloud2> pointCloud {nullptr};
    std::mutex materialHandlerMutex;
};