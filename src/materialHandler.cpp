#include "materialHandler.h"

#define QUIET_NAN std::numeric_limits<double>::quiet_NaN()

// Image Dimensions
#define IMAGE_WIDTH static_cast<int>(640)
#define IMAGE_HEIGHT static_cast<int>(480)
// Binary thesholding values
#define THRESH_MAX_VAL static_cast<double>(255)
#define BINARIZE_THRESH static_cast<double>(200)
// Canny edge detector parameters
#define CANNY_THRESH_1 static_cast<double>(40)
#define CANNY_THRESH_2 static_cast<double>(100)
#define CANNY_APERTURE_SIZE static_cast<int>(3)
// Hough line extractor
#define HOUGH_RHO static_cast<double>(1)
#define HOUGH_THETA static_cast<double>(M_PI/180.0f)
#define HOUGH_THRESHOLD static_cast<int>(40)

MaterialHandler::MaterialHandler() : nh(), it(nh){
    pointCloud = std::make_unique<sensor_msgs::PointCloud2>();
    bool isHomePoseSet = false;
    currentPose.pose.position = cv::Point(QUIET_NAN, QUIET_NAN, QUIET_NAN);
    
    // Setup the ROS subscribers and callbacks
    nh.subscribe("/j2n6s300_driver/out/tool_pose", 1, &MaterialHandler::getCurrentPoseCb, this);
	it.subscribe("/enet/seg", 1, &MaterialHandler::getSegmetedImageCb, this);
	it.subscribe("/zed/depth/depth_registered", 1, &MaterialHandler::getDepthImageCb, this);
	nh.subscribe("/zed/point_cloud/cloud_registered", 1, &MaterialHandler::getPointCloudCb, this);

    // Set the home position for the arm
    while (std::isnan(currentPose.position.x) || std::isnan(currentPose.position.y) || std::isnan(currentPose.position.z)){
        ros::Duration(2).sleep(); // Sleep for 2s
    }
    home = currentPose;
    
	ROS_INFO_STREAM("All subscribers and callbacks setup for material handling");
}

void MaterialHandler::runLoop(const ros::Rate& loopRate) {
    // Compute the midpoint of the object of interest
    // If no object in image or the midpoint is invalid (outside the camera's field of view), return
    if(!computeObjectMidpoint()) return;

    geometry_msgs::Pose firstPose, preGraspPose, graspPose dropPose;
    preGraspPose = generateNextWaypoint(object.yaw);
    graspPose = preGraspPose;
    graspPose.position.z -= 0.1;
    // This is known first pose computed by trial and error 
    firstPose.position.x = 0.4;
	firstPose.position.y = 0.05;
	firstPose.position.z = 0.3;
    firstPose.orientation = euler_to_quaterion(static_cast<double>(M_PI), static_cast<double>(0.0f), static_cast<double>(-1.0f));
    // Pre set drop pose (due to kinematic restrictions of the arm)
    dropPose.position.x = 0.4;
	dropPose.position.y = -0.5;                                               
    dropPose.position.z = 0.3;
    dropPose.orientation = firstPose.pose.orientation;

    // Send the mission to the arm
    runMission(firstPose, preGraspPose, graspPose, dropPose, loopRate);
}

/* ROS subscriber callbacks */
void MaterialHandler::getCurrentPoseCb(const geometry_msgs::PoseStamped& pose) {
    std::scoped_lock lock(materialHandlerMutex);
    currentPose = pose;
}

void MaterialHandler::getDepthImageCb(sensor_msgs::Image::ConstPtr depth) {
    std::scoped_lock lock(materialHandlerMutex);
    depthImagePtr = cv_bridge::toCvCopy(depth_img, sensor_msgs::image_encodings::TYPE_32FC1);
}

void MaterialHandler::getPointCloudCb(sensor_msgs::PointCloud::ConstPtr cloud) {
    std::scoped_lock lock(materialHandlerMutex);
    pointCloud = cloud;
}

void MaterialHandler::getSegmetedImageCb(sensor_msgs::Image::ConstPtr segmented) {
    std::scoped_lock lock(materialHandlerMutex);
    segmentedImgPtr = cv_bridge::toCvCopy(segmented, sensor_msgs::image_encodings::BGR8);
}

/* Command functions */
void MaterialHandler::sendGoalToArm(geometry_msgs::PoseStamped &goal_pose){
    // Setup ROS service for the arm controller (in-built in the Kinova ROS driver)
    armPose client("/j2n6s300_driver/pose_action/tool_pose", true);
    kinova_msgs::ArmPoseGoal goal;
    client.waitForServer();

    goal_pose.header.stamp = ros::Time::now();
    goal_pose.header.frame_id = "j2n6s300_link_base";
    goal.pose = goal_pose;
    client.sendGoal(goal);
}

void MaterialHandler::sendGraspGoal(float closure){
    // Setup ROS service for the finger controller
    fingerPose client("/j2n6s300_driver/fingers_action/finger_positions", true);
    kinova_msgs::SetFingersPositionGoal goal;
    client.waitForServer();

    if (closure < 0) closure = 0;
    else closure = std::min(closure, float(55));

    goal.fingers.finger1 = closure;
    goal.fingers.finger2 = goal.fingers.finger1;
    goal.fingers.finger3 = goal.fingers.finger2;
    client.sendGoal(goal);
}

void MaterialHandler::goHome() {
    home.header.stamp = ros::Time::now();
    home.header.frame_id = "j2n6s300_link_base";

    sendGraspGoal(0); // Completely open fingers
    sendGoalToArm(home);
    ROS_INFO_STREAM("Back home");
}

/* Private methods that execute specific tasks */
geometry_msgs::Point MaterialHandler::convertPixelToPointCloud(int u, int v) {
    int w = pointCloud->width;
    int h = pointCloud->height;

    int arrayPos = v*pointCloud->row_step + u*pointCloud->point_step;
    int arrayPosX = arrayPos + pointCloud->fields[1].offset; // X has an offset of 0
    int arrayPosY = arrayPos + pointCloud->fields[2].offset; // Y has an offset of 4
    int arrayPosZ = arrayPos + pointCloud->fields[0].offset; // Z has an offset of 8

    geometry_msgs::Point p;
    // put data into the point p
    p.x = -1 * static_cast<float>(pointCloud->data[arrayPosX]);
    p.y = -1 * static_cast<float>(pointCloud->data[arrayPosY]); 
    // Use the depth image from z for accurate depth estimation
    p.z = depthPtr->image.at<float>(u,v);
    return std::move(p);
}

cv::Mat MaterialHandler::binarizeColorImage(const cv::Mat& image) {
    cv::Mat gray, thresh;
    cvtColor(image, gray, cv::CV_RGB2GRAY);
	threshold(gray, thresh, BINARIZE_THRESH, THRESH_MAX_VAL, cv::THRESH_BINARY);
    return std::move(thresh);
}

bool MaterialHandler::findObject(const cv::Mat& image){
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    std::vector<cv::Point> objectContour;
    cv::Mat canny;

    // To find the pixel centroids of various objects in the frame
    findContours(binarizeColorImage(image), 
                                    contours, 
                                    hierarchy, 
                                    cv::CV_RETR_TREE, 
                                    cv::CV_CHAIN_APPROX_SIMPLE, 
                                    cv::Point(0,0));

    // If the contour list is empty, nothing to do ==> No objects in the segmented image
    if(!contours.size()) return false;

    // This calculates the centroid of the object closest to the origin of the camera
    for(int i=0;i<contours.size();i++){
        cv::Moments m = moments(contours[i], false);
        object.centroid = Point2f(m.m10/m.m00, m.m01/m.m00);
        if(!isnan(object.centroid.x) || !isnan(object.centroid.y)) {
            // Compute the orientation of the object in the range (-180, 180)
            auto boundingRect = cv::minAreaRect(objectContour);
            object.yaw = (boundingRect.yaw > 180) ? boundingRect.yaw - 360 : boundingRect.yaw;
            object.yaw *= M_PI / 180.0f; // Convert the yaw to radians 
            return true;
        }
    }
    return false;
}

bool MaterialHandler::computeObjectMidpoint() {
    // Get the latest segmented image from the ROS topic
    cv::Mat segmentedImage = segmementedImagePtr->image;
    // Pre process the image and get the orientation of the object
    if(!findObject(image)) return false;
    if(std::isnan(object.centroid.x) || std::isnan(object.centroid.y)) return false;
    if(abs(object.centroid.x) > IMAGE_WIDTH || abs(object.centroid.y) > IMAGE_HEIGHT) return false;
    // Compute object midpoint
    midpoint = convertPixelToPointCloud(object.centroid.x, object.centroid.y);
    return true;
}

geometry_msgs::Pose MaterialHandler::generateNextWaypoint(double yaw) {
    geometry_msgs::Pose goal;

    // Trasformation from the camera frame to the hand frame
    float mean_x_pose, mean_y_pose, mean_z_pose;
    mean_x_pose = 0.05 - pts.x;
    mean_y_pose = pts.y + 0.26;
    mean_z_pose = 0.77 - pts.z;

    // Arm has a mean error of 9cm in the x direction
    // The z-coordinate of the waypoint is 10cm above the object
    goal.position.x = mean_x_pose - 0.09;
    goal.position.y = mean_y_pose;
    // Capping off the z value 
    // This is to ensure that the arm doesn't go too low to a pose where the fingers can't close
    // This is primarily to overcome the depth estimation issue of the zed camera
    goal.position.z = std::max(mean_z_pose + 0.1, -0.1635f);

    // Align the robotic arm to the yaw of the object for easy picking		
    goal.orientation = euler_to_quaterion(0, 0, yaw);
    return std::move(goal);
}

void MaterialHandler::runMission(const geometry_msgs::Pose& firstPose,
                                 const geometry_msgs::Pose& preGraspPose,
                                 const geometry_msgs::Pose& graspPose, 
                                 const geometry_msgs::Pose& dropPose,
                                 const ros::Rate& loopRate) {

    // Go to the first pose
    ROS_INFO_STREAM("Moving to initial pose");
    sendGoalToArm(firstPose);
    loopRate.sleep();

    // Go to pre-grasp pose
    ROS_INFO_STREAM("Moving to pre-grasp pose");
    sendGoalToArm(preGraspPose);
    loopRate.sleep();

    // Go to grasp pose
    ROS_INFO_STREAM("Moving to grasp the object");
    sendGoalToArm(graspPose);
    loopRate.sleep();

    // Pick up the object
    ROS_INFO_STREAM("Picking up the object");
	sendGraspGoal(55); // Completely closed fingers
	loop_rate.sleep();

    // Go to grasp pose
    ROS_INFO_STREAM("Moving to grasp the object");
    sendGoalToArm(graspPose);
    loopRate.sleep();

    // Move to drop pose
	ROS_INFO_STREAM("Moving to drop pose");
    sendGoalToArm(dropPose);
    loopRate.sleep();

    // Move back to home
    ROS_INFO_STREAM("Finished pick and place, going back home")
    goHome();
    loopRate.sleep();
}

int main(int argc, char** argv){
	ros::init(argc, argv, "Jaco");
	ros::Rate loop_rate(0.2);

	// System initialization
	MaterialHandler jacoArm
	ros::Duration(5).sleep();

	while(!ros::shutdown()){
		ros::spinOnce();
		jacoArm.runLoop(loop_rate);
        loop_rate.sleep();
	}
	return 0;
}