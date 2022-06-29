#pragma once
#include <ros/ros.h>
#include <opencv2/highgui/highgui.hpp>

extern int ROW;
extern int COL;
extern int FOCAL_LENGTH;
const int NUM_OF_CAM = 1; // 1: monocular, 2: stereo


extern std::string IMAGE_TOPIC;
extern std::string IMU_TOPIC;
extern std::string FISHEYE_MASK;
extern std::vector<std::string> CAM_NAMES;
extern int MAX_CNT; // max number of features to be tracked in one frame
extern int MIN_DIST; // minimal distance between two keypoints to be connected
extern int WINDOW_SIZE; // 20 frames are needed to compute the feature point position in the previous frame
extern int FREQ; // // if the frequency is 0, the frequency is set to the maximum possible value
extern double F_THRESHOLD; // Threshold for feature point matching
extern int SHOW_TRACK; // 1: show, 0: hide
extern int STEREO_TRACK; // false: monocular, true: stereo
extern int EQUALIZE;
extern int FISHEYE;
extern bool PUB_THIS_FRAME;

void readParameters(ros::NodeHandle &n);
