#pragma once

#include <ros/ros.h>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "utility/utility.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <fstream>

const double FOCAL_LENGTH = 460.0;
const int WINDOW_SIZE = 10;
const int NUM_OF_CAM = 1;
const int NUM_OF_F = 1000;

extern double INIT_DEPTH;
extern double MIN_PARALLAX; // keyframe selection threshold (pixel)
extern int ESTIMATE_EXTRINSIC;

extern double ACC_N; // accelerometer measurement noise standard deviation
extern double ACC_W; // accelerometer bias random work noise standard deviation
extern double GYR_N; // gyroscope measurement noise standard deviation
extern double GYR_W; // gyroscope bias random work noise standard deviation

extern std::vector<Eigen::Matrix3d> RIC; // rotation from imu to camera
extern std::vector<Eigen::Vector3d> TIC; // translation from imu to camera
extern Eigen::Vector3d G;

extern double SOLVER_TIME; // max solver iteration time (ms), to guarantee real time
extern int NUM_ITERATIONS; // max solver iterations, to guarantee real time
extern std::string EX_CALIB_RESULT_PATH;
extern std::string VINS_RESULT_PATH;
extern std::string IMU_TOPIC;
extern double TD;
extern double TR;
extern int ESTIMATE_TD;
extern int ROLLING_SHUTTER;
extern double ROW, COL;


void readParameters(ros::NodeHandle &n);

enum SIZE_PARAMETERIZATION
{
    SIZE_POSE = 7,
    SIZE_SPEEDBIAS = 9,
    SIZE_FEATURE = 1
};

enum StateOrder
{
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};
