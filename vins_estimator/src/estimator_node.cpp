#include <cstdio>
#include <cmath>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "estimator.h"
#include "parameters.h"
#include "utility/visualization.h"

#define DEF EIGEN_DONT_PARALLELIZE

// main vio operator
Estimator estimator;

// buffer related
std::condition_variable con;
double current_time = -1;
queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<sensor_msgs::PointCloudConstPtr> relo_buf;

// mutex for buf, status value and vio processing
std::mutex m_buf;
std::mutex m_state;
std::mutex m_estimator;

// temp status values
double latest_time;
Eigen::Vector3d tmp_P;
Eigen::Quaterniond tmp_Q;
Eigen::Vector3d tmp_V;
Eigen::Vector3d tmp_Ba;
Eigen::Vector3d tmp_Bg;
Eigen::Vector3d acc_0;
Eigen::Vector3d gyr_0;

// flags
bool init_feature = false;
bool init_imu = true;
double last_imu_t = false;

/**
 * @brief  IMU propagate: predict status values (Ps/Qs/Vs)
 * @param  imu_msg    imu_msg from imu_callback()
 */
void predict(const sensor_msgs::ImuConstPtr &imu_msg)
{
  double t = imu_msg->header.stamp.toSec(); // get current time

  if (init_imu) // first imu msg
  {
    latest_time = t;
    init_imu = false;
    return;
  }

  double dt = t - latest_time; // get dt
  latest_time = t;

  // get the IMU sampling data at the current moment
  double dx = imu_msg->linear_acceleration.x;
  double dy = imu_msg->linear_acceleration.y;
  double dz = imu_msg->linear_acceleration.z;
  Eigen::Vector3d linear_acceleration{dx, dy, dz};

  double rx = imu_msg->angular_velocity.x;
  double ry = imu_msg->angular_velocity.y;
  double rz = imu_msg->angular_velocity.z;
  Eigen::Vector3d angular_velocity{rx, ry, rz};

  Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator.g;

  // calculate the average angular velocity during dt
  Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg;

  // predict Quaternion from previous to current imu_msg
  tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);

  Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba) - estimator.g;

  // calculate the average acceleration during dt
  Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

  // predict Position, Velocity from previous to current imu_msg
  tmp_P = tmp_P + tmp_V * dt + 0.5 * un_acc * dt * dt;
  tmp_V = tmp_V + un_acc * dt;

  // update last measurement
  acc_0 = linear_acceleration;
  gyr_0 = angular_velocity;
}

/*
 * brief  update IMU parameters [P, Q, V, ba, bg, a, g],
 *        and this function will only be called in process() during nonlinear optimization
 */
void update()
{
  // get the imu update item of the last image frame in the sliding window from the estimator [P, Q, V, ba, bg, a, g]
  TicToc t_predict;
  latest_time = current_time;
  tmp_P = estimator.Ps[WINDOW_SIZE];
  tmp_Q = estimator.Rs[WINDOW_SIZE];
  tmp_V = estimator.Vs[WINDOW_SIZE];
  tmp_Ba = estimator.Bas[WINDOW_SIZE];
  tmp_Bg = estimator.Bgs[WINDOW_SIZE];
  acc_0 = estimator.acc_0;
  gyr_0 = estimator.gyr_0;

  // Perform PVQ recursion on the remaining imu_msg in imu_buf
  // (because the frequency of imu is much higher than the image frequency,
  // after getMeasurements() aligns the image and imu time, imu data will still exist in imu_buf)
  queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf;
  for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
    predict(tmp_imu_buf.front());
}

/**
 * @brief   pair the image frame with the corresponding IMU data,
 *          and the IMU data time is in front of the image frame
 */
std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>>
getMeasurements()
{
  std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;

  while (true)
  {
    // Boundary judgment: After the data is fetched, the pairing is completed
    if (imu_buf.empty() || feature_buf.empty())
      return measurements;

    // Boundary judgment: The timestamp of all data in IMU buf is earlier
    // than the timestamp of the first frame of img buf,
    // indicating that there is a lack of IMU data, and it is necessary to wait for IMU data
    if (imu_buf.back()->header.stamp.toSec() <= feature_buf.front()->header.stamp.toSec() + estimator.td)
      return measurements;

    // Boundary judgment: the time of the first IMU data is greater
    // than the time of the first image feature data,
    // indicating that there are many image frames
    if (imu_buf.front()->header.stamp.toSec() >= feature_buf.front()->header.stamp.toSec() + estimator.td)
    {
      ROS_WARN("throw img, only should happen at the beginning");
      feature_buf.pop();
      continue;
    }

    // core operation: load visual frame feature information
    sensor_msgs::PointCloudConstPtr img_msg = feature_buf.front();
    feature_buf.pop();

    // core operation: transfer to IMU information
    std::vector<sensor_msgs::ImuConstPtr> IMUs;
    while (imu_buf.front()->header.stamp.toSec() < img_msg->header.stamp.toSec() + estimator.td)
    {
      IMUs.emplace_back(imu_buf.front());
      imu_buf.pop();
    }
    // because the last frame of IMU information is shared by two adjacent visual frames
    IMUs.emplace_back(imu_buf.front());

    if (IMUs.empty())
      ROS_WARN("no imu between two image");

    // pair the image frame with the corresponding IMU data
    measurements.emplace_back(IMUs, img_msg);
  }
}

/**
 * @brief  put IMU measurement in buffer and publish status values: P/Q/V
 */
void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
  if (imu_msg->header.stamp.toSec() <= last_imu_t)
  {
    ROS_WARN("imu message in disorder!");
    return;
  }

  // update IMU timestamp
  last_imu_t = imu_msg->header.stamp.toSec();

  m_buf.lock();
  imu_buf.push(imu_msg); // push imu message to buffer
  m_buf.unlock();
  con.notify_one();

  last_imu_t = imu_msg->header.stamp.toSec();

  {
    std::lock_guard<std::mutex> lg(m_state);
    predict(imu_msg); // IMU propagate: predict state based on imu message

    // update header timestamp for visualization
    std_msgs::Header header = imu_msg->header;
    header.frame_id = "world";
    // if estimator's solver is in the nonlinear optimization stage,
    // publish the PVQ calculated from predict() to visualize
    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
      pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header); // for visualization (/imu_propagate)
  }
}

/**
 * @brief  put feature measurement in buffer from feature_tracker node
 */
void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
  if (!init_feature)
  {
    //skip the first detected feature, which doesn't contain optical flow speed
    init_feature = true;
    return;
  }
  m_buf.lock();
  feature_buf.push(feature_msg); // push feature message to buffer
  m_buf.unlock();
  con.notify_one();
}

/**
 * @brief  reset all state quantities to zero and clear all the data in buffer
 */
void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
  if (restart_msg->data == true)
  {
    ROS_WARN("restart the estimator!");
    m_buf.lock();
    while(!feature_buf.empty())
      feature_buf.pop();
    while(!imu_buf.empty())
      imu_buf.pop();
    m_buf.unlock();
    m_estimator.lock();
    estimator.clearState();
    estimator.setParameter();
    m_estimator.unlock();
    current_time = -1;
    last_imu_t = 0;
  }
}

/**
 * @brief  put relocalization flag in buffer
 */
void relocalization_callback(const sensor_msgs::PointCloudConstPtr &points_msg)
{
    m_buf.lock();
    relo_buf.push(points_msg);
    m_buf.unlock();
}

/**
 * @brief  IMU pre-integration and get pre-optimized status values Ps/Vs/Rs
 */
void processIMU(sensor_msgs::ImuConstPtr &imu_msg,
                sensor_msgs::PointCloudConstPtr &img_msg)
{
  double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
  double t = imu_msg->header.stamp.toSec();
  double img_t = img_msg->header.stamp.toSec() + estimator.td;

  // For each imu_msg in the measurement, calculate dt and execute processIMU()

  // For most cases, the timestamp of the IMU will be earlier than that of img,
  // so just select the data of the IMU directly
  if (t <= img_t)
  {
    if (current_time < 0) // first imu message
      current_time = t;
    double dt = t - current_time; // time interval between two imu messages
    ROS_ASSERT(dt >= 0); // dt should be positive
    current_time = t; // update current_time
    dx = imu_msg->linear_acceleration.x;
    dy = imu_msg->linear_acceleration.y;
    dz = imu_msg->linear_acceleration.z;
    rx = imu_msg->angular_velocity.x;
    ry = imu_msg->angular_velocity.y;
    rz = imu_msg->angular_velocity.z;

    // the IMU roughly pre-integrates, and then the value is passed to a newly created IntegrationBase object
    // IMU Pre-integration
    estimator.processIMU(dt,
                         Vector3d(dx, dy, dz),
                         Vector3d(rx, ry, rz));
  }
  // For the IMU data at the boundary position, it is shared by two adjacent frames,
  // and the impact on the previous frame will be greater. Here, the data is allocated linearly
  // The first imu_msg greater than the image frame timestamp is shared by two image frames (few occurrences)
  else
  {
    // current_time < img_time < t
    double dt_1 = img_t - current_time;
    double dt_2 = t - img_t;
    current_time = img_t;
    ROS_ASSERT(dt_1 >= 0);
    ROS_ASSERT(dt_2 >= 0);
    ROS_ASSERT(dt_1 + dt_2 > 0);

    // a simple linear allocation
    double w1 = dt_2 / (dt_1 + dt_2);
    double w2 = dt_1 / (dt_1 + dt_2);
    dx = w1 * dx + w2 * imu_msg->linear_acceleration.x;
    dy = w1 * dy + w2 * imu_msg->linear_acceleration.y;
    dz = w1 * dz + w2 * imu_msg->linear_acceleration.z;
    rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
    ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
    rz = w1 * rz + w2 * imu_msg->angular_velocity.z;
    estimator.processIMU(dt_1,
                         Vector3d(dx, dy, dz),
                         Vector3d(rx, ry, rz));
  }
}

/**
 * @brief  set the relocalization frame
 */
void setReloFrame(sensor_msgs::PointCloudConstPtr &relo_msg)
{
  // take out the last relocation frame in relo_buf
  while (!relo_buf.empty())
  {
    relo_msg = relo_buf.front();
    relo_buf.pop();
  }

  if (relo_msg != nullptr)
  {
    vector<Vector3d> match_points;
    double frame_stamp = relo_msg->header.stamp.toSec();
    for (auto point : relo_msg->points)
    {
      Vector3d u_v_id; // point to vector
      u_v_id.x() = point.x;
      u_v_id.y() = point.y;
      u_v_id.z() = point.z;
      match_points.push_back(u_v_id);
    }
    Vector3d relo_t(relo_msg->channels[0].values[0], relo_msg->channels[0].values[1], relo_msg->channels[0].values[2]);
    Quaterniond relo_q(relo_msg->channels[0].values[3],
                       relo_msg->channels[0].values[4],
                       relo_msg->channels[0].values[5],
                       relo_msg->channels[0].values[6]);
    Matrix3d relo_r = relo_q.toRotationMatrix();
    int frame_index = static_cast<int>(relo_msg->channels[0].values[7]);

    // execute setReloFrame() in the estimator
    estimator.setReloFrame(frame_stamp, frame_index, match_points, relo_t, relo_r);
  }
}

/**
 * @brief  stores the information of the feature points
 */
void processVIO(sensor_msgs::PointCloudConstPtr& img_msg)
{
  map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;
  for (unsigned int i = 0; i < img_msg->points.size(); i++)
  {
    // extract the img information and put it in the image container
    int v = img_msg->channels[0].values[i] + 0.5; // get id of point
    int feature_id = v / NUM_OF_CAM; // hash
    int camera_id = v % NUM_OF_CAM; // mono: 0

    // normalized coordinates
    double x = img_msg->points[i].x;
    double y = img_msg->points[i].y;
    double z = img_msg->points[i].z;

    // pixel coordinates
    double p_u = img_msg->channels[1].values[i];
    double p_v = img_msg->channels[2].values[i];

    // pixel motion speed
    double velocity_x = img_msg->channels[3].values[i];
    double velocity_y = img_msg->channels[4].values[i];
    ROS_ASSERT(z == 1);
    Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
    xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
    // key: feature_id, value: camera_id, xyz_uv_velocity
    image[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
  }

  // execute processImage() in the estimator
  estimator.processImage(image, img_msg->header);
}

/**
 * @brief  visualize through rviz
 */
void visualize(sensor_msgs::PointCloudConstPtr &relo_msg, std_msgs::Header &header)
{
  pubOdometry(estimator, header);
  pubKeyPoses(estimator, header);
  pubCameraPose(estimator, header);
  pubPointCloud(estimator, header);
  pubTF(estimator, header);
  pubKeyframe(estimator);
  if (relo_msg != nullptr)
    pubRelocalization(estimator);
}

/**
 * @brief  main vio function, including initialization and optimization
 * @param  measurements     pair data(IMUs, feature) vector
 */
void processMeasurement(vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>>& measurements)
{
  for (auto &measurement : measurements)
  {
    auto img_msg = measurement.second;

    // For each imu_msg in the measurement, calculate dt and execute processIMU()
    for (auto &imu_msg : measurement.first)
      processIMU(imu_msg,img_msg);

    // set relocalization frame
    sensor_msgs::PointCloudConstPtr relo_msg = nullptr;
    setReloFrame(relo_msg);
    ROS_DEBUG("processing vision data with stamp %f \n", img_msg->header.stamp.toSec());

    // main function for vio
    TicToc t_s;
    processVIO(img_msg);

    double whole_t = t_s.toc();
    printStatistics(estimator, whole_t);
    std_msgs::Header header = img_msg->header;
    header.frame_id = "world";

    // show in rviz
    visualize(relo_msg, header);
  }
}

/**
 * @brief  main vio function thread, including initialization and optimization
 */
[[noreturn]] void process()
{
  while (true)
  {
    // get measurement in buf and make them aligned
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;
    std::unique_lock<std::mutex> lk(m_buf);

    // keep the locked state and continue to pair data(IMUs, feature) vector
    con.wait(lk,
             [&] { return !(measurements = getMeasurements()).empty(); });
    lk.unlock();

    // main function of vio
    m_estimator.lock();
    processMeasurement(measurements);
    m_estimator.unlock();

    // update status value Rs/Ps/Vs
    m_buf.lock();
    m_state.lock();
    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
      update();
    m_state.unlock();
    m_buf.unlock();
  }
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "vins_estimator");
  ros::NodeHandle n("~");
  ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

  // read parameters & set configs
  readParameters(n);
  estimator.setParameter();

#ifdef DEF
  ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif
  ROS_WARN("waiting for image and imu...");

  registerPub(n); // register publisher for visualization

  // register subscribers
  ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
  ros::Subscriber sub_image = n.subscribe("/feature_tracker/feature", 2000, feature_callback);
  ros::Subscriber sub_restart = n.subscribe("/feature_tracker/restart", 2000, restart_callback);
  ros::Subscriber sub_relo_points = n.subscribe("/pose_graph/match_points", 2000, relocalization_callback);

  std::thread measurement_process{process};
  ros::spin();

  return 0;
}
