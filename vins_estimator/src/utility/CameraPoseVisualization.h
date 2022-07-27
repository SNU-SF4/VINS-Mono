#pragma once

#include <ros/ros.h>
#include <std_msgs/ColorRGBA.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

class CameraPoseVisualization {
public:
	std::string m_marker_ns;

	CameraPoseVisualization(float r, float g, float b, float a);

	void setScale(double s); // set scale of the visualization
	void setLineWidth(double width); // set line width of the visualization

	void add_pose(const Eigen::Vector3d& p, const Eigen::Quaterniond& q);
	void reset();

	void publish_by(ros::Publisher& pub, const std_msgs::Header& header); // publish the visualization

private:
	std::vector<visualization_msgs::Marker> m_markers;
	std_msgs::ColorRGBA m_image_boundary_color;
	std_msgs::ColorRGBA m_optical_center_connector_color;
	double m_scale;
	double m_line_width;

	static const Eigen::Vector3d imlt; // image left top
	static const Eigen::Vector3d imlb; // image left bottom
	static const Eigen::Vector3d imrt; // image right top
	static const Eigen::Vector3d imrb; // image right bottom
	static const Eigen::Vector3d oc  ; // optical center
	static const Eigen::Vector3d lt0 ; // left top 0
	static const Eigen::Vector3d lt1 ; // left top 1
	static const Eigen::Vector3d lt2 ; // left top 2
};
