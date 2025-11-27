//
// Created by kaylor on 11/27/25.
//

#pragma once
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

class DisplayCloud {
public:
  DisplayCloud();
  ~DisplayCloud() = default;
  void ShowCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud);

private:
  pcl::visualization::PCLVisualizer::Ptr viewer_;
};
