//
// Created by kaylor on 11/27/25.
//

#include "display_cloud.h"
#include "kaylordut/log/logger.h"

DisplayCloud::DisplayCloud() {
  viewer_ = std::make_shared<pcl::visualization::PCLVisualizer>("3D Viewer");
  viewer_->setBackgroundColor(0, 0, 0);
  viewer_->addCoordinateSystem(1.0);
  viewer_->initCameraParameters();
  KAYLORDUT_LOG_INFO("3D Viewer initialized");
}
void DisplayCloud::ShowCloud(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud) {
  viewer_->removeAllPointClouds();
  viewer_->addPointCloud(cloud, "cloud");
  viewer_->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");
  viewer_->spinOnce(1);
}
