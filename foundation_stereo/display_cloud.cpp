//
// Created by kaylor on 11/27/25.
//

#include "display_cloud.h"
#include "kaylordut/log/logger.h"

DisplayCloud::DisplayCloud() {
  viewer_ = std::make_shared<pcl::visualization::PCLVisualizer>("3D Viewer");
  viewer_->setBackgroundColor(255, 255, 255);
  viewer_->addCoordinateSystem(1.0);
  viewer_->initCameraParameters();
  // 设置相机位置，看向Z轴方向，扩大视野
  viewer_->setCameraPosition(0, 0, -2.0, // 相机位置 (在原点上方)
                             0, 0, 1,  // 观察点位置 (看向Z轴正方向)
                             0, -1, 0, // 相机上方向量
                             0.0       // 视口大小 (0.0表示自动)
  );
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
