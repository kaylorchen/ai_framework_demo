//
// Created by kaylor on 12/1/25.
//
#include "rclcpp/rclcpp.hpp"
int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<rclcpp::Node>("my_node"));
  rclcpp::shutdown();
  return 0;
}
