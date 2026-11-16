#!/usr/bin/env bash
set -e

# 실제 로봇(ROS_DOMAIN_ID=4, Discovery Server)과 100% 격리된 오프라인 테스트 환경
unset ROS_DISCOVERY_SERVER
export ROS_SUPER_CLIENT=False
export ROS_DOMAIN_ID=99
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

source /opt/ros/jazzy/setup.bash
source ~/turtlebot4_ws/install/setup.bash

MODE="${1:-}"
BAG_PATH="${2:-$HOME/bag/pose_estimate}"

case "$MODE" in
  play)
    echo "[Offline Bag Play] DOMAIN_ID=$ROS_DOMAIN_ID | Bag: $BAG_PATH"
    exec ros2 bag play "$BAG_PATH" --clock
    ;;
  tracker)
    echo "[Offline Multi-Tracker] DOMAIN_ID=$ROS_DOMAIN_ID | use_sim_time:=true"
    exec ros2 run person_tracker multi_tracker_node --ros-args -p use_sim_time:=true
    ;;
  rviz)
    echo "[Offline RViz2] DOMAIN_ID=$ROS_DOMAIN_ID | use_sim_time:=true"
    exec ros2 launch turtlebot4_viz view_navigation.launch.py namespace:=/robot4 use_sim_time:=true
    ;;
  *)
    echo "사용법:"
    echo "  $0 tracker                  # 오프라인 트래커 실행 (DOMAIN_ID=99)"
    echo "  $0 play [bag_경로]          # 녹화본 재생 (기본: ~/bag/pose_estimate)"
    echo "  $0 rviz                     # RViz2 실행 (DOMAIN_ID=99)"
    exit 1
    ;;
esac
