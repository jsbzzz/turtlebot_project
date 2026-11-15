# Antigravity Workspace Guide (`turtlebot4_ws`)

This workspace targets **TurtleBot4 (`robot4`)** on **ROS 2 Jazzy** with a top-mounted camera (`top_cam`), multi-model YOLO person tracking (`person_tracker`), 2D LiDAR sensor fusion, and Nav2 localization/navigation.

> Full architecture, runbooks, desktop migration steps, and pending tasks are documented in [`HANDOFF.md`](./HANDOFF.md). Read [`HANDOFF.md`](./HANDOFF.md) first when resuming work or asking about current status.

## Environment & Build Rules

1. **Python Virtualenv & `numpy` Constraint**:
   - Runtime uses `~/venvs/rokey_venv` (created with `--system-site-packages`) via `person_tracker.env_bootstrap.bootstrap_venv()`.
   - **Never upgrade `numpy` to `2.x`** in `rokey_venv` — ROS 2 Jazzy `cv_bridge` is compiled against `numpy 1.26.x` (`numpy==1.26.4`).
   - On desktop with NVIDIA GPU, PyTorch CUDA (`torch` + `torchvision`) can be used (`device` toggleable via `D` key in `multi_tracker_node`); on CPU laptops, CPU-only wheels are used.
2. **ROS 2 Discovery & Namespace**:
   - Real robot namespace: `/robot4`, `ROS_DOMAIN_ID=4`, `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`, `ROS_DISCOVERY_SERVER=";;;;192.168.108.104:11811;"` (`/etc/turtlebot4_discovery/setup.bash`).
   - For isolated `ros2 bag play`, unset `ROS_DISCOVERY_SERVER` and switch `ROS_DOMAIN_ID` (e.g., `99`) with `--clock` and `use_sim_time:=true` to avoid `TF_OLD_DATA` collisions with the live robot.
3. **Always Rebuild After Editing Python Files**:
   - In this workspace, `colcon build --symlink-install --packages-select person_tracker` copies `.py` scripts into `build/person_tracker`. Always run:
     ```bash
     source /opt/ros/jazzy/setup.bash && colcon build --symlink-install --packages-select person_tracker
     ```

## Critical Technical Invariants

- **Top Camera Separation (`top_cam`)**: Never publish the added camera on `/oakd/...` or `oakd_rgb_camera_frame`; all configs live in `src/person_tracker/config/phone_camera.yaml` (`top_cam_link`, `top_cam_optical_frame`, `/robot4/top_cam/image_raw/compressed`).
- **Namespaced TF Topics**: TurtleBot4 publishes TF on `/robot4/tf` (`VOLATILE`) and `/robot4/tf_static` (`TRANSIENT_LOCAL`), **not** global `/tf` and `/tf_static`. Custom nodes subscribe directly to `f'{ns_prefix}/tf'` and `f'{ns_prefix}/tf_static'` into a local `tf2_ros.Buffer`.
- **RPLIDAR Orientation**: `rplidar_link` on `/robot4` is rotated **+90° yaw** relative to `base_link`. Always transform scan points with TF matrices (`lidar_person_localizer.py`) rather than assuming `scan angle == 0` is robot forward.
- **Executable Names (`setup.py`)**: `camera_pub`, `top_cam_tf`, `init_pose`, `net_monitor`, `multi_tracker_node`, `tracker_node`, `bbox_tracker_node`, `seg_tracker_node`. (Use `multi_tracker_node`, not `multi_tracker`.)
- **Upstream Modifications**: `src/turtlebot4/turtlebot4_navigation/config/localization.yaml` has `set_initial_pose: true` (`0, 0, 0`), and `nav2.yaml` / `nav2.launch.py` / `navigation_launch.py` were customized in `src/turtlebot4`. Preserve those changes unless explicitly asked to reset.
