# TurtleBot4 Person Tracker & Localization 인수인계 문서 (`HANDOFF.md`)

> **작성 기준일**: 2026-11-15  
> **목적**: 노트북에서 개발·검증한 `/home/jsb/turtlebot4_ws` 작업물을 데스크탑(또는 다른 PC)으로 옮겨 **Antigravity**로 바로 이어서 작업·실행할 수 있도록 현재까지의 변경 사항, 구조, 실행법, 보류 중인 작업(Backlog)을 정리한 문서입니다.

---

## 1. 프로젝트 개요 및 현재 달성 상태 (Done)

현재 워크스페이스(`turtlebot4_ws`)는 **TurtleBot4 (`/robot4`)** 에 상단 카메라(스마트폰 DroidCam 또는 웹캠)를 추가 장착하고, **YOLO 다중 모델 추적 + TF 좌표계 + 2D LiDAR(`/robot4/scan`) + AMCL 지도(`/robot4/map`)** 를 결합해 **특정 타겟 인물의 위치를 로봇 기준(`base_link`)과 저장된 맵 기준(`map`)으로 실시간 추정·시각화·발행**하는 단계까지 완성 및 실기 검증되었습니다.

### 완료 및 검증된 항목
1. **상단 추가 카메라(`top_cam`) 토픽/프레임 분리**:
   - 기본 OAK-D(`/robot4/oakd/...`)와 충돌하지 않도록 `/robot4/top_cam/image_raw/compressed` 및 `top_cam_link` / `top_cam_optical_frame`으로 분리.
   - 설정 단일화: `src/person_tracker/config/phone_camera.yaml`
2. **상단 카메라 전용 TF & 위치 모니터 (`top_cam_tf`)**:
   - `oakd_rgb_camera_frame -> top_cam_link -> top_cam_optical_frame` Static TF 발행.
   - `base_link`, `base_footprint`, `odom`, `map` 기준 상단 카메라의 실시간 3D 좌표 계산 및 토픽(`/robot4/top_cam/pose_in_base`, `/robot4/top_cam/pose_in_map`) 발행.
   - 도킹 등으로 `/robot4/scan`이 꺼져 AMCL TF가 아직 없을 때도 `initial_pose + odom` 기반 폴백 좌표 및 원인 진단 출력.
3. **AMCL 초기 위치 설정 노드 (`init_pose`) & 맵 파일 정리**:
   - `src/turtlebot4/turtlebot4_navigation/config/localization.yaml`에 `set_initial_pose: true` (`0, 0, 0`) 추가.
   - `ros2 run person_tracker init_pose` 로 RViz `2D Pose Estimate` 없이도 `/robot4/initialpose` 발행 및 `map -> base_link` / `top_cam_link` 연결 확인.
   - 맵 파일 오타(`gallary.yaml` 내부 `image` 경로 및 `gallery.yaml` / `gallery.pgm` 신규 생성) 수정 완료.
   - `turtlebot4_ws1`의 깨진 심볼릭 링크 문제를 해결하기 위해 `turtlebot4`, `turtlebot4_desktop`, `irobot_create_description` 소스를 `src/`에 복구 후 빌드 완료.
4. **웹캠 TF + YOLO + 2D LiDAR 타겟 위치 정합 (`lidar_person_localizer.py` + `multi_tracker_node.py`)**:
   - YOLO 9종 모델(`YOLOv8n / 11n / 26n` × `Detect / Segment / Pose`) + 5종 추적 알고리즘(ByteTrack / Task-ReID / BoT-SORT / BoT+ReID / Adaptive) 지원.
   - 타겟 인물 바운딩 박스의 수평 방위각과 `/robot4/scan`(TF 행렬로 `rplidar_link -> top_cam_link` 변환)을 정합해 배경 벽을 제외한 앞쪽 인물 거리(25백분위수 클러스터) 추출 (`[LiDAR]`, 미수신 시 바운딩 박스 기반 `[CamEst]`).
   - OpenCV HUD, 바운딩 박스 하단 오버레이, 터미널 로그(1초 주기), ROS 2 토픽(`/robot4/target_person/point_base`, `point_map`, `marker`) 출력 및 **실기 테스트 검증 완료** (`[타겟 위치 ID=1 | LiDAR]`).
5. **네트워크 대역폭 분리 모니터 (`net_monitor`)**:
   - `[1] robot4 기본 토픽`, `[2] 추가 토픽`, `[3] 실제 Wi-Fi NIC(wlp2s0) RX/TX`를 분리 측정.

---

## 2. 패키지 및 주요 파일 구조

```text
turtlebot4_ws/
├── AGENTS.md                           # Antigravity 자동 인식 워크스페이스 규칙/컨텍스트
├── HANDOFF.md                          # (본 문서) 인수인계 및 데스크탑 이식 가이드
└── src/
    ├── person_tracker/                 # 메인 개발 패키지
    │   ├── config/phone_camera.yaml    # ★ 네임스페이스, 영상 소스, 카메라 외참수(x,y,z,rpy), hfov_deg, initial_pose
    │   ├── launch/phone_camera.launch.py # top_cam_tf + camera_pub 동시 실행 런치
    │   ├── models/*.pt                 # 오프라인 YOLO 가중치 9종 (v8n/11n/26n x det/seg/pose)
    │   ├── package.xml / setup.py      # 실행 파일(console_scripts) 등록
    │   └── person_tracker/
    │       ├── env_bootstrap.py        # 가상환경(rokey_venv) 자동 연결, 모델/토픽 경로 해석
    │       ├── camera_publisher_node.py # 웹캠/DroidCam 영상 캡처 및 압축/원본 토픽 발행 (camera_pub)
    │       ├── top_cam_tf_node.py      # 상단 캠 TF 발행 및 base/odom/map 좌표 모니터 (top_cam_tf)
    │       ├── init_pose_node.py       # AMCL /initialpose 발행 유틸 노드 (init_pose)
    │       ├── lidar_person_localizer.py # ★ 웹캠 방위각 + 2D LiDAR + TF 기반 타겟 3D 위치 계산기
    │       ├── multi_tracker_node.py   # ★ 통합 9종 모델 x 5종 트래커 GUI + LiDAR/Map HUD (multi_tracker_node)
    │       ├── multi_tracker_core.py   # 모델 전환 / Adaptive 추적 코어
    │       ├── feature_extractors.py   # BBox / Seg / Pose 외형 Re-ID 특징 추출기
    │       ├── net_monitor_node.py     # ROS 2 토픽 + NIC 대역폭 모니터 (net_monitor)
    │       ├── tracker_node.py / seg_tracker_node.py (+ *_core.py) # 단일 모델 레거시 노드
    ├── turtlebot4/                     # 수정됨: localization.yaml, nav2.yaml, nav2.launch.py, navigation_launch.py
    ├── turtlebot4_desktop/             # RViz 시각화 패키지 (view_navigation.launch.py)
    └── irobot_create_description/      # Create3 URDF 모델 (robot_state_publisher 연동)
```

### 등록된 ROS 2 실행 파일 (`ros2 run person_tracker <이름>`)
* `camera_pub` — 상단 웹캠 / DroidCam 퍼블리셔
* `top_cam_tf` — 상단 캠 TF 발행 및 좌표 모니터
* `init_pose` — AMCL 초기 위치 설정기
* `multi_tracker_node` — 통합 YOLO + LiDAR/Map 타겟 위치 추적 GUI *(주의: 실행 파일명은 `multi_tracker`가 아니라 `multi_tracker_node`)*
* `net_monitor` — 네트워크 대역폭 모니터
* `tracker_node` / `bbox_tracker_node` / `seg_tracker_node` — 구버전 단일 모델 노드

---

## 3. 데스크탑으로 옮겨(Import) 이어서 작업하는 방법

데스크탑에 이 `turtlebot4_ws` 폴더를 복사(또는 동기화)한 뒤 아래 항목만 확인하면 **똑같은 환경에서 바로 작업**할 수 있습니다.

### 3-1. 외부 파일 체크 (워크스페이스 바깥 경로)
1. **맵 파일 (`~/maps/`)**:
   - 워크스페이스 바깥인 `/home/jsb/maps/`에 저장되어 있습니다. 데스크탑의 `~/maps/`로 함께 복사해 두세요.
   - `gallery.yaml` (`image: gallery.pgm`), `gallary.yaml` (`image: gallary.pgm`), `c_4map.yaml`이 포함되어 있습니다.
2. **Python 가상환경 (`~/venvs/rokey_venv`)**:
   - `person_tracker/env_bootstrap.py`가 `~/venvs/rokey_venv` (또는 현재 활성화된 `$VIRTUAL_ENV`)의 `site-packages`를 `sys.path`에 자동 주입합니다.
   - **핵심 호환성**: ROS 2 Jazzy `cv_bridge` 충돌 방지를 위해 **`numpy==1.26.4` (`numpy<2`) 유지 필수**.
   - 데스크탑에 GPU가 있다면 CUDA 버전 `torch` / `torchvision`을 설치하면 `multi_tracker_node`에서 `D` 키로 CPU $\leftrightarrow$ GPU(`CUDA:0`) 전환이 바로 동작합니다. (현재 노트북 검증 버전: `torch 2.15.0+cpu`, `ultralytics 8.4.213`, `opencv-python 4.11.0`, `numpy 1.26.4`, `lap 0.5.14`)
3. **터틀봇4 Discovery & Shell 함수 (`~/.bashrc`)**:
   - `/etc/turtlebot4_discovery/setup.bash`: `ROS_DOMAIN_ID=4`, `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`, `ROS_DISCOVERY_SERVER=";;;;192.168.108.104:11811;"`
   - `~/.bashrc` 편의 함수: `loc 4 <map.yaml>`, `nav 4`, `rv 4`, `dock 4`, `undock 4`

### 3-2. 데스크탑에서 첫 빌드 명령어
```bash
cd ~/turtlebot4_ws
source ~/venvs/rokey_venv/bin/activate
source /opt/ros/jazzy/setup.bash
rm -rf build/person_tracker install/person_tracker   # 다른 PC의 경로 캐시/pyc 제거
colcon build --symlink-install
source install/setup.bash
```
> **중요**: 이 환경의 `colcon build --symlink-install --packages-select person_tracker`는 `.py` 수정 시 `build/person_tracker` 반영을 위해 **코드 수정 후 반드시 재빌드**해야 합니다.

---

## 4. 반드시 알아야 할 기술적 핵심 사항 (트러블슈팅 정리)

1. **TurtleBot4 네임스페이스 TF (`/robot4/tf`, `/robot4/tf_static`)**:
   - `robot4`는 글로벌 `/tf`가 아니라 `/robot4/tf`(`VOLATILE`), `/robot4/tf_static`(`TRANSIENT_LOCAL`)을 사용합니다.
   - `top_cam_tf_node.py`, `init_pose_node.py`, `lidar_person_localizer.py`는 이를 직접 구독해 `tf2_ros.Buffer`에 채우므로 별도 CLI 리매핑 없이 동작합니다.
2. **RPLIDAR 좌표계 회전 (+90°)**:
   - `/robot4/tf_static` 기준 `base_link -> rplidar_link`는 `xyz=(-0.04, 0.0, 0.193)`이고 **Z축 +90° (`yaw = pi/2`) 회전**되어 있습니다.
   - 반드시 TF 행렬로 `rplidar_link -> top_cam_link` 변환 후 방위각을 비교해야 합니다(`lidar_person_localizer.py`에 구현됨).
3. **도킹 중 라이다 0Hz & AMCL `map` 대기 문제**:
   - 도킹 시 라이다 모터가 멈추거나(`0Hz`) 로봇의 `rplidar` 노드가 죽어 있으면 `/robot4/scan`이 안 올라와 AMCL이 `map -> odom` TF를 발행하지 않습니다.
   - 현재 코드에는 도킹 중 `/robot4/start_motor` 자동 호출 및 `Init+Odom` 폴백이 들어가 있습니다. 만약 `/robot4/scan` 발행자 자체가 `0개`라면 로봇(`ssh-robot`)에서 `sudo systemctl restart turtlebot4.service` (또는 `ros2 launch turtlebot4_bringup rplidar.launch.py`)로 복구하면 `[AMCL]` 및 `[LiDAR]`가 즉시 활성화됩니다.
4. **`ros2 bag play` 시 `TF_OLD_DATA` 경고 방지**:
   - 실제 로봇이 켜진 같은 도메인(`ROS_DOMAIN_ID=4`)에서 과거 bag을 재생하면 현재 시각 TF와 충돌합니다.
   - 재생 시에는 `unset ROS_DISCOVERY_SERVER && export ROS_DOMAIN_ID=99`로 분리한 뒤 `--clock` 및 `use_sim_time:=true`로 실행하세요.

---

## 5. 실전 실행 및 종료 순서 (Runbook)

### 실행 (순서/시간차 무관하게 자동 연결되나, 초기 위치 리셋 시점에만 로봇이 기준점에 있어야 함)
```bash
# [터미널 1] 로컬라이제이션 (localization.yaml에 initial_pose 0,0,0 설정됨)
loc 4 /home/jsb/maps/gallery.yaml

# [터미널 2] (필요 시) AMCL 초기 위치 수동 재전송
ros2 run person_tracker init_pose

# [터미널 3] 상단 카메라 영상 + TF 발행 (웹캠 0번 또는 DroidCam URL)
ros2 launch person_tracker phone_camera.launch.py video_source:=0
# 또는 DroidCam 사용 시 phone_camera.yaml의 video_source 그대로 사용:
# ros2 launch person_tracker phone_camera.launch.py

# [터미널 4] 타겟 인물 추적 + 로봇/맵 3D 위치 추정 GUI
ros2 run person_tracker multi_tracker_node

# [터미널 5] (선택) RViz2에서 지도 위 타겟 마커(/robot4/target_person/marker) 확인
rv 4
```

### 발행되는 주요 결과 토픽
* `/robot4/target_person/point_base` (`geometry_msgs/msg/PointStamped`, `frame_id: "base_link"`)
* `/robot4/target_person/point_map` (`geometry_msgs/msg/PointStamped`, `frame_id: "map"`)
* `/robot4/target_person/marker` (`visualization_msgs/msg/Marker`, RViz 원기둥 마커)
* `/target_person/offset` (`geometry_msgs/msg/Point`, 화면 정규화 오프셋 및 면적비)

### 권장 종료 순서 (`Ctrl + C`)
1. `multi_tracker_node` 종료
2. `phone_camera.launch.py` 및 `teleop` 종료 *(이동했다면 `dock 4` 복귀 후 종료)*
3. `rv 4` 및 `loc 4` 종료

---

## 6. 보류된 작업 및 다음 할 일 (Backlog)

데스크탑에서 이어서 바로 진행할 수 있는 후속 작업 목록입니다:

1. **[보류 중이던 사용자 요청] TurtleBot4 후진 텔레옵(`teleop`) 덜컥거림/가속 개선**:
   - 사용자가 *"텔레옵으로 후진하고 멈출 때 덜컥거리면서 불안정해 보이는데, 처음 가속/감속을 조절하는 설정을 수정해 개선할 수 있는지 확인"* 을 요청했다가, 라이다+카메라 사람 위치 추정을 먼저 완성하기 위해 잠시 보류한 상태입니다.
   - 데스크탑에서 이어서 진행 시: `robot4`의 Create3 모션 파라미터 / `cmd_vel` 램핑 / `teleop` 속도 프로파일 또는 중간 스무딩 노드 적용 여부를 점검·테스트할 것.
2. **추출된 타겟 좌표(`/robot4/target_person/point_base`, `point_map`) 기반 추종 주행(Follow) 연동**:
   - 현재는 타겟 위치 계산 및 시각화/토픽 발행까지 완료된 상태이며, 실제 로봇이 타겟을 따라 이동(`cmd_vel` 추종 또는 Nav2 `NavigateToPose` 목표 갱신)하는 제어 노드는 다음 단계.
3. **상단 카메라 외참수/화각 실측 보정 (`config/phone_camera.yaml`)**:
   - 실제 스마트폰/웹캠 장착 위치(`z: 0.17`, `pitch`)와 실제 수평 화각(`hfov_deg: 65.0`)을 현장 측정치에 맞춰 미세 조정 (현재 사용자는 높이 약 2cm 오차 가능성 언급).
