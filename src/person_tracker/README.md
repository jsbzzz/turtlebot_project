# TurtleBot4 Person Tracker (`person_tracker`)

TurtleBot4(ROS 2 Jazzy)의 **2D LiDAR(`/robot4/scan`)**, **상단 추가 카메라(노트북 웹캠 / 스마트폰 DroidCam) TF**, **YOLO + Part-ReID 다중 모델 추적기**를 결합하여 특정 타겟 인물을 식별하고 **로봇 기준(`base_link`) 및 저장된 맵(`map`) 기준 3D 좌표**를 실시간 계산·발행하는 패키지입니다.

---

## 1. 주요 기능

- **다중 YOLO 모델 & 추적 알고리즘 실시간 전환**
  - 모델 9종: `YOLOv8n / YOLO11n / YOLO26n` × `Detect / Segment / Pose`
  - 추적 5종: `ByteTrack`, `Task-ReID`, `BoT-SORT`, `BoT+ReID`, `Adaptive`
  - 무채색 옷(검정/흰/회색) 구분을 위한 `H-S + 밝기(V)` 결합 히스토그램 및 겹침 영역 제외 마스킹
- **라이다 + 상단 카메라 TF 융합 3D 위치 추정**
  - 상단 카메라(`top_cam_link`) 수평 각도와 2D 라이다(`/robot4/scan`) 점군을 정합하여 타겟 거리 측정
  - 로봇 몸체(`base_link`) 및 저장된 맵(`map`) 기준 좌표 산출 및 3D 공간 게이팅
- **원클릭 통합 실행 (`bringup.launch.py`)**
  - 저장된 맵 로컬라이제이션(`localization.launch.py`) + 초기 위치(`init_pose`) + 노트북 웹캠(`video_source:=0`) + `top_cam_tf` + `multi_tracker_node` 일괄 실행

---

## 2. 다른 PC에서 설치 및 빌드 (GitHub 클론 후)

### 사전 요구 사항
- **Ubuntu 24.04 + ROS 2 Jazzy** (또는 Humble/Iron)
- TurtleBot4 연동 시: `ros-jazzy-turtlebot4-navigation`, `ros-jazzy-turtlebot4-viz`, `ros-jazzy-turtlebot4-desktop` 설치 권장
  ```bash
  sudo apt-get update
  sudo apt-get install -y ros-jazzy-turtlebot4-navigation ros-jazzy-turtlebot4-viz ros-jazzy-turtlebot4-desktop
  ```

### 워크스페이스 클론 및 원클릭 설치/빌드
```bash
# 1. 워크스페이스 루트로 클론 (src/ 안이 아닌 ~ 등 원하는 위치에 클론)
cd ~
git clone https://github.com/jsbzzz/turtlebot_project.git

# 2. 시스템 의존성, Python 가상환경(rokey_venv), PyTorch/YOLO 설치 및 colcon build 자동 수행
cd ~/turtlebot_project/src/person_tracker
bash setup_and_build.sh
```
> **주의 (`cv_bridge` 호환성)**: `requirements.txt`는 ROS 2 Jazzy의 `cv_bridge` 충돌을 방지하기 위해 `numpy<2.0.0`으로 고정되어 있습니다.

---

## 3. 실행 방법

매 터미널 시작 시 환경 로드:
```bash
source ~/turtlebot_project/env_tracker.sh
```

### ① 원클릭 통합 실행 (권장: 맵 + 초기위치 + 노트북 웹캠 + 트래커)
로봇을 도킹 스테이션(또는 맵 원점 기준점)에 둔 상태에서 아래 한 줄만 실행하면 됩니다:
```bash
ros2 launch person_tracker bringup.launch.py video_source:=0
```

#### 자주 쓰는 실행 옵션
```bash
# RViz2까지 한 번에 띄우기
ros2 launch person_tracker bringup.launch.py video_source:=0 run_rviz:=true

# 다른 로봇 번호(/robot5 등) 또는 다른 맵 파일 사용 시
ros2 launch person_tracker bringup.launch.py namespace:=/robot5 map:=/path/to/your_map.yaml video_source:=0

# 스마트폰 DroidCam 스트림 사용 시
ros2 launch person_tracker bringup.launch.py video_source:=http://192.168.108.100:4747/video

# 맵 로컬라이제이션 없이 카메라+라이다(로봇 기준 좌표)만 빠르게 테스트할 때
ros2 launch person_tracker bringup.launch.py use_localization:=false video_source:=0
```

### ② 터미널별 개별 실행 (디버깅 / 분리 실행용)
```bash
# [터미널 1] 맵 로컬라이제이션
ros2 launch turtlebot4_navigation localization.launch.py namespace:=/robot4 map:=$(ros2 pkg prefix person_tracker)/share/person_tracker/maps/gallery.yaml

# [터미널 2] AMCL 초기 위치 발행 (config/phone_camera.yaml의 initial_pose 사용)
ros2 run person_tracker init_pose

# [터미널 3] 노트북 웹캠(0) + 상단 캠 TF 발행
ros2 launch person_tracker phone_camera.launch.py video_source:=0

# [터미널 4] 멀티 트래커 (영상에서 사람을 마우스 왼쪽 클릭해 타겟 지정)
ros2 run person_tracker multi_tracker_node
```

---

## 4. 발행 토픽 및 조작법

### 실시간 발행 토픽
| 토픽 이름 | 메시지 타입 | 설명 |
| :--- | :--- | :--- |
| `/robot4/target_person/point_base` | `geometry_msgs/PointStamped` | 로봇(`base_link`) 기준 타겟 3D 좌표 (m) |
| `/robot4/target_person/point_map` | `geometry_msgs/PointStamped` | 저장된 맵(`map`) 기준 타겟 3D 좌표 (m) |
| `/robot4/target_person/marker` | `visualization_msgs/Marker` | RViz2 시각화용 원기둥 마커 |
| `/target_person/offset` | `geometry_msgs/Point` | 영상 화면 중심 대비 정규화 오프셋 |

### UI 단축키 (`multi_tracker_node`)
- **마우스 좌클릭**: 사람 클릭 시 타겟 지정 / 빈 공간 클릭 시 타겟 해제
- **모델 전환**: `Q`(v8n) / `W`(11n) / `E`(26n) | **태스크**: `A`(Det) / `S`(Seg) / `F`(Pose)
- **알고리즘**: `1`~`5` (`5:Adaptive` 기본) | **장치 전환**: `D`(CPU/CUDA) | **오버레이**: `M`

---

## 5. 카메라 및 초기 위치 커스터마이징
- [config/phone_camera.yaml](config/phone_camera.yaml)에서 상단 카메라 위치(`extrinsics: x, y, z, roll, pitch, yaw`), 수평 화각(`camera_intrinsics: hfov_deg`), 초기 위치(`initial_pose`)를 수정한 뒤 `colcon build --symlink-install --packages-select person_tracker`로 반영할 수 있습니다.
