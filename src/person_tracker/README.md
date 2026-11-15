# TurtleBot4 Person Tracker (`person_tracker`) 사용 설명서

터틀봇4(TurtleBot4) 및 DroidCam/웹캠 환경에서 **특정 사람 1명을 선택해 지속적으로 추적**하는 ROS 2 패키지입니다.
바운딩 박스 방식(`bbox_tracker_node`)과 세그멘테이션 마스킹 + 3부위 2중 기억 방식(`seg_tracker_node`) 두 가지를 모두 포함하고 있으며, 오프라인 실행을 위한 YOLO 가중치(`models/yolov8n.pt`, `models/yolov8n-seg.pt`)가 내장되어 있습니다.

---

## 1. 강의실 노트북에 설치 및 빌드하기

### 방법 A: 이미 `~/turtlebot4_ws`와 `env_setup.sh`(`rokey_venv`)가 세팅된 강의실 노트북일 때 (권장)
다운로드 받은 `person_tracker_portable.tar.gz` 파일이 `~/Downloads`에 있다고 가정할 때:

```bash
# 1. 워크스페이스 src 폴더에 압축 해제
tar -xzf ~/Downloads/person_tracker_portable.tar.gz -C ~/turtlebot4_ws/src

# 2. 환경 로드 및 심볼릭 링크(--symlink-install) 빌드
cd ~/turtlebot4_ws
source env_setup.sh
rm -rf build/person_tracker install/person_tracker
colcon build --symlink-install --packages-select person_tracker
source env_setup.sh
```
*(만약 실행 시 `lap` 모듈이 없다는 경고가 뜨면 `pip install lap` 한 줄만 실행해 주세요.)*

### 방법 B: 아무것도 안 깔린 초기 상태의 우분투 + ROS 2 노트북일 때 (자동 설치 스크립트)
```bash
tar -xzf ~/Downloads/person_tracker_portable.tar.gz -C ~/turtlebot4_ws/src
cd ~/turtlebot4_ws/src/person_tracker
bash setup_and_build.sh
source ~/turtlebot4_ws/env_tracker.sh
```

---

## 2. DroidCam (스마트폰 카메라) 연결 방법

실제 터틀봇4 카메라 대신 스마트폰 DroidCam을 연결해 테스트할 때는 아래 두 가지 중 편한 방법을 사용하세요.

### ① 우분투 DroidCam 클라이언트로 가상 웹캠(`/dev/video0`) 연결
```bash
# 가상 비디오 커널 모듈 활성화 (/dev/video0 생성)
sudo modprobe v4l2loopback

# DroidCam GUI 실행 후 스마트폰 IP 입력 및 [Connect] 클릭
droidcam &
```
*(또는 CLI로 바로 연결: `droidcam-cli 스마트폰IP 4747 &`)*

### ② 커널 모듈 없이 스마트폰 Wi-Fi IP로 직접 스트리밍 연결
`camera_pub` 실행 시 스마트폰 화면에 적힌 IP 주소를 파라미터로 바로 넘깁니다:
```bash
ros2 run person_tracker camera_pub --ros-args -p video_source:="http://스마트폰IP:4747/video"
```

---

## 3. 노드 실행 방법

새 터미널을 열 때마다 먼저 환경을 불러옵니다:
```bash
cd ~/turtlebot4_ws && source env_setup.sh
```

### ① 카메라 이미지 발행 노드 (`camera_pub`)
* `/oakd/rgb/image_raw` (원본) 및 `/oakd/rgb/image_raw/compressed` (압축본) 토픽을 동시에 발행합니다.
* 실제 터틀봇4와 연결된 경우에는 터틀봇4가 자체적으로 위 토픽을 발행하므로 `camera_pub`을 켤 필요가 없습니다.
```bash
ros2 run person_tracker camera_pub
```

### ② 바운딩 박스 기반 추적 노드 (`bbox_tracker_node` / `tracker_node`)
* `yolov8n.pt` 모델과 상/하의 HSV 히스토그램을 사용합니다.
```bash
ros2 run person_tracker bbox_tracker_node
```

### ③ 세그멘테이션 마스크 + 3부위 2중 기억 추적 노드 (`seg_tracker_node`) ⭐추천
* `yolov8n-seg.pt` 모델로 사람 실루엣만 마스킹하여 배경 노이즈를 100% 제거합니다.
* 상부/중부(바지)/하부(신발) 3부위 분할 및 **원본 기억(Anchor) + 최근 기억(Recent)** 2중 구조로 전신 $\leftrightarrow$ 하반신(무릎/발목) 변화를 모두 커버합니다.
```bash
ros2 run person_tracker seg_tracker_node
```

---

## 4. 화면(`cv2.imshow`) 조작법 및 단축키

| 조작 / 단축키 | 동작 설명 |
| :--- | :--- |
| **마우스 왼쪽 클릭 (사람 위)** | 해당 사람을 추적 타겟(`TARGET`)으로 지정 및 외형 특징 저장 |
| **마우스 왼쪽 클릭 (빈 배경)** | 타겟 지정 해제 및 저장된 특징 초기화 |
| **상단 탭 클릭** 또는 **숫자키 `1` ~ `5`** | 실시간 추적 알고리즘 전환 (타겟 자동 승계)<br>• `1:ByteTrack` (순수 위치 추적)<br>• `2:ColorReID / Seg-PartReID` (색상/부위별 마스킹 Re-ID)<br>• `3:BoT-SORT` (카메라 흔들림 보정 GMC + YOLO 특징 Re-ID)<br>• `4:BoT+Hist / BoT+SegReID` (3번 + 2번 완전체 결합)<br>• `5:Adaptive` (평상시 `SAFE` 고속 $\rightarrow$ 근접 시 `GUARD` 교차방어 $\rightarrow$ 이탈 시 `SEARCH` 자동 스위칭) |
| **키보드 `D` 키** | `GPU (CUDA:0)` $\leftrightarrow$ `CPU` 연산 장치 실시간 전환 (성능 비교용) |
| **키보드 `M` 키** *(Seg 전용)* | 세그멘테이션 반투명 실루엣 마스크 표시 `ON` / `OFF` |

---

## 5. 유용한 실행 파라미터 (ROS 2 Arguments)

```bash
# 원본 Raw 이미지(/oakd/rgb/image_raw)를 구독하고 싶을 때:
ros2 run person_tracker seg_tracker_node --ros-args -p use_compressed:=false

# 인식 거리를 늘리기 위해 입력 해상도를 320 -> 480 또는 640으로 올릴 때:
ros2 run person_tracker seg_tracker_node --ros-args -p imgsz:=480

# 로봇 네임스페이스(/robot0 등)가 설정된 환경에서 실행할 때:
ros2 run person_tracker seg_tracker_node --ros-args -r __ns:=/robot0
```

---

## 6. 발행되는 제어 연동 토픽 (`/target_person/offset`)
추적 중인 타겟이 화면에 있을 때마다 `geometry_msgs/msg/Point` 타입으로 실시간 발행됩니다:
* `Point.x`: 화면 가로 중심 대비 오프셋 (`-1.0` 왼쪽 끝 ~ `+1.0` 오른쪽 끝 $\rightarrow$ 회전 제어 `angular.z`용)
* `Point.y`: 화면 세로 중심 대비 오프셋 (`-1.0` 위쪽 끝 ~ `+1.0` 아래쪽 끝)
* `Point.z`: 화면 전체 대비 타겟 박스 면적 비율 (`0.0` ~ `1.0` $\rightarrow$ 거리 추정 및 전후진 제어 `linear.x`용)
