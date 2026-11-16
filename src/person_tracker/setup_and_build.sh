#!/usr/bin/env bash
# ==============================================================================
# person_tracker 타 기기 이식용 원클릭 설치 및 빌드 스크립트
# 사용법:
#   cd <워크스페이스>/src/person_tracker
#   bash setup_and_build.sh
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="${VENV_DIR:-$HOME/venvs/rokey_venv}"

echo "=========================================================="
echo " [1/5] ROS 2 배포판 감지 및 시스템 의존성 확인"
echo "=========================================================="
ROS_DISTRO_FOUND=""
for d in jazzy humble iron rolling; do
    if [ -f "/opt/ros/$d/setup.bash" ]; then
        ROS_DISTRO_FOUND="$d"
        break
    fi
done

if [ -z "$ROS_DISTRO_FOUND" ]; then
    echo "[ERROR] /opt/ros/ 아래에서 ROS 2 설치를 찾을 수 없습니다."
    exit 1
fi
echo " -> 감지된 ROS 2 배포판: $ROS_DISTRO_FOUND"
source "/opt/ros/$ROS_DISTRO_FOUND/setup.bash"

sudo apt-get update
sudo apt-get install -y \
    python3-venv \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    "ros-${ROS_DISTRO_FOUND}-cv-bridge" \
    "ros-${ROS_DISTRO_FOUND}-vision-opencv" \
    "ros-${ROS_DISTRO_FOUND}-image-transport" \
    v4l2loopback-dkms || true

echo "=========================================================="
echo " [2/5] Python 가상환경 구성 ($VENV_DIR)"
echo "=========================================================="
# --system-site-packages 옵션으로 생성하여 ROS 2의 rclpy, cv_bridge를 가상환경 내부에서 그대로 공유
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    mkdir -p "$(dirname "$VENV_DIR")"
    python3 -m venv --system-site-packages "$VENV_DIR"
    echo " -> 새 가상환경 생성 완료: $VENV_DIR"
else
    echo " -> 기존 가상환경 재사용: $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "=========================================================="
echo " [3/5] PyTorch 및 YOLO/추적 Python 패키지 설치"
echo "=========================================================="
python -m pip install --upgrade pip
if ! python -c "import torch" &>/dev/null; then
    echo " -> PyTorch 설치 중..."
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 || \
        python -m pip install torch torchvision
fi
python -m pip install -r "$SCRIPT_DIR/requirements.txt"

echo "=========================================================="
echo " [4/5] colcon build --symlink-install 빌드 수행"
echo "=========================================================="
cd "$WS_DIR"
rm -rf "$WS_DIR/build/person_tracker" "$WS_DIR/install/person_tracker"
colcon build --symlink-install --packages-select person_tracker

echo "=========================================================="
echo " [5/5] 실행용 환경 로드 스크립트 생성 ($WS_DIR/env_tracker.sh)"
echo "=========================================================="
cat <<EOF > "$WS_DIR/env_tracker.sh"
#!/usr/bin/env bash
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
fi
source "/opt/ros/$ROS_DISTRO_FOUND/setup.bash"
source "$WS_DIR/install/setup.bash"
if [ -f "/etc/turtlebot4_discovery/setup.bash" ]; then
    source "/etc/turtlebot4_discovery/setup.bash"
fi
export RMW_IMPLEMENTATION=\${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}
export ROS_DOMAIN_ID=\${ROS_DOMAIN_ID:-4}
echo "[person_tracker] 환경 로드 완료 (ROS_DISTRO=$ROS_DISTRO_FOUND, ROS_DOMAIN_ID=\$ROS_DOMAIN_ID)"
EOF
chmod +x "$WS_DIR/env_tracker.sh"

echo ""
echo "✅ 설치 및 빌드가 모두 완료되었습니다!"
echo "실행 방법:"
echo "  source $WS_DIR/env_tracker.sh"
echo "  # [원클릭 통합 실행] 맵 + init_pose + 노트북 웹캠(0) + 트래커:"
echo "  ros2 launch person_tracker bringup.launch.py video_source:=0"
echo "  # (RViz2까지 같이 띄우려면 run_rviz:=true 추가)"
