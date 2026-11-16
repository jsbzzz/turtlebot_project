# Person Tracker - 추적 성능 향상 연구 및 오프라인 테스트 공간

이 디렉터리는 실시간 주행 노드의 안정성을 유지하면서 **사람 재식별(ReID), 가려짐(Occlusion) 대응, 다중 센서 융합** 기법을 단계별로 실험·검증하기 위한 별도 연구 공간입니다.

## 1. 단계별 연구 로드맵

1. **[완료 - 본체 반영] 무채색(밝기 V) 구분 + 라이다/맵 3D 위치 게이팅**
   - `feature_extractors.py`: `S=0` 무채색 허용 + `H-S (72%) + V (28%)` 결합 히스토그램
   - `multi_tracker_core.py`: 타겟의 직전 `map`(또는 `base_link`) 위치와 경과 시간 기반 이동 가능 반경 게이팅
2. **[다음 연구 과제] 좌/우 가려짐 겹침 영역 제외 마스킹**
   - 다른 사람 바운딩 박스와 겹치는 좌/우 영역을 마스크에서 제거해, 옆 사람이 지나갈 때 옷 색깔 오염 방지
3. **[다음 연구 과제] Pose 골반/어깨 중심 & Seg 마스크 폭 기반 정밀 라이다 각도 매칭**
   - `bbox` 중심 대신 `Pose` 골반/몸통 중심과 `Seg` 실제 실루엣 폭으로 라이다 매칭 정확도 개선
4. **[실험 과제] 라이다 클러스터링 + 칼만 궤적 추적 선행 후 YOLO 융합**
   - 라이다로 먼저 3D 사람 후보 클러스터를 추적하고, YOLO 검출과 양방향 매칭
5. **[분석 과제] YOLO 백본 Feature Map / XAI 채널 활성화 분석**
   - Detect/Seg 백본의 특정 채널이 신체 부위(상체/다리 등)에 반응하는지 오프라인 시각화 스크립트로 확인

---

## 2. 실제 로봇과 분리된 오프라인 `ros2 bag` 테스트 방법

실제 터틀봇(`ROS_DOMAIN_ID=4`, `ROS_DISCOVERY_SERVER`)과 토픽이 절대 섞이지 않도록 **`ROS_DOMAIN_ID=99` + `unset ROS_DISCOVERY_SERVER`** 로 격리하여 테스트합니다.

### 보유 테스트 데이터 (`~/bag`)
- `pose_estimate` (267s): `/robot4/top_cam/image_raw/compressed` + `/robot4/scan` + `/robot4/map` + `/robot4/tf` (전체 파이프라인 검증용)
- `pose_estimate2` (161s): `/robot4/top_cam/image_raw/compressed` + `/robot4/scan` + `/robot4/tf`
- `one_man` (124s): `/robot4/top_cam/image_raw/compressed` (영상 단독 ReID/밝기 테스트용)

### 간편 실행
```bash
# [터미널 1] 오프라인 트래커 실행 (자동으로 격리된 DOMAIN_ID=99 사용)
bash ~/turtlebot4_ws/src/person_tracker/research/run_offline_bag_test.sh tracker

# [터미널 2] 녹화본 재생 (--clock)
bash ~/turtlebot4_ws/src/person_tracker/research/run_offline_bag_test.sh play ~/bag/pose_estimate

# [터미널 3] (선택) RViz2 동시 확인
bash ~/turtlebot4_ws/src/person_tracker/research/run_offline_bag_test.sh rviz
```
