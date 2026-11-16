import math
import os
import sys
import time

from person_tracker.env_bootstrap import bootstrap_venv, resolve_image_topic
bootstrap_venv()

import cv2
import numpy as np
import psutil
import torch
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge

from typing import Optional, Tuple
from person_tracker.multi_tracker_core import MultiModelTrackerCore, UnifiedTrackedPerson
from person_tracker.lidar_person_localizer import LidarPersonLocalizer, TargetLocalizationResult


class MultiModelTrackerNode(Node):
    """
    9종 모델 (YOLOv8n / YOLO11n / YOLO26n x Detect / Segment / Pose) 및
    5종 추적 알고리즘(ByteTrack / Task-ReID / BoT-SORT / BoT+ReID / Adaptive)을
    2단 상단 탭 클릭 또는 단축키로 실시간 교체하며 비교할 수 있는 통합 벤치마크 노드
    """

    ROW1_HEIGHT = 36   # 1단 바: 모델 버전(Q/W/E) + 태스크(A/S/F) 선택
    ROW2_HEIGHT = 36   # 2단 바: 추적 알고리즘(1~5) 선택
    TOTAL_BAR_HEIGHT = ROW1_HEIGHT + ROW2_HEIGHT

    # Pose 스켈레톤 연결선 정의 (부위별 색상 구분)
    POSE_EDGES = [
        # 상반신 (Torso: 파란/하늘색)
        (5, 6, (255, 200, 50)),
        (5, 11, (255, 200, 50)),
        (6, 12, (255, 200, 50)),
        (11, 12, (255, 200, 50)),
        (5, 7, (255, 170, 50)),
        (7, 9, (255, 170, 50)),
        (6, 8, (255, 170, 50)),
        (8, 10, (255, 170, 50)),
        # 중하반신 (허벅지/바지: 주황색)
        (11, 13, (0, 165, 255)),
        (12, 14, (0, 165, 255)),
        # 최하단 (정강이/신발: 보라/마젠타색)
        (13, 15, (255, 50, 255)),
        (14, 16, (255, 50, 255)),
    ]

    def __init__(self):
        super().__init__('multi_person_tracker_node')

        self.declare_parameter('use_compressed', True)
        self.declare_parameter('image_topic', '')
        self.declare_parameter('depth_topic', '/oakd/stereo/image_raw')
        self.declare_parameter('initial_family', 3)  # 1:v8n, 2:11n, 3:26n
        self.declare_parameter('initial_task', 3)    # 1:Det, 2:Seg, 3:Pose
        self.declare_parameter('initial_mode', 5)    # 1~5
        self.declare_parameter('imgsz', 320)
        self.declare_parameter('conf', 0.15)
        self.declare_parameter('iou', 0.55)
        self.declare_parameter('show_gui', True)
        self.declare_parameter('draw_overlay', True)
        self.declare_parameter('auto_lock_first', False)
        self.declare_parameter('device', '')

        ns = self.get_namespace()
        ns_prefix = '' if ns == '/' else ns
        use_compressed = bool(self.get_parameter('use_compressed').value)
        custom_topic = str(self.get_parameter('image_topic').value)
        image_topic = resolve_image_topic(ns, use_compressed, custom_topic)

        self.depth_topic = ns_prefix + str(self.get_parameter('depth_topic').value)
        family_idx = int(self.get_parameter('initial_family').value)
        task_idx = int(self.get_parameter('initial_task').value)
        initial_mode = int(self.get_parameter('initial_mode').value)
        imgsz = int(self.get_parameter('imgsz').value)
        conf = float(self.get_parameter('conf').value)
        iou = float(self.get_parameter('iou').value)
        self.show_gui = bool(self.get_parameter('show_gui').value)
        self.draw_overlay = bool(self.get_parameter('draw_overlay').value)
        self.auto_lock_first = bool(self.get_parameter('auto_lock_first').value)
        device = str(self.get_parameter('device').value)

        self.tracker = MultiModelTrackerCore(
            family_idx=family_idx,
            task_idx=task_idx,
            mode=initial_mode,
            imgsz=imgsz,
            device=device,
            conf=conf,
            iou=iou,
        )
        self.bridge = CvBridge()
        self.frame_width = 640

        self.prev_time = time.time()
        self.cam_fps = 0.0
        self.ros_lag_ms = 0.0
        self.process = psutil.Process(os.getpid())
        self.cpu_percent = 0.0
        self.ram_mb = 0.0
        self.last_sys_check = 0.0

        if use_compressed:
            self.sub = self.create_subscription(
                CompressedImage,
                image_topic,
                self.compressed_callback,
                qos_profile_sensor_data,
            )
        else:
            self.sub = self.create_subscription(
                Image,
                image_topic,
                self.raw_callback,
                qos_profile_sensor_data,
            )

        self.target_pub = self.create_publisher(Point, '/target_person/offset', 10)
        self.localizer = LidarPersonLocalizer(self)
        self.last_loc_result: Optional[TargetLocalizationResult] = None
        self.last_loc_log_time = 0.0
        self.image_topic = image_topic
        self.last_frame_wall_time = 0.0
        self.last_display_frame: Optional[np.ndarray] = None
        self.win_name = 'TurtleBot4 Multi-Model Benchmark (v8 / v11 / v26 x Det / Seg / Pose)'
        self.info_win_name = 'Tracker Status & Coordinates'

        if self.show_gui:
            cv2.namedWindow(self.win_name)
            cv2.setMouseCallback(self.win_name, self.on_mouse_click)
            cv2.namedWindow(self.info_win_name)

        self.get_logger().info(
            f'MultiModelTrackerNode 시작됨 | 토픽: {image_topic} | 모델: {self.tracker.current_model_name} | '
            f'모드: {MultiModelTrackerCore.MODE_NAMES[self.tracker.mode]} | 장치: {self.tracker.device}'
        )

    def on_mouse_click(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # 1단 바 클릭: 왼쪽 절반(3칸)은 모델 버전(v8n/11n/26n), 오른쪽 절반(3칸)은 태스크(Det/Seg/Pose)
        if y <= self.ROW1_HEIGHT:
            slot_w = max(1, self.frame_width // 6)
            slot_idx = min(5, max(0, x // slot_w))
            if slot_idx < 3:
                new_model = self.tracker.set_model(family_idx=slot_idx + 1)
                self.get_logger().info(f'[모델 버전 변경] -> {new_model}')
            else:
                new_model = self.tracker.set_model(task_idx=(slot_idx - 3) + 1)
                self.get_logger().info(f'[모델 태스크 변경] -> {new_model}')
            return

        # 2단 바 클릭: 추적 알고리즘 1~5 변경
        if y <= self.TOTAL_BAR_HEIGHT:
            tab_w = max(1, self.frame_width // 5)
            clicked_mode = min(5, max(1, (x // tab_w) + 1))
            self.tracker.set_mode(clicked_mode)
            self.get_logger().info(
                f'[추적 알고리즘 변경] -> {MultiModelTrackerCore.MODE_NAMES[clicked_mode]}'
            )
            return

        # 영상 영역 클릭: 추적 타겟 지정 / 해제
        selected_id = self.tracker.select_target_by_point(x, y)
        if selected_id is not None:
            self.get_logger().info(
                f'[타겟 선택] ID={selected_id} (현재 모델: {self.tracker.current_model_name})'
            )
        else:
            self.get_logger().info('[타겟 해제] 추적을 중지합니다.')

    def _update_system_metrics(self, msg_stamp):
        now = time.time()
        dt = now - self.prev_time
        self.prev_time = now
        if dt > 0:
            inst_fps = 1.0 / dt
            self.cam_fps = 0.9 * self.cam_fps + 0.1 * inst_fps if self.cam_fps > 0 else inst_fps

        if msg_stamp.sec > 0:
            now_ros = self.get_clock().now().nanoseconds / 1e9
            stamp_sec = msg_stamp.sec + msg_stamp.nanosec / 1e9
            lag = max(0.0, (now_ros - stamp_sec) * 1000.0)
            self.ros_lag_ms = 0.85 * self.ros_lag_ms + 0.15 * lag if self.ros_lag_ms > 0 else lag

        if now - self.last_sys_check >= 0.5:
            self.last_sys_check = now
            self.cpu_percent = psutil.cpu_percent(interval=None)
            self.ram_mb = self.process.memory_info().rss / (1024 * 1024)

    def _draw_task_overlays(self, frame: np.ndarray, detections: list[UnifiedTrackedPerson]):
        if not self.draw_overlay or len(detections) == 0:
            return

        task = self.tracker.current_task_name

        # Segment 태스크: 반투명 폴리곤 마스크 렌더링
        if task == 'segment':
            mask_overlay = frame.copy()
            for p in detections:
                if p.mask_poly is not None and len(p.mask_poly) >= 3:
                    pts = p.mask_poly.astype(np.int32)
                    fill_color = (0, 230, 80) if p.is_target else (140, 120, 90)
                    cv2.fillPoly(mask_overlay, [pts], fill_color)
                    cv2.polylines(
                        frame, [pts], isClosed=True, color=fill_color, thickness=2 if p.is_target else 1
                    )
            cv2.addWeighted(mask_overlay, 0.28, frame, 0.72, 0, frame)

        # Pose 태스크: 17개 관절 스켈레톤 및 가시성 키포인트 렌더링
        elif task == 'pose':
            for p in detections:
                if p.keypoints is None or len(p.keypoints) < 17:
                    continue
                kp = p.keypoints
                for i1, i2, edge_color in self.POSE_EDGES:
                    if kp[i1, 2] >= 0.35 and kp[i2, 2] >= 0.35:
                        pt1 = (int(kp[i1, 0]), int(kp[i1, 1]))
                        pt2 = (int(kp[i2, 0]), int(kp[i2, 1]))
                        color = edge_color if p.is_target else (150, 150, 150)
                        cv2.line(frame, pt1, pt2, color, 2 if p.is_target else 1, cv2.LINE_AA)

                for idx in (5, 6, 11, 12, 13, 14, 15, 16):
                    if kp[idx, 2] >= 0.35:
                        pt = (int(kp[idx, 0]), int(kp[idx, 1]))
                        cv2.circle(
                            frame,
                            pt,
                            4 if p.is_target else 2,
                            (0, 255, 255) if p.is_target else (180, 180, 180),
                            -1,
                        )

    def _draw_hud(self, frame: np.ndarray, num_persons: int):
        h, w = frame.shape[:2]
        self.frame_width = w

        # 1. 상단 1열: 왼쪽 3칸 = 모델 버전(Q:v8n, W:11n, E:26n), 오른쪽 3칸 = 태스크(A:Det, S:Seg, F:Pose)
        slot_w = w // 6
        family_keys = {1: 'Q', 2: 'W', 3: 'E'}
        task_keys = {1: 'A', 2: 'S', 3: 'F'}

        for f_idx in (1, 2, 3):
            x1 = (f_idx - 1) * slot_w
            x2 = f_idx * slot_w
            is_act = (self.tracker.family_idx == f_idx)
            bg = (180, 100, 20) if is_act else (38, 38, 38)
            cv2.rectangle(frame, (x1, 0), (x2, self.ROW1_HEIGHT), bg, -1)
            cv2.rectangle(frame, (x1, 0), (x2, self.ROW1_HEIGHT), (85, 85, 85), 1)
            short_name = MultiModelTrackerCore.FAMILIES[f_idx][0]
            label = f'[{family_keys[f_idx]}] YOLO{short_name}'
            cv2.putText(
                frame,
                label,
                (x1 + 6, 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255) if is_act else (180, 180, 180),
                2 if is_act else 1,
                cv2.LINE_AA,
            )

        for t_idx in (1, 2, 3):
            slot = 3 + (t_idx - 1)
            x1 = slot * slot_w
            x2 = w if t_idx == 3 else (slot + 1) * slot_w
            is_act = (self.tracker.task_idx == t_idx)
            bg = (140, 50, 140) if is_act else (38, 38, 38)
            cv2.rectangle(frame, (x1, 0), (x2, self.ROW1_HEIGHT), bg, -1)
            cv2.rectangle(frame, (x1, 0), (x2, self.ROW1_HEIGHT), (85, 85, 85), 1)
            t_label = MultiModelTrackerCore.TASKS[t_idx][0]
            label = f'[{task_keys[t_idx]}] {t_label}'
            cv2.putText(
                frame,
                label,
                (x1 + 10, 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (255, 255, 255) if is_act else (180, 180, 180),
                2 if is_act else 1,
                cv2.LINE_AA,
            )

        # 2. 상단 2열: 추적 알고리즘 1~5 탭
        tab_w = w // 5
        for m_idx in range(1, 6):
            x1 = (m_idx - 1) * tab_w
            x2 = w if m_idx == 5 else m_idx * tab_w
            is_act = (self.tracker.mode == m_idx)
            bg = (40, 155, 40) if is_act else (50, 50, 50)
            cv2.rectangle(frame, (x1, self.ROW1_HEIGHT), (x2, self.TOTAL_BAR_HEIGHT), bg, -1)
            cv2.rectangle(
                frame, (x1, self.ROW1_HEIGHT), (x2, self.TOTAL_BAR_HEIGHT), (90, 90, 90), 1
            )
            label = MultiModelTrackerCore.MODE_NAMES[m_idx]
            cv2.putText(
                frame,
                label,
                (x1 + 6, self.ROW1_HEIGHT + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (255, 255, 255) if is_act else (185, 185, 185),
                2 if is_act else 1,
                cv2.LINE_AA,
            )

        # 3. 영상 좌측 상단에 작은 '맵 기준 좌표축 나침반' 표시 (화면 가림 최소화)
        cam_status = self.localizer.get_camera_pose_status()
        cx_cmp, cy_cmp = 48, self.TOTAL_BAR_HEIGHT + 52
        radius = 28
        overlay = frame.copy()
        cv2.circle(overlay, (cx_cmp, cy_cmp), radius + 6, (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.circle(frame, (cx_cmp, cy_cmp), radius, (110, 110, 110), 1, cv2.LINE_AA)

        if cam_status.map_yaw_deg is not None:
            psi = math.radians(cam_status.map_yaw_deg)
            # Map +X 축을 현재 카메라 화면 좌표(위=-Y_img, 좌=-X_img)로 회전 투영
            x_end = (int(cx_cmp + radius * 0.85 * math.sin(psi)), int(cy_cmp - radius * 0.85 * math.cos(psi)))
            # Map +Y 축은 Map +X보다 +90도(psi - pi/2)
            y_end = (int(cx_cmp + radius * 0.85 * math.sin(psi - math.pi * 0.5)), int(cy_cmp - radius * 0.85 * math.cos(psi - math.pi * 0.5)))
            cv2.arrowedLine(frame, (cx_cmp, cy_cmp), x_end, (60, 70, 255), 2, cv2.LINE_AA, tipLength=0.28)
            cv2.arrowedLine(frame, (cx_cmp, cy_cmp), y_end, (60, 230, 80), 2, cv2.LINE_AA, tipLength=0.28)
            cv2.putText(frame, 'X', (x_end[0] + 3, x_end[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (80, 90, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, 'Y', (y_end[0] + 3, y_end[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (80, 240, 100), 1, cv2.LINE_AA)
            cmp_title = f'Map Axes ({cam_status.map_source})'
        else:
            cv2.arrowedLine(frame, (cx_cmp, cy_cmp), (cx_cmp, cy_cmp - int(radius * 0.8)), (60, 70, 255), 2, cv2.LINE_AA, tipLength=0.28)
            cv2.arrowedLine(frame, (cx_cmp, cy_cmp), (cx_cmp - int(radius * 0.8), cy_cmp), (60, 230, 80), 2, cv2.LINE_AA, tipLength=0.28)
            cmp_title = 'Cam Axes (No Map)'

        cv2.putText(
            frame,
            cmp_title,
            (10, self.TOTAL_BAR_HEIGHT + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        # 4. 별도 정보 창(Tracker Status & Coordinates)에 좌표 및 성능 표시
        self._render_info_window(num_persons, cam_status)

    def _draw_target_camera_axes(self, frame: np.ndarray, person: UnifiedTrackedPerson):
        """타겟 중심에 카메라 기준(top_cam_link) 3D 좌표축(+X 정면, +Y 왼쪽, +Z 위) 방향 투영 표시."""
        loc = self.last_loc_result
        if loc is None:
            return
        h, w = frame.shape[:2]
        half_w = max(1.0, w * 0.5)
        fx = half_w / math.tan(math.radians(self.localizer.hfov_deg * 0.5))

        cx_p, cy_p = person.center
        tx, ty, tz = loc.cam_xyz
        tx = max(0.35, tx)
        axis_len = 0.35

        def proj(dx: float, dy: float, dz: float) -> Tuple[int, int]:
            px = max(0.20, tx + dx)
            py = ty + dy
            pz = tz + dz
            u = int(half_w - fx * (py / px))
            v = int(cy_p - fx * (pz / px))
            return max(4, min(w - 4, u)), max(self.TOTAL_BAR_HEIGHT + 4, min(h - 4, v))

        origin = (cx_p, cy_p)
        ax_x = proj(axis_len, 0.0, 0.0)   # +X_cam (정면 깊이)
        ax_y = proj(0.0, axis_len, 0.0)   # +Y_cam (왼쪽)
        ax_z = proj(0.0, 0.0, axis_len)   # +Z_cam (위쪽)

        cv2.arrowedLine(frame, origin, ax_x, (60, 70, 255), 2, cv2.LINE_AA, tipLength=0.25)
        cv2.arrowedLine(frame, origin, ax_y, (60, 235, 80), 2, cv2.LINE_AA, tipLength=0.25)
        cv2.arrowedLine(frame, origin, ax_z, (255, 160, 40), 2, cv2.LINE_AA, tipLength=0.25)
        cv2.putText(frame, '+X', (ax_x[0] + 4, ax_x[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 90, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, '+Y', (ax_y[0] + 4, ax_y[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 245, 100), 1, cv2.LINE_AA)
        cv2.putText(frame, '+Z', (ax_z[0] + 4, ax_z[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 180, 70), 1, cv2.LINE_AA)

    def _render_info_window(self, num_persons: int, cam_status=None, status_banner: str = ''):
        """영상 화면을 가리지 않도록 별도 창에 성능·라이다·로봇/맵 좌표를 표시."""
        if not self.show_gui:
            return
        if cam_status is None:
            cam_status = self.localizer.get_camera_pose_status()

        m = self.tracker.metrics
        vram_str = ''
        if 'CUDA' in m.device and torch.cuda.is_available():
            vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            vram_str = f' | VRAM:{vram_mb:.0f}MB'

        adapt_str = f' [{m.adaptive_state}]' if self.tracker.mode == 5 else ''

        if cam_status.map_xyz is not None:
            mx, my, _ = cam_status.map_xyz
            yaw_d = cam_status.map_yaw_deg or 0.0
            cam_pose_str = (
                f'Cam(Map): X={mx:+.2f}m Y={my:+.2f}m ({yaw_d:+.0f}deg) '
                f'[{cam_status.map_source}] | Scan:{cam_status.lidar_hz:.1f}Hz'
            )
        elif cam_status.base_xyz is not None:
            bx, by, bz = cam_status.base_xyz
            cam_pose_str = f'Cam(Bot): x={bx:+.2f}m y={by:+.2f}m z={bz:+.2f}m | Scan:{cam_status.lidar_hz:.1f}Hz'
        else:
            cam_pose_str = f'Cam TF: Waiting... | LiDAR Scan:{cam_status.lidar_hz:.1f}Hz'

        loc = self.last_loc_result
        if loc is not None:
            bx, by, _ = loc.base_xyz
            cx_m, cy_m, _ = loc.cam_xyz
            cam_tgt_line = f'Target(Cam): x={cx_m:+.2f}m y={cy_m:+.2f}m ({loc.cam_bearing_deg:+.1f}deg)'
            bot_line = (
                f'Target(Bot): x={bx:+.2f}m y={by:+.2f}m '
                f'({loc.base_dist_m:.2f}m, {loc.base_angle_deg:+.1f}deg) [{loc.source}]'
            )
            bot_color = (80, 255, 120) if loc.source == 'LiDAR' else (80, 220, 255)
            if loc.map_xyz is not None:
                tmx, tmy, _ = loc.map_xyz
                map_line = f'Target(Map): X={tmx:+.2f}m Y={tmy:+.2f}m [{loc.map_source}]'
                map_color = (120, 255, 255)
            else:
                map_line = 'Target(Map): Waiting for TF (run loc 4)'
                map_color = (180, 180, 180)
        else:
            cam_tgt_line = 'Target(Cam): -'
            bot_line = 'Target(Bot): Click person to lock target'
            bot_color = (170, 170, 170)
            map_line = 'Target(Map): -'
            map_color = (150, 150, 150)

        lines = [
            (f'Model  : {m.model_label} | {m.device} [D:Dev, M:Draw]', (0, 255, 255)),
            (f'Cam FPS: {self.cam_fps:4.1f} | Max Infer: {m.max_infer_fps:5.1f} FPS', (0, 255, 0)),
            (f'Total  : {m.total_ms:5.1f} ms (ROS Lag: {self.ros_lag_ms:4.0f} ms)', (255, 255, 255)),
            (f' - YOLO ({m.task[:3]}) : {m.inference_ms:5.1f} ms (Pre:{m.preprocess_ms:3.1f})', (200, 220, 255)),
            (f' - Track+ReID : {m.post_track_ms + m.reid_ms:5.1f} ms (ReID:{m.reid_ms:3.1f})', (200, 220, 255)),
            (f'Sys CPU: {self.cpu_percent:4.1f}% | RAM:{self.ram_mb:.0f}MB{vram_str}', (180, 255, 180)),
            (f'Persons: {num_persons} | Target:{self.tracker.target_id}{adapt_str}', (255, 220, 120)),
            (cam_pose_str, (220, 210, 255)),
            (cam_tgt_line, (210, 240, 255)),
            (bot_line, bot_color),
            (map_line, map_color),
        ]
        if status_banner:
            lines.append((status_banner, (0, 200, 255)))

        panel = np.full((280, 540, 3), 24, dtype=np.uint8)
        for idx, (text, color) in enumerate(lines):
            cv2.putText(
                panel,
                text,
                (14, 24 + idx * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
                cv2.LINE_AA,
            )
        cv2.imshow(self.info_win_name, panel)

    def compressed_callback(self, msg: CompressedImage):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is not None:
                self.handle_frame(frame, msg.header.stamp)
        except Exception as exc:
            self.get_logger().warn(f'Compressed image decode error: {exc}', throttle_duration_sec=5.0)

    def raw_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.handle_frame(frame, msg.header.stamp)
        except Exception as exc:
            self.get_logger().warn(f'Raw image convert error: {exc}', throttle_duration_sec=5.0)

    def _process_key(self, key: int):
        if key in (ord('1'), ord('2'), ord('3'), ord('4'), ord('5')):
            self.tracker.set_mode(int(chr(key)))
        elif key in (ord('q'), ord('Q')):
            self.tracker.set_model(family_idx=1)
        elif key in (ord('w'), ord('W')):
            self.tracker.set_model(family_idx=2)
        elif key in (ord('e'), ord('E')):
            self.tracker.set_model(family_idx=3)
        elif key in (ord('a'), ord('A')):
            self.tracker.set_model(task_idx=1)
        elif key in (ord('s'), ord('S')):
            self.tracker.set_model(task_idx=2)
        elif key in (ord('f'), ord('F')):
            self.tracker.set_model(task_idx=3)
        elif key in (ord('d'), ord('D')):
            self.tracker.toggle_device()
        elif key in (ord('m'), ord('M')):
            self.draw_overlay = not self.draw_overlay

    def refresh_idle_gui(self):
        """토픽이 아직 안 들어오거나 bag 재생이 멈췄을 때도 에러/창 멈춤 없이 대기 화면 유지."""
        if not self.show_gui:
            return
        now = time.time()
        # 최근 0.4초 이내에 새 프레임이 들어왔으면 handle_frame에서 이미 그렸으므로 생략
        if now - self.last_frame_wall_time < 0.4:
            return

        if self.last_display_frame is not None:
            idle_frame = self.last_display_frame.copy()
            status_txt = f'PAUSED / WAITING TOPIC ({self.image_topic})'
        else:
            idle_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            status_txt = f'WAITING FOR TOPIC: {self.image_topic}'

        h, w = idle_frame.shape[:2]
        cv2.rectangle(idle_frame, (0, h - 34), (w, h), (20, 20, 20), -1)
        cv2.putText(
            idle_frame,
            status_txt,
            (12, h - 11),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(self.win_name, idle_frame)
        self._render_info_window(0, status_banner=status_txt)
        key = cv2.waitKey(1) & 0xFF
        if key != 255:
            self._process_key(key)

    def _estimate_candidate_spatial(self, person: UnifiedTrackedPerson, w: int, h: int):
        res = self.localizer.localize_target(person, w, h, publish=False)
        if res.map_xyz is not None:
            return (res.map_xyz[0], res.map_xyz[1]), 'map', res.source
        return (res.base_xyz[0], res.base_xyz[1]), 'base_link', res.source

    def handle_frame(self, frame: np.ndarray, msg_stamp):
        h, w = frame.shape[:2]
        self.last_frame_wall_time = time.time()
        detections, target = self.tracker.process_frame(
            frame,
            auto_lock_first=self.auto_lock_first,
            spatial_estimator=self._estimate_candidate_spatial,
        )
        self._update_system_metrics(msg_stamp)

        if target is not None:
            msg = Point()
            msg.x = float(target.norm_offset[0])
            msg.y = float(target.norm_offset[1])
            msg.z = float(target.area_ratio)
            self.target_pub.publish(msg)

            loc_res = self.localizer.localize_target(target, w, h)
            self.last_loc_result = loc_res

            now_t = time.time()
            if now_t - self.last_loc_log_time >= 1.0:
                self.last_loc_log_time = now_t
                bx, by, _ = loc_res.base_xyz
                if loc_res.map_xyz is not None:
                    mx, my, _ = loc_res.map_xyz
                    map_str = f'X={mx:+.2f}m, Y={my:+.2f}m [{loc_res.map_source}]'
                else:
                    map_str = '맵(map) TF 대기중'
                self.get_logger().info(
                    f'[타겟 위치 ID={loc_res.track_id} | {loc_res.source}] '
                    f'로봇기준(base_link): x={bx:+.2f}m, y={by:+.2f}m '
                    f'(거리 {loc_res.base_dist_m:.2f}m, 각도 {loc_res.base_angle_deg:+.1f}°) | '
                    f'맵기준(map): {map_str}'
                )
        else:
            self.last_loc_result = None

        if self.show_gui:
            self._draw_task_overlays(frame, detections)
            show_sim = self.tracker.mode in (2, 4, 5)

            for p in detections:
                x1, y1, x2, y2 = p.bbox
                cx, cy = p.center
                sim_str = f' sim:{p.similarity:.2f}' if show_sim else ''

                if p.is_target:
                    box_color = (
                        (0, 200, 255)
                        if (self.tracker.mode == 5 and self.tracker.metrics.adaptive_state == 'GUARD')
                        else (0, 255, 0)
                    )
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.circle(frame, p.bottom_center, 4, (255, 0, 255), -1)

                    # 현재 인식된 신체 부위(U:상반신, M:허벅지, L:종아리/신발) 표시
                    u_ok, m_ok, l_ok = p.visible_parts
                    parts_tag = f'[{"U" if u_ok else "-"}{"M" if m_ok else "-"}{"L" if l_ok else "-"}]'
                    cv2.putText(
                        frame,
                        f'TARGET ID:{p.track_id} {parts_tag}{sim_str}',
                        (x1, max(self.TOTAL_BAR_HEIGHT + 18, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        box_color,
                        2,
                    )
                    if self.last_loc_result is not None:
                        lr = self.last_loc_result
                        bx, by, _ = lr.base_xyz
                        coord_txt = f'Bot:({bx:+.2f},{by:+.2f})m {lr.base_dist_m:.2f}m/{lr.base_angle_deg:+.0f}deg [{lr.source}]'
                        if lr.map_xyz is not None:
                            mx, my, _ = lr.map_xyz
                            coord_txt += f' | Map:({mx:+.2f},{my:+.2f})m'
                        cv2.putText(
                            frame,
                            coord_txt,
                            (x1, min(h - 8, y2 + 16)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.42,
                            (50, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (150, 150, 150), 1)
                    cv2.putText(
                        frame,
                        f'ID:{p.track_id}{sim_str}',
                        (x1, max(self.TOTAL_BAR_HEIGHT + 15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.44,
                        (150, 150, 150),
                        1,
                    )

            self._draw_hud(frame, len(detections))
            self.last_display_frame = frame.copy()
            cv2.imshow(self.win_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                self._process_key(key)


def main(args=None):
    rclpy.init(args=args)
    node = MultiModelTrackerNode()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            node.refresh_idle_gui()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
