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

from person_tracker.seg_tracker_core import SegPersonTrackerCore


class SegPersonTrackerNode(Node):
    """
    YOLOv8n-seg 인스턴스 세그멘테이션 마스킹 + 부위별 2중 기억(Anchor+Recent)을 사용하여
    배경 노이즈 없이 전신<->하반신 변화에도 강건하게 특정 인물을 추적하는 ROS 2 노드
    """

    UI_BAR_HEIGHT = 42
    NUM_MODES = 5

    def __init__(self):
        super().__init__('seg_person_tracker_node')

        self.declare_parameter('use_compressed', True)
        self.declare_parameter('image_topic', '')
        self.declare_parameter('depth_topic', '/oakd/stereo/image_raw')
        self.declare_parameter('model_name', 'yolov8n-seg.pt')
        self.declare_parameter('imgsz', 320)
        self.declare_parameter('conf', 0.15)
        self.declare_parameter('iou', 0.55)
        self.declare_parameter('show_gui', True)
        self.declare_parameter('draw_mask', True)       # 세그멘테이션 실루엣 마스크 표시 여부 (키보드 M으로 토글)
        self.declare_parameter('auto_lock_first', False)
        self.declare_parameter('initial_mode', 5)
        self.declare_parameter('device', '')

        ns = self.get_namespace()
        ns_prefix = '' if ns == '/' else ns
        use_compressed = bool(self.get_parameter('use_compressed').value)
        custom_topic = str(self.get_parameter('image_topic').value)
        image_topic = resolve_image_topic(ns, use_compressed, custom_topic)

        self.depth_topic = ns_prefix + str(self.get_parameter('depth_topic').value)
        model_name = str(self.get_parameter('model_name').value)
        imgsz = int(self.get_parameter('imgsz').value)
        conf = float(self.get_parameter('conf').value)
        iou = float(self.get_parameter('iou').value)
        self.show_gui = bool(self.get_parameter('show_gui').value)
        self.draw_mask = bool(self.get_parameter('draw_mask').value)
        self.auto_lock_first = bool(self.get_parameter('auto_lock_first').value)
        initial_mode = int(self.get_parameter('initial_mode').value)
        device = str(self.get_parameter('device').value)

        self.tracker = SegPersonTrackerCore(
            model_name=model_name,
            imgsz=imgsz,
            mode=initial_mode,
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
        self.win_name = 'TurtleBot4 SEG Person Tracker (Mask + Part-Memory)'

        if self.show_gui:
            cv2.namedWindow(self.win_name)
            cv2.setMouseCallback(self.win_name, self.on_mouse_click)

        self.get_logger().info(
            f'SegPersonTrackerNode 시작됨 | 모델: {model_name} | 토픽: {image_topic} | 초기 모드: {SegPersonTrackerCore.MODE_NAMES.get(initial_mode)}'
        )

    def on_mouse_click(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if y <= self.UI_BAR_HEIGHT:
            tab_width = max(1, self.frame_width // self.NUM_MODES)
            clicked_mode = min(self.NUM_MODES, max(1, (x // tab_width) + 1))
            self.tracker.set_mode(clicked_mode)
            self.get_logger().info(
                f'[모드 전환] {SegPersonTrackerCore.MODE_NAMES[clicked_mode]} 적용됨'
            )
            return

        selected_id = self.tracker.select_target_by_point(x, y)
        if selected_id is not None:
            self.get_logger().info(
                f'[타겟 선택] ID={selected_id} (세그멘테이션 3부위 원본 마스크 뱅크 저장 완료)'
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

    def _draw_hud(self, frame: np.ndarray, num_persons: int):
        h, w = frame.shape[:2]
        self.frame_width = w
        tab_w = w // self.NUM_MODES

        for mode_idx in range(1, self.NUM_MODES + 1):
            x1 = (mode_idx - 1) * tab_w
            x2 = w if mode_idx == self.NUM_MODES else mode_idx * tab_w
            is_active = (self.tracker.mode == mode_idx)

            bg_color = (160, 90, 20) if is_active else (45, 45, 45)
            txt_color = (255, 255, 255) if is_active else (185, 185, 185)

            cv2.rectangle(frame, (x1, 0), (x2, self.UI_BAR_HEIGHT), bg_color, -1)
            cv2.rectangle(frame, (x1, 0), (x2, self.UI_BAR_HEIGHT), (90, 90, 90), 1)

            label = SegPersonTrackerCore.MODE_NAMES[mode_idx]
            cv2.putText(
                frame,
                label,
                (x1 + 5, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                txt_color,
                2 if is_active else 1,
                cv2.LINE_AA,
            )

        m = self.tracker.metrics
        panel_x, panel_y = 8, self.UI_BAR_HEIGHT + 8
        panel_w, panel_h = 285, 152

        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (20, 20, 20),
            -1,
        )
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (120, 100, 60),
            1,
        )

        vram_str = ''
        if 'CUDA' in m.device and torch.cuda.is_available():
            vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            vram_str = f' | VRAM:{vram_mb:.0f}MB'

        adapt_str = f' [{m.adaptive_state}]' if self.tracker.mode == 5 else ''

        lines = [
            (f'SEG Mode | Dev:{m.device} [Key:D, Mask:M]', (0, 255, 255)),
            (f'Cam FPS: {self.cam_fps:4.1f} | Max Infer: {m.max_infer_fps:5.1f} FPS', (0, 255, 0)),
            (f'Total  : {m.total_ms:5.1f} ms (ROS Lag: {self.ros_lag_ms:4.0f} ms)', (255, 255, 255)),
            (f' - Seg Infer  : {m.inference_ms:5.1f} ms (Pre:{m.preprocess_ms:3.1f})', (200, 220, 255)),
            (f' - Track/Mask : {m.post_track_ms + m.reid_ms:5.1f} ms (PartReID:{m.reid_ms:3.1f})', (200, 220, 255)),
            (f'Sys CPU: {self.cpu_percent:4.1f}% | RAM:{self.ram_mb:.0f}MB{vram_str}', (180, 255, 180)),
            (f'Persons: {num_persons} | Target:{self.tracker.target_id}{adapt_str}', (255, 220, 120)),
        ]

        for idx, (text, color) in enumerate(lines):
            y_pos = panel_y + 18 + idx * 19
            cv2.putText(
                frame,
                text,
                (panel_x + 8, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )

    def compressed_callback(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is not None:
            self.handle_frame(frame, msg.header.stamp)

    def raw_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.handle_frame(frame, msg.header.stamp)

    def handle_frame(self, frame: np.ndarray, msg_stamp):
        detections, target = self.tracker.process_frame(
            frame, auto_lock_first=self.auto_lock_first
        )
        self._update_system_metrics(msg_stamp)

        if target is not None:
            msg = Point()
            msg.x = float(target.norm_offset[0])
            msg.y = float(target.norm_offset[1])
            msg.z = float(target.area_ratio)
            self.target_pub.publish(msg)

        if self.show_gui:
            # 세그멘테이션 실루엣 마스크 반투명 렌더링
            if self.draw_mask and len(detections) > 0:
                mask_overlay = frame.copy()
                for p in detections:
                    if p.mask_poly is not None and len(p.mask_poly) >= 3:
                        pts = p.mask_poly.astype(np.int32)
                        fill_color = (0, 230, 80) if p.is_target else (140, 120, 90)
                        cv2.fillPoly(mask_overlay, [pts], fill_color)
                        cv2.polylines(frame, [pts], isClosed=True, color=fill_color, thickness=2 if p.is_target else 1)
                cv2.addWeighted(mask_overlay, 0.30, frame, 0.70, 0, frame)

            show_sim = self.tracker.mode in (2, 4, 5)
            for p in detections:
                x1, y1, x2, y2 = p.bbox
                cx, cy = p.center
                sim_str = f' sim:{p.similarity:.2f}' if show_sim else ''

                if p.is_target:
                    box_color = (0, 200, 255) if (self.tracker.mode == 5 and self.tracker.metrics.adaptive_state == 'GUARD') else (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    # 하단 접지점(발 위치) 표시
                    cv2.circle(frame, p.bottom_center, 4, (255, 0, 255), -1)
                    cv2.putText(
                        frame,
                        f'SEG TARGET ID:{p.track_id} dx:{p.norm_offset[0]:+.2f}{sim_str}',
                        (x1, max(self.UI_BAR_HEIGHT + 18, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        box_color,
                        2,
                    )
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (160, 160, 160), 1)
                    cv2.putText(
                        frame,
                        f'ID:{p.track_id}{sim_str}',
                        (x1, max(self.UI_BAR_HEIGHT + 15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (160, 160, 160),
                        1,
                    )

            self._draw_hud(frame, len(detections))
            cv2.imshow(self.win_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('1'), ord('2'), ord('3'), ord('4'), ord('5')):
                new_mode = int(chr(key))
                self.tracker.set_mode(new_mode)
                self.get_logger().info(
                    f'[키보드 모드 전환] {SegPersonTrackerCore.MODE_NAMES[new_mode]} 적용됨'
                )
            elif key in (ord('d'), ord('D')):
                new_dev = self.tracker.toggle_device()
                self.get_logger().info(f'[연산 장치 전환] Device -> {new_dev}')
            elif key in (ord('m'), ord('M')):
                self.draw_mask = not self.draw_mask
                self.get_logger().info(f'[마스크 오버레이 표시] -> {self.draw_mask}')


def main(args=None):
    rclpy.init(args=args)
    node = SegPersonTrackerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
