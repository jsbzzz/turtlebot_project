from dataclasses import dataclass
import math
import os
import time
from typing import Optional, Tuple

import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from geometry_msgs.msg import PointStamped, TransformStamped
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer
from visualization_msgs.msg import Marker

from person_tracker.multi_tracker_core import UnifiedTrackedPerson
from person_tracker.top_cam_tf_node import make_static_transform, quaternion_to_euler


@dataclass
class TargetLocalizationResult:
    track_id: int
    source: str                      # 'LiDAR' 또는 'CamEst'
    cam_bearing_deg: float           # 상단 캠 기준 수평 각도 (+왼쪽, -오른쪽)
    cam_dist_m: float                # 상단 캠 기준 수평 거리 (m)
    cam_xyz: Tuple[float, float, float]
    base_xyz: Tuple[float, float, float]
    base_dist_m: float               # 로봇(base_link) 중심 기준 수평 거리 (m)
    base_angle_deg: float            # 로봇(base_link) 정면 기준 각도 (+왼쪽, -오른쪽)
    map_xyz: Optional[Tuple[float, float, float]]
    map_source: str                  # 'AMCL' | 'Init+Odom' | 'None'
    matched_rays: int                # 매칭된 라이다 포인트 개수


@dataclass
class CameraPoseStatus:
    base_xyz: Optional[Tuple[float, float, float]]
    odom_xyz: Optional[Tuple[float, float, float]]
    map_xyz: Optional[Tuple[float, float, float]]
    map_yaw_deg: Optional[float]
    map_source: str                  # 'AMCL' | 'Init+Odom' | 'None'
    lidar_hz: float
    lidar_frame: str


def transform_to_matrix(tf_msg: TransformStamped) -> np.ndarray:
    """Convert geometry_msgs/TransformStamped into a 4x4 homogeneous matrix."""
    tr = tf_msg.transform.translation
    rot = tf_msg.transform.rotation
    qx, qy, qz, qw = rot.x, rot.y, rot.z, rot.w

    # Quaternion to 3x3 rotation matrix
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    mat = np.eye(4, dtype=np.float64)
    mat[0, 0] = 1.0 - 2.0 * (yy + zz)
    mat[0, 1] = 2.0 * (xy - wz)
    mat[0, 2] = 2.0 * (xz + wy)
    mat[1, 0] = 2.0 * (xy + wz)
    mat[1, 1] = 1.0 - 2.0 * (xx + zz)
    mat[1, 2] = 2.0 * (yz - wx)
    mat[2, 0] = 2.0 * (xz - wy)
    mat[2, 1] = 2.0 * (yz + wx)
    mat[2, 2] = 1.0 - 2.0 * (xx + yy)

    mat[0, 3] = tr.x
    mat[1, 3] = tr.y
    mat[2, 3] = tr.z
    return mat


def apply_transform_point(mat: np.ndarray, pt: Tuple[float, float, float]) -> Tuple[float, float, float]:
    vec = np.array([pt[0], pt[1], pt[2], 1.0], dtype=np.float64)
    out = mat @ vec
    return float(out[0]), float(out[1]), float(out[2])


class LidarPersonLocalizer:
    """
    상단 웹캠(top_cam_link)의 YOLO 바운딩 박스 각도와
    2D 라이다(/robot4/scan, rplidar_link) 점군을 TF 행렬로 정합하여
    특정 타겟 인물의:
      1) 상단 카메라 기준 3D 좌표 (top_cam_link)
      2) 로봇 몸체 기준 3D 좌표 및 거리/방위각 (base_link)
      3) 저장된 맵 기준 절대 3D 좌표 (map)
    를 실시간으로 계산하고 ROS 2 토픽/마커로 발행하는 모듈.
    """

    def __init__(self, node: Node):
        self.node = node
        cfg = self._load_yaml_defaults()

        ns = node.get_namespace()
        if ns == '/':
            cfg_ns = str(cfg.get('namespace', '/robot4')).rstrip('/')
            self.ns_prefix = cfg_ns
        else:
            self.ns_prefix = ns.rstrip('/')

        frames_cfg = cfg.get('frames', {})
        ext_cfg = cfg.get('extrinsics', {})
        intr_cfg = cfg.get('camera_intrinsics', {})
        init_cfg = cfg.get('initial_pose', {})

        self.parent_frame = str(frames_cfg.get('parent_frame', 'oakd_rgb_camera_frame'))
        self.camera_link_frame = str(frames_cfg.get('camera_link_frame', 'top_cam_link'))
        self.optical_frame = str(frames_cfg.get('optical_frame', 'top_cam_optical_frame'))
        self.base_frame = 'base_link'
        self.odom_frame = 'odom'
        self.map_frame = 'map'

        init_x = float(init_cfg.get('x', 0.0))
        init_y = float(init_cfg.get('y', 0.0))
        init_yaw = float(init_cfg.get('yaw', 0.0))
        if bool(init_cfg.get('use_degrees', True)):
            init_yaw = math.radians(init_yaw)
        self.init_map_xy_yaw = (init_x, init_y, init_yaw)
        self._initial_odom_inv: Optional[np.ndarray] = None

        self.hfov_deg = float(intr_cfg.get('hfov_deg', 65.0))
        self.xyz = (
            float(ext_cfg.get('x', 0.0)),
            float(ext_cfg.get('y', 0.0)),
            float(ext_cfg.get('z', 0.17)),
        )
        roll = float(ext_cfg.get('roll', 0.0))
        pitch = float(ext_cfg.get('pitch', 0.0))
        yaw = float(ext_cfg.get('yaw', 0.0))
        if bool(ext_cfg.get('use_degrees', True)):
            self.rpy_rad = (math.radians(roll), math.radians(pitch), math.radians(yaw))
        else:
            self.rpy_rad = (roll, pitch, yaw)

        self.tf_topic = f'{self.ns_prefix}/tf' if self.ns_prefix else '/tf'
        self.tf_static_topic = f'{self.ns_prefix}/tf_static' if self.ns_prefix else '/tf_static'
        self.scan_topic = f'{self.ns_prefix}/scan' if self.ns_prefix else '/scan'
        self.start_motor_srv = f'{self.ns_prefix}/start_motor' if self.ns_prefix else '/start_motor'

        # 1. TF Buffer 및 /robot4/tf, /robot4/tf_static 직접 구독
        self.tf_buffer = Buffer()
        static_qos = QoSProfile(
            depth=20,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.tf_static_pub = node.create_publisher(TFMessage, self.tf_static_topic, static_qos)
        self._publish_static_cam_tf()

        self.tf_sub = node.create_subscription(
            TFMessage, self.tf_topic, self._on_tf_msg, qos_profile_sensor_data
        )
        self.tf_static_sub = node.create_subscription(
            TFMessage, self.tf_static_topic, self._on_tf_static_msg, static_qos
        )

        # 2. 2D LiDAR (/robot4/scan) 구독 및 도킹 시 모터 자동 기동 클라이언트
        self.latest_scan: Optional[LaserScan] = None
        self.last_scan_WallTime: float = 0.0
        self.scan_fps: float = 0.0
        self.scan_sub = node.create_subscription(
            LaserScan, self.scan_topic, self._on_scan, qos_profile_sensor_data
        )
        self.motor_client = node.create_client(Empty, self.start_motor_srv)
        self._motor_requested = False
        self._startup_time = time.time()
        self._watchdog_timer = node.create_timer(2.5, self._check_lidar_motor)

        # 3. 타겟 인물 위치 발행 토픽 (로봇 기준 / 맵 기준 / RViz 마커)
        self.pub_point_base = node.create_publisher(
            PointStamped, f'{self.ns_prefix}/target_person/point_base', 10
        )
        self.pub_point_map = node.create_publisher(
            PointStamped, f'{self.ns_prefix}/target_person/point_map', 10
        )
        self.pub_marker = node.create_publisher(
            Marker, f'{self.ns_prefix}/target_person/marker', 10
        )

    def _load_yaml_defaults(self) -> dict:
        candidates = [
            os.path.expanduser('~/turtlebot4_ws/src/person_tracker/config/phone_camera.yaml'),
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'config',
                'phone_camera.yaml',
            ),
        ]
        for path in candidates:
            if os.path.isfile(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        return yaml.safe_load(f) or {}
                except Exception:
                    pass
        return {}

    def _publish_static_cam_tf(self):
        now = self.node.get_clock().now().to_msg()
        t_link = make_static_transform(
            now, self.parent_frame, self.camera_link_frame, self.xyz, self.rpy_rad
        )
        t_opt = make_static_transform(
            now,
            self.camera_link_frame,
            self.optical_frame,
            (0.0, 0.0, 0.0),
            (-math.pi / 2.0, 0.0, -math.pi / 2.0),
        )
        msg = TFMessage(transforms=[t_link, t_opt])
        self.tf_static_pub.publish(msg)
        for t in msg.transforms:
            self.tf_buffer.set_transform_static(t, 'local_static')

    def _on_tf_msg(self, msg: TFMessage):
        for t in msg.transforms:
            self.tf_buffer.set_transform(t, 'default_authority')

    def _on_tf_static_msg(self, msg: TFMessage):
        for t in msg.transforms:
            self.tf_buffer.set_transform_static(t, 'default_authority')

    def _on_scan(self, msg: LaserScan):
        now = time.time()
        if self.last_scan_WallTime > 0.0:
            dt = now - self.last_scan_WallTime
            if dt > 0:
                inst = 1.0 / dt
                self.scan_fps = 0.85 * self.scan_fps + 0.15 * inst if self.scan_fps > 0 else inst
        self.last_scan_WallTime = now
        self.latest_scan = msg

    def _check_lidar_motor(self):
        """도킹 스테이션 위에서 라이다 모터가 멈춰 있을 때 자동으로 1회 기동."""
        if self._motor_requested:
            return
        if time.time() - self.last_scan_WallTime > 2.2:
            self._motor_requested = True
            if self.motor_client.service_is_ready():
                self.node.get_logger().info(
                    f'[LiDAR] {self.scan_topic} 수신 대기 중 -> {self.start_motor_srv} 모터 기동 요청'
                )
                self.motor_client.call_async(Empty.Request())

    def _lookup(self, target_frame: str, source_frame: str) -> Optional[TransformStamped]:
        try:
            if self.tf_buffer.can_transform(target_frame, source_frame, Time()):
                return self.tf_buffer.lookup_transform(target_frame, source_frame, Time())
        except Exception:
            pass
        return None

    def _get_map_from_base_matrix(self) -> Tuple[Optional[np.ndarray], str]:
        """
        1순위: AMCL이 발행하는 map -> base_link TF 사용 ('AMCL')
        2순위: 도킹 등으로 라이다가 0Hz라 AMCL TF가 아직 없을 때,
               initial_pose + (시작 시점 대비 odom 변화량)으로 map 좌표 계산 ('Init+Odom')
        """
        tf_map_base = self._lookup(self.map_frame, self.base_frame)
        if tf_map_base is not None:
            return transform_to_matrix(tf_map_base), 'AMCL'

        tf_odom_base = self._lookup(self.odom_frame, self.base_frame)
        if tf_odom_base is not None:
            mat_odom_base = transform_to_matrix(tf_odom_base)
            if self._initial_odom_inv is None:
                self._initial_odom_inv = np.linalg.inv(mat_odom_base)
            # 시작 시점 대비 상대 이동 행렬 (delta_odom)
            mat_delta = mat_odom_base @ self._initial_odom_inv

            ix, iy, iyaw = self.init_map_xy_yaw
            c, s = math.cos(iyaw), math.sin(iyaw)
            mat_init = np.array(
                [
                    [c, -s, 0.0, ix],
                    [s,  c, 0.0, iy],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            return mat_init @ mat_delta, 'Init+Odom'

        return None, 'None'

    def get_camera_pose_status(self) -> CameraPoseStatus:
        tf_base = self._lookup(self.base_frame, self.camera_link_frame)
        tf_odom = self._lookup(self.odom_frame, self.camera_link_frame)
        tf_map = self._lookup(self.map_frame, self.camera_link_frame)

        base_xyz = None
        if tf_base is not None:
            tr = tf_base.transform.translation
            base_xyz = (tr.x, tr.y, tr.z)

        odom_xyz = None
        if tf_odom is not None:
            tr = tf_odom.transform.translation
            odom_xyz = (tr.x, tr.y, tr.z)

        map_xyz = None
        map_yaw_deg = None
        map_source = 'None'
        if tf_map is not None:
            tr = tf_map.transform.translation
            rot = tf_map.transform.rotation
            _, _, yaw = quaternion_to_euler(rot.x, rot.y, rot.z, rot.w)
            map_xyz = (tr.x, tr.y, tr.z)
            map_yaw_deg = math.degrees(yaw)
            map_source = 'AMCL'
        else:
            mat_map_base, src = self._get_map_from_base_matrix()
            if mat_map_base is not None and base_xyz is not None:
                map_xyz = apply_transform_point(mat_map_base, base_xyz)
                map_yaw_deg = math.degrees(math.atan2(mat_map_base[1, 0], mat_map_base[0, 0]))
                map_source = src

        scan_fresh = (time.time() - self.last_scan_WallTime) < 2.5
        lidar_hz = self.scan_fps if scan_fresh else 0.0
        lidar_frame = self.latest_scan.header.frame_id if self.latest_scan is not None else 'rplidar_link'

        return CameraPoseStatus(
            base_xyz=base_xyz,
            odom_xyz=odom_xyz,
            map_xyz=map_xyz,
            map_yaw_deg=map_yaw_deg,
            map_source=map_source,
            lidar_hz=lidar_hz,
            lidar_frame=lidar_frame,
        )

    def localize_target(
        self,
        target: UnifiedTrackedPerson,
        frame_width: int,
        frame_height: int,
    ) -> TargetLocalizationResult:
        x1, y1, x2, y2 = target.bbox
        cx, cy = target.center

        half_w = max(1.0, frame_width * 0.5)
        fx = half_w / math.tan(math.radians(self.hfov_deg * 0.5))
        fy = fx

        # top_cam_link 기준 (+X 정면, +Y 왼쪽, +Z 위):
        # 이미지 오른쪽(cx > W/2)은 -Y 방향, 이미지 왼쪽(cx < W/2)은 +Y 방향
        alpha_c = math.atan2(-(cx - half_w), fx)
        alpha_left = math.atan2(-(x1 - half_w), fx)
        alpha_right = math.atan2(-(x2 - half_w), fx)
        bbox_span = abs(alpha_left - alpha_right)
        half_sector = max(math.radians(3.5), min(math.radians(18.0), bbox_span * 0.35))

        cam_dist_m: Optional[float] = None
        matched_rays = 0
        source = 'CamEst'

        # 1. 최신 라이다 스캔을 top_cam_link 좌표계로 변환하여 바운딩 박스 방향 거리 추출
        scan = self.latest_scan
        if scan is not None and (time.time() - self.last_scan_WallTime) < 2.5:
            lidar_frame = scan.header.frame_id or 'rplidar_link'
            tf_cam_from_lidar = self._lookup(self.camera_link_frame, lidar_frame)
            if tf_cam_from_lidar is not None:
                mat_cam_from_lidar = transform_to_matrix(tf_cam_from_lidar)
            else:
                # TF 미수신 시 로봇 기본 구조 기반 폴백 (rplidar_link는 base_link 대비 +90deg 회전)
                mat_cam_from_lidar = np.array(
                    [
                        [0.0, -1.0, 0.0, 0.02],
                        [1.0, 0.0, 0.0, 0.00],
                        [0.0, 0.0, 1.0, -0.221],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                )

            ranges = np.asarray(scan.ranges, dtype=np.float64)
            n_pts = len(ranges)
            if n_pts > 0:
                angles = scan.angle_min + np.arange(n_pts, dtype=np.float64) * scan.angle_increment
                r_min = max(0.12, float(scan.range_min))
                r_max = min(12.0, float(scan.range_max) if scan.range_max > 0 else 12.0)
                valid = np.isfinite(ranges) & (ranges >= r_min) & (ranges <= r_max)

                if np.any(valid):
                    rv = ranges[valid]
                    av = angles[valid]
                    pts_lidar = np.vstack(
                        (rv * np.cos(av), rv * np.sin(av), np.zeros_like(rv), np.ones_like(rv))
                    )
                    pts_cam = mat_cam_from_lidar @ pts_lidar
                    xc = pts_cam[0, :]
                    yc = pts_cam[1, :]

                    front_mask = xc > 0.12
                    if np.any(front_mask):
                        xc = xc[front_mask]
                        yc = yc[front_mask]
                        betas = np.arctan2(yc, xc)
                        diffs = np.abs(np.arctan2(np.sin(betas - alpha_c), np.cos(betas - alpha_c)))
                        in_sector = diffs <= half_sector

                        if np.any(in_sector):
                            dists = np.hypot(xc[in_sector], yc[in_sector])
                            matched_rays = int(dists.size)
                            # 배경 벽면을 배제하고 앞쪽 사람 다리/몸통 클러스터(25백분위수 부근)를 선택
                            p25 = float(np.percentile(dists, 25))
                            cluster = dists[(dists >= p25 - 0.15) & (dists <= p25 + 0.25)]
                            cam_dist_m = float(np.mean(cluster)) if cluster.size > 0 else p25
                            source = 'LiDAR'

        # 2. 라이다 포인트가 없을 때 바운딩 박스 크기 기반 폴백 추정
        if cam_dist_m is None:
            bbox_h = max(24.0, float(y2 - y1))
            cam_dist_m = float(np.clip((1.65 * fy) / bbox_h, 0.35, 8.0))
            source = 'CamEst'

        # 상단 카메라(top_cam_link) 기준 타겟 3D 좌표
        cam_x = cam_dist_m * math.cos(alpha_c)
        cam_y = cam_dist_m * math.sin(alpha_c)
        cam_z = 0.0
        cam_xyz = (cam_x, cam_y, cam_z)

        # 3. 로봇 몸체(base_link) 기준 타겟 3D 좌표 변환
        tf_base_from_cam = self._lookup(self.base_frame, self.camera_link_frame)
        if tf_base_from_cam is not None:
            mat_base_from_cam = transform_to_matrix(tf_base_from_cam)
            base_xyz = apply_transform_point(mat_base_from_cam, cam_xyz)
        else:
            # base_link -> top_cam_link 기본 오프셋 폴백 (-0.06m, 0.0m, +0.414m)
            base_xyz = (cam_x - 0.060, cam_y, cam_z)

        base_dist_m = math.hypot(base_xyz[0], base_xyz[1])
        base_angle_deg = math.degrees(math.atan2(base_xyz[1], base_xyz[0]))

        # 4. 저장된 맵(map) 기준 타겟 3D 좌표 변환 (AMCL 또는 도킹 시 Init+Odom 폴백)
        map_xyz: Optional[Tuple[float, float, float]] = None
        mat_map_from_base, map_source = self._get_map_from_base_matrix()
        if mat_map_from_base is not None:
            map_xyz = apply_transform_point(mat_map_from_base, base_xyz)

        res = TargetLocalizationResult(
            track_id=target.track_id,
            source=source,
            cam_bearing_deg=math.degrees(alpha_c),
            cam_dist_m=cam_dist_m,
            cam_xyz=cam_xyz,
            base_xyz=base_xyz,
            base_dist_m=base_dist_m,
            base_angle_deg=base_angle_deg,
            map_xyz=map_xyz,
            map_source=map_source,
            matched_rays=matched_rays,
        )
        self._publish_target_outputs(res)
        return res

    def _publish_target_outputs(self, res: TargetLocalizationResult):
        now_msg = self.node.get_clock().now().to_msg()

        pt_base = PointStamped()
        pt_base.header.stamp = now_msg
        pt_base.header.frame_id = self.base_frame
        pt_base.point.x = res.base_xyz[0]
        pt_base.point.y = res.base_xyz[1]
        pt_base.point.z = 0.0
        self.pub_point_base.publish(pt_base)

        if res.map_xyz is not None:
            pt_map = PointStamped()
            pt_map.header.stamp = now_msg
            pt_map.header.frame_id = self.map_frame
            pt_map.point.x = res.map_xyz[0]
            pt_map.point.y = res.map_xyz[1]
            pt_map.point.z = 0.0
            self.pub_point_map.publish(pt_map)

        marker = Marker()
        marker.header.stamp = now_msg
        if res.map_xyz is not None:
            marker.header.frame_id = self.map_frame
            mx, my = res.map_xyz[0], res.map_xyz[1]
        else:
            marker.header.frame_id = self.base_frame
            mx, my = res.base_xyz[0], res.base_xyz[1]

        marker.ns = 'target_person'
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x = mx
        marker.pose.position.y = my
        marker.pose.position.z = 0.5
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.35
        marker.scale.y = 0.35
        marker.scale.z = 1.0
        marker.color.a = 0.85
        if res.source == 'LiDAR':
            marker.color.r = 0.1
            marker.color.g = 1.0
            marker.color.b = 0.3
        else:
            marker.color.r = 1.0
            marker.color.g = 0.75
            marker.color.b = 0.1
        self.pub_marker.publish(marker)
