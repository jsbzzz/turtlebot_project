import math
import os
from typing import Optional, Tuple
import yaml
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    """Convert ZYX Euler angles (radians) into a (x, y, z, w) quaternion."""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qx, qy, qz, qw


def quaternion_to_euler(qx: float, qy: float, qz: float, qw: float) -> Tuple[float, float, float]:
    """Convert a (x, y, z, w) quaternion into ZYX Euler angles (roll, pitch, yaw) in radians."""
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def make_static_transform(
    stamp,
    parent_frame: str,
    child_frame: str,
    xyz: Tuple[float, float, float],
    rpy: Tuple[float, float, float],
) -> TransformStamped:
    """Build a geometry_msgs/TransformStamped from translation (m) and RPY (rad)."""
    t = TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = parent_frame
    t.child_frame_id = child_frame
    t.transform.translation.x = float(xyz[0])
    t.transform.translation.y = float(xyz[1])
    t.transform.translation.z = float(xyz[2])

    qx, qy, qz, qw = euler_to_quaternion(*rpy)
    t.transform.rotation.x = qx
    t.transform.rotation.y = qy
    t.transform.rotation.z = qz
    t.transform.rotation.w = qw
    return t


def format_transform(tf_msg: TransformStamped) -> str:
    tr = tf_msg.transform.translation
    rot = tf_msg.transform.rotation
    r, p, y = quaternion_to_euler(rot.x, rot.y, rot.z, rot.w)
    return (
        f'x={tr.x:+.3f}m, y={tr.y:+.3f}m, z={tr.z:+.3f}m | '
        f'rpy=({math.degrees(r):+.1f}°, {math.degrees(p):+.1f}°, {math.degrees(y):+.1f}°)'
    )


class TopCamTFNode(Node):
    """
    상단 추가 카메라(top_cam) 전용 TF 발행 및 좌표 모니터 노드:
    1) 시작 시점: 설정된 본인 상대좌표(parent_frame -> top_cam_link) 발행 및 출력
    2) 로봇 연결 시점: 로봇 몸체 기준 좌표(base_link / base_footprint -> top_cam_link) 확정 출력
    3) 이동/내비게이션 시점: 도킹 스테이션 출발 오도메트리(odom) 및 저장된 맵(map) 기준
       상단 카메라의 실시간 3D 좌표를 계산·표시 및 토픽(/robot4/top_cam/pose_in_map) 발행
    """

    def __init__(self):
        super().__init__('top_cam_tf_node')

        cfg = self._load_yaml_defaults()
        frames_cfg = cfg.get('frames', {})
        ext_cfg = cfg.get('extrinsics', {})
        init_cfg = cfg.get('initial_pose', {})

        self.declare_parameter('parent_frame', frames_cfg.get('parent_frame', 'oakd_rgb_camera_frame'))
        self.declare_parameter('camera_link_frame', frames_cfg.get('camera_link_frame', 'top_cam_link'))
        self.declare_parameter('optical_frame', frames_cfg.get('optical_frame', 'top_cam_optical_frame'))
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('footprint_frame', 'base_footprint')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('map_frame', 'map')

        self.declare_parameter('x', float(ext_cfg.get('x', 0.0)))
        self.declare_parameter('y', float(ext_cfg.get('y', 0.0)))
        self.declare_parameter('z', float(ext_cfg.get('z', 0.10)))
        self.declare_parameter('roll', float(ext_cfg.get('roll', 0.0)))
        self.declare_parameter('pitch', float(ext_cfg.get('pitch', 0.0)))
        self.declare_parameter('yaw', float(ext_cfg.get('yaw', 0.0)))
        self.declare_parameter('use_degrees', bool(ext_cfg.get('use_degrees', True)))
        self.declare_parameter('log_interval', 1.0)

        self.parent_frame = str(self.get_parameter('parent_frame').value)
        self.camera_link_frame = str(self.get_parameter('camera_link_frame').value)
        self.optical_frame = str(self.get_parameter('optical_frame').value)
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.footprint_frame = str(self.get_parameter('footprint_frame').value)
        self.odom_frame = str(self.get_parameter('odom_frame').value)
        self.map_frame = str(self.get_parameter('map_frame').value)

        self.xyz = (
            float(self.get_parameter('x').value),
            float(self.get_parameter('y').value),
            float(self.get_parameter('z').value),
        )
        roll = float(self.get_parameter('roll').value)
        pitch = float(self.get_parameter('pitch').value)
        yaw = float(self.get_parameter('yaw').value)
        if bool(self.get_parameter('use_degrees').value):
            self.rpy_rad = (math.radians(roll), math.radians(pitch), math.radians(yaw))
        else:
            self.rpy_rad = (roll, pitch, yaw)

        init_x = float(init_cfg.get('x', 0.0))
        init_y = float(init_cfg.get('y', 0.0))
        init_yaw = float(init_cfg.get('yaw', 0.0))
        if bool(init_cfg.get('use_degrees', True)):
            init_yaw = math.radians(init_yaw)
        self.init_map_xy_yaw = (init_x, init_y, init_yaw)

        ns = self.get_namespace()
        if ns == '/':
            cfg_ns = str(cfg.get('namespace', '/robot4')).rstrip('/')
            ns_prefix = cfg_ns
        else:
            ns_prefix = ns.rstrip('/')
        self.ns_prefix = ns_prefix
        self.tf_topic = f'{ns_prefix}/tf' if ns_prefix else '/tf'
        self.tf_static_topic = f'{ns_prefix}/tf_static' if ns_prefix else '/tf_static'
        self.map_topic = f'{ns_prefix}/map' if ns_prefix else '/map'
        self.scan_topic = f'{ns_prefix}/scan' if ns_prefix else '/scan'

        # 1. Static TF 발행 (parent_frame -> top_cam_link -> top_cam_optical_frame)
        static_qos = QoSProfile(
            depth=10,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.tf_static_pub = self.create_publisher(TFMessage, self.tf_static_topic, static_qos)
        self._publish_static_transforms()

        # 2. 네임스페이스(/robot4/tf, /robot4/tf_static) 직접 구독하여 TF Buffer 채우기
        self.tf_buffer = Buffer()
        self.tf_sub = self.create_subscription(
            TFMessage, self.tf_topic, self._on_tf_msg, qos_profile_sensor_data
        )
        self.tf_static_sub = self.create_subscription(
            TFMessage, self.tf_static_topic, self._on_tf_static_msg, static_qos
        )
        self.map_pose_pub = self.create_publisher(PoseStamped, f'{ns_prefix}/top_cam/pose_in_map', 10)
        self.base_pose_pub = self.create_publisher(PoseStamped, f'{ns_prefix}/top_cam/pose_in_base', 10)

        self._initial_base_logged = False
        self._initial_odom_pos: Optional[Tuple[float, float, float]] = None

        self.get_logger().info('=' * 78)
        self.get_logger().info(
            f'[1] 상단 캠 설정 좌표 ({self.parent_frame} -> {self.camera_link_frame}): '
            f'x={self.xyz[0]:+.3f}m, y={self.xyz[1]:+.3f}m, z={self.xyz[2]:+.3f}m | '
            f'rpy=({math.degrees(self.rpy_rad[0]):+.1f}°, '
            f'{math.degrees(self.rpy_rad[1]):+.1f}°, '
            f'{math.degrees(self.rpy_rad[2]):+.1f}°)'
        )
        self.get_logger().info(
            f'    광학 프레임 연결 ({self.camera_link_frame} -> {self.optical_frame}): '
            f'rpy=(-90.0°, +0.0°, -90.0°) | 발행 토픽: {self.tf_static_topic}'
        )
        self.get_logger().info('=' * 78)

        log_interval = float(self.get_parameter('log_interval').value)
        self.timer = self.create_timer(log_interval, self._on_timer)

    def _load_yaml_defaults(self) -> dict:
        candidates = [
            os.path.expanduser('~/turtlebot4_ws/src/person_tracker/config/phone_camera.yaml'),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'phone_camera.yaml'),
        ]
        for path in candidates:
            if os.path.isfile(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        return yaml.safe_load(f) or {}
                except Exception:
                    pass
        return {}

    def _publish_static_transforms(self):
        now = self.get_clock().now().to_msg()
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
        self.tf_static_pub.publish(TFMessage(transforms=[t_link, t_opt]))

    def _on_tf_msg(self, msg: TFMessage):
        for t in msg.transforms:
            self.tf_buffer.set_transform(t, 'default_authority')

    def _on_tf_static_msg(self, msg: TFMessage):
        for t in msg.transforms:
            self.tf_buffer.set_transform_static(t, 'default_authority')

    def _lookup(self, target_frame: str, source_frame: str) -> Optional[TransformStamped]:
        try:
            if self.tf_buffer.can_transform(target_frame, source_frame, Time()):
                return self.tf_buffer.lookup_transform(target_frame, source_frame, Time())
        except Exception:
            pass
        return None

    def _to_pose_stamped(self, tf_msg: TransformStamped) -> PoseStamped:
        pose = PoseStamped()
        pose.header = tf_msg.header
        pose.pose.position.x = tf_msg.transform.translation.x
        pose.pose.position.y = tf_msg.transform.translation.y
        pose.pose.position.z = tf_msg.transform.translation.z
        pose.pose.orientation = tf_msg.transform.rotation
        return pose

    def _on_timer(self):
        tf_base = self._lookup(self.base_frame, self.camera_link_frame)
        tf_foot = self._lookup(self.footprint_frame, self.camera_link_frame)

        # 최초 1회: 로봇 몸체(base_link) 및 바닥(base_footprint) 기준 상단 캠 확정 좌표 출력
        if tf_base is not None and not self._initial_base_logged:
            self._initial_base_logged = True
            self.get_logger().info('=' * 78)
            self.get_logger().info(
                f'✅ [2] 로봇 몸체 기준 상단 캠 좌표 확정 ({self.base_frame} -> {self.camera_link_frame}):\n'
                f'    • {self.base_frame:<14s} 기준: {format_transform(tf_base)}'
                + (
                    f'\n    • {self.footprint_frame:<14s} 기준: {format_transform(tf_foot)}'
                    if tf_foot is not None
                    else ''
                )
            )
            self.get_logger().info('=' * 78)

        if tf_base is not None:
            self.base_pose_pub.publish(self._to_pose_stamped(tf_base))

        # 주행 오도메트리(odom) 및 저장된 맵(map) 기준 좌표 조회
        tf_odom = self._lookup(self.odom_frame, self.camera_link_frame)
        tf_map = self._lookup(self.map_frame, self.camera_link_frame)

        odom_str = '대기 중'
        dist_from_start_str = ''
        if tf_odom is not None:
            tr = tf_odom.transform.translation
            if self._initial_odom_pos is None:
                self._initial_odom_pos = (tr.x, tr.y, tr.z)
            dx = tr.x - self._initial_odom_pos[0]
            dy = tr.y - self._initial_odom_pos[1]
            dist = math.hypot(dx, dy)
            odom_str = format_transform(tf_odom)
            dist_from_start_str = f' (시작/도킹 위치로부터 이동거리: {dist:.2f}m)'

        if tf_map is not None:
            self.map_pose_pub.publish(self._to_pose_stamped(tf_map))
            map_str = f'{format_transform(tf_map)} [AMCL 보정]'
        elif tf_odom is not None and self._initial_odom_pos is not None:
            # AMCL(loc 4)을 켰더라도 라이다(/robot4/scan)가 꺼져 있으면 AMCL이 map->odom TF를 발행하지 않음.
            # 이 경우 초기 위치(initial_pose) + 시작점 대비 odom 이동량으로 map 좌표를 폴백 계산하여 표시.
            tr = tf_odom.transform.translation
            rot = tf_odom.transform.rotation
            r, p, y = quaternion_to_euler(rot.x, rot.y, rot.z, rot.w)
            dx = tr.x - self._initial_odom_pos[0]
            dy = tr.y - self._initial_odom_pos[1]
            ix, iy, iyaw = self.init_map_xy_yaw
            mx = ix + dx * math.cos(iyaw) - dy * math.sin(iyaw)
            my = iy + dx * math.sin(iyaw) + dy * math.cos(iyaw)
            mz = tr.z
            myaw = iyaw + y

            has_map_server = self.count_publishers(self.map_topic) > 0
            has_scan_pub = self.count_publishers(self.scan_topic) > 0
            if has_map_server and not has_scan_pub:
                status_tag = 'Init+Odom 기준 (⚠️ 현재 로봇 라이다 /robot4/scan 꺼짐 -> AMCL 대기 중)'
            elif has_map_server:
                status_tag = 'Init+Odom 기준 (AMCL 초기 스캔 매칭 대기 중)'
            else:
                status_tag = 'Init+Odom 기준 (loc 4 미실행)'

            map_str = (
                f'x={mx:+.3f}m, y={my:+.3f}m, z={mz:+.3f}m | '
                f'rpy=({math.degrees(r):+.1f}°, {math.degrees(p):+.1f}°, {math.degrees(myaw):+.1f}°) '
                f'[{status_tag}]'
            )
        else:
            map_str = '미수신 (로봇 odom/map TF 대기 중)'

        base_short = format_transform(tf_base) if tf_base is not None else '로봇 TF 대기 중'
        self.get_logger().info(
            f'\n  [로봇몸체 {self.base_frame}] {base_short}'
            f'\n  [주행기준 {self.odom_frame:<9s}] {odom_str}{dist_from_start_str}'
            f'\n  [저장된맵 {self.map_frame:<9s}] {map_str}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = TopCamTFNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
