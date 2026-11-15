import math
import os
import time
import yaml
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer
from person_tracker.top_cam_tf_node import euler_to_quaternion, format_transform


class InitialPoseNode(Node):
    """
    RViz(2D Pose Estimate)를 켜지 않고도 도킹 스테이션(기본 0, 0, 0) 또는 지정 좌표로
    AMCL(/robot4/initialpose) 초기 위치를 설정하고, map -> base_link / top_cam_link TF 연결을 확인하는 노드
    """

    def __init__(self):
        super().__init__('init_pose_node')

        cfg = self._load_yaml_defaults()
        init_cfg = cfg.get('initial_pose', {})
        frames_cfg = cfg.get('frames', {})

        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('camera_link_frame', frames_cfg.get('camera_link_frame', 'top_cam_link'))
        self.declare_parameter('x', float(init_cfg.get('x', 0.0)))
        self.declare_parameter('y', float(init_cfg.get('y', 0.0)))
        self.declare_parameter('yaw', float(init_cfg.get('yaw', 0.0)))
        self.declare_parameter('use_degrees', bool(init_cfg.get('use_degrees', True)))
        self.declare_parameter('cov_xy', 0.05)
        self.declare_parameter('cov_yaw', 0.05)

        self.map_frame = str(self.get_parameter('map_frame').value)
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.camera_link_frame = str(self.get_parameter('camera_link_frame').value)
        self.x = float(self.get_parameter('x').value)
        self.y = float(self.get_parameter('y').value)
        yaw_in = float(self.get_parameter('yaw').value)
        self.yaw_rad = math.radians(yaw_in) if bool(self.get_parameter('use_degrees').value) else yaw_in
        self.cov_xy = float(self.get_parameter('cov_xy').value)
        self.cov_yaw = float(self.get_parameter('cov_yaw').value)

        ns = self.get_namespace()
        if ns == '/':
            cfg_ns = str(cfg.get('namespace', '/robot4')).rstrip('/')
            ns_prefix = cfg_ns
        else:
            ns_prefix = ns.rstrip('/')

        self.pose_topic = f'{ns_prefix}/initialpose' if ns_prefix else '/initialpose'
        self.pub = self.create_publisher(PoseWithCovarianceStamped, self.pose_topic, 10)

        tf_topic = f'{ns_prefix}/tf' if ns_prefix else '/tf'
        tf_static_topic = f'{ns_prefix}/tf_static' if ns_prefix else '/tf_static'
        static_qos = QoSProfile(
            depth=10,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.tf_buffer = Buffer()
        self.tf_sub = self.create_subscription(
            TFMessage,
            tf_topic,
            lambda m: [self.tf_buffer.set_transform(t, 'default_authority') for t in m.transforms],
            qos_profile_sensor_data,
        )
        self.tf_static_sub = self.create_subscription(
            TFMessage,
            tf_static_topic,
            lambda m: [self.tf_buffer.set_transform_static(t, 'default_authority') for t in m.transforms],
            static_qos,
        )

        self._sent_count = 0
        self._start_time = time.monotonic()
        self.get_logger().info(
            f'AMCL 초기 위치 설정 대기 중 -> 토픽: {self.pose_topic} | '
            f'목표 좌표: (x={self.x:+.3f}m, y={self.y:+.3f}m, yaw={math.degrees(self.yaw_rad):+.1f}°)'
        )
        self.timer = self.create_timer(0.5, self._on_timer)

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

    def _build_pose_msg(self) -> PoseWithCovarianceStamped:
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        msg.pose.pose.position.x = self.x
        msg.pose.pose.position.y = self.y
        msg.pose.pose.position.z = 0.0

        qx, qy, qz, qw = euler_to_quaternion(0.0, 0.0, self.yaw_rad)
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw

        cov = [0.0] * 36
        cov[0] = self.cov_xy
        cov[7] = self.cov_xy
        cov[35] = self.cov_yaw
        msg.pose.covariance = cov
        return msg

    def _on_timer(self):
        sub_count = self.pub.get_subscription_count()
        if sub_count == 0 and (time.monotonic() - self._start_time) < 10.0:
            self.get_logger().info(
                f'AMCL 구독자({self.pose_topic}) 대기 중... (먼저 loc 4 ~/maps/gallery.yaml 실행 필요)',
                throttle_duration_sec=2.0,
            )
            return

        if self._sent_count < 3:
            self.pub.publish(self._build_pose_msg())
            self._sent_count += 1
            self.get_logger().info(
                f'[{self._sent_count}/3] 초기 위치 발행 완료 -> {self.pose_topic}: '
                f'x={self.x:+.3f}m, y={self.y:+.3f}m, yaw={math.degrees(self.yaw_rad):+.1f}°'
            )
            return

        # TF에서 map -> base_link (및 top_cam_link) 변환이 생성되었는지 확인
        if self.tf_buffer.can_transform(self.map_frame, self.base_frame, Time()):
            tf_base = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, Time())
            self.get_logger().info('=' * 76)
            self.get_logger().info(
                f'✅ 초기 맵(map) 좌표 설정 완료!\n'
                f'   • 로봇 몸체 ({self.map_frame} -> {self.base_frame})    : {format_transform(tf_base)}'
            )
            if self.tf_buffer.can_transform(self.map_frame, self.camera_link_frame, Time()):
                tf_cam = self.tf_buffer.lookup_transform(self.map_frame, self.camera_link_frame, Time())
                self.get_logger().info(
                    f'   • 상단 카메라 ({self.map_frame} -> {self.camera_link_frame}): {format_transform(tf_cam)}'
                )
            self.get_logger().info('=' * 76)
            raise SystemExit(0)

        if (time.monotonic() - self._start_time) >= 8.0:
            self.get_logger().info(
                '초기 위치 토픽 발행을 마쳤습니다. (map -> base_link TF 확인은 top_cam_tf 로그를 참고하세요)'
            )
            raise SystemExit(0)


def main(args=None):
    rclpy.init(args=args)
    node = InitialPoseNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, SystemExit):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
