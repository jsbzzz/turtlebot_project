import math
from typing import Union
import cv2
import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from tf2_msgs.msg import TFMessage
from cv_bridge import CvBridge
from person_tracker.top_cam_tf_node import make_static_transform


class CameraPublisherNode(Node):
    """
    웹캠 또는 DroidCam(가상 비디오 장치 / HTTP 스트림)으로부터 영상을 캡처하여
    ROS 2 토픽과 선택적 Static TF로 발행하는 노드
    """

    def __init__(self):
        super().__init__('camera_publisher_node')

        # 파라미터 선언
        # video_source: 웹캠 번호('0', '2' 등) 또는 DroidCam URL('http://192.168.x.x:4747/video')
        # 기본 터틀봇 카메라(/robot4/oakd/rgb/...)와 충돌하지 않도록 기본 토픽/프레임을 top_cam으로 분리
        self.declare_parameter('video_source', '0', ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('raw_topic', '/top_cam/image_raw')
        self.declare_parameter('compressed_topic', '/top_cam/image_raw/compressed')
        self.declare_parameter('publish_raw', True)
        self.declare_parameter('publish_compressed', True)
        self.declare_parameter('fps', 15.0)          # 저사양 및 무선 네트워크를 고려한 기본 15 FPS
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('jpeg_quality', 80)   # 압축률 (0~100, 낮을수록 전송량 감소)

        # 카메라 프레임 및 상대 위치 TF 파라미터 (ROS 링크 좌표계: +X 전방, +Y 좌측, +Z 상단, 단위 m/rad)
        self.declare_parameter('frame_id', 'top_cam_optical_frame')
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('tf_topic', '/tf_static')
        self.declare_parameter('parent_frame', 'oakd_rgb_camera_frame')
        self.declare_parameter('camera_link_frame', 'top_cam_link')
        self.declare_parameter('publish_optical_tf', True)
        self.declare_parameter('tf_x', 0.0)
        self.declare_parameter('tf_y', 0.0)
        self.declare_parameter('tf_z', 0.0)
        self.declare_parameter('tf_roll', 0.0)
        self.declare_parameter('tf_pitch', 0.0)
        self.declare_parameter('tf_yaw', 0.0)
        self.declare_parameter('tf_degrees', False)

        raw_source = str(self.get_parameter('video_source').value)
        ns = self.get_namespace()
        ns_prefix = '' if ns == '/' else ns
        raw_topic = ns_prefix + str(self.get_parameter('raw_topic').value)
        compressed_topic = ns_prefix + str(self.get_parameter('compressed_topic').value)
        tf_topic_param = str(self.get_parameter('tf_topic').value)
        tf_topic = tf_topic_param if (ns_prefix and tf_topic_param.startswith(ns_prefix + '/')) else (ns_prefix + tf_topic_param)
        self.publish_raw = bool(self.get_parameter('publish_raw').value)
        self.publish_compressed = bool(self.get_parameter('publish_compressed').value)
        fps = float(self.get_parameter('fps').value)
        width = int(self.get_parameter('width').value)
        height = int(self.get_parameter('height').value)
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)
        self.frame_id = str(self.get_parameter('frame_id').value)
        self.get_logger().info(f"{compressed_topic}")
        self._tf_pub = None
        if bool(self.get_parameter('publish_tf').value):
            self._publish_camera_static_tf(tf_topic)

        # 숫자 형태('0', '1' 등)면 정수 인덱스로 변환, 아니면 경로/URL 문자열 그대로 사용
        video_source: Union[int, str] = int(raw_source) if raw_source.isdigit() else raw_source

        # 로컬 비디오 장치(/dev/video*)는 GStreamer 충돌 방지를 위해 CAP_V4L2 백엔드 사용
        if isinstance(video_source, int) or str(video_source).startswith('/dev/video'):
            self.cap = cv2.VideoCapture(video_source, cv2.CAP_V4L2)
        else:
            self.cap = cv2.VideoCapture(video_source)

        if not self.cap.isOpened():
            self.get_logger().error(
                f'카메라를 열 수 없습니다: {video_source} '
                '(DroidCam 연결 상태나 장치 번호/URL을 확인하세요)'
            )

        self.bridge = CvBridge()

        # 원본(/oakd/rgb/image_raw) 및 압축본(/oakd/rgb/image_raw/compressed) 퍼블리셔 생성
        self.raw_pub = (
            self.create_publisher(Image, raw_topic, qos_profile_sensor_data)
            if self.publish_raw
            else None
        )
        self.comp_pub = (
            self.create_publisher(CompressedImage, compressed_topic, qos_profile_sensor_data)
            if self.publish_compressed
            else None
        )

        self.timer = self.create_timer(1.0 / fps, self.timer_callback)
        self.get_logger().info(
            f'CameraPublisherNode 시작됨 | 소스: {video_source} -> '
            f'raw: {raw_topic if self.publish_raw else "OFF"}, '
            f'compressed: {compressed_topic if self.publish_compressed else "OFF"} ({fps} FPS)'
        )

    def timer_callback(self):
        if not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.get_logger().warn('프레임을 읽어오지 못했습니다.', throttle_duration_sec=5.0)
            return

        now = self.get_clock().now().to_msg()

        if self.comp_pub is not None:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            success, encoded_img = cv2.imencode('.jpg', frame, encode_param)
            if success:
                msg = CompressedImage()
                msg.header.stamp = now
                msg.header.frame_id = self.frame_id
                msg.format = 'jpeg'
                msg.data = encoded_img.tobytes()
                self.comp_pub.publish(msg)

        if self.raw_pub is not None:
            raw_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            raw_msg.header.stamp = now
            raw_msg.header.frame_id = self.frame_id
            self.raw_pub.publish(raw_msg)

    def _publish_camera_static_tf(self, tf_topic: str):
        parent_frame = str(self.get_parameter('parent_frame').value)
        camera_link_frame = str(self.get_parameter('camera_link_frame').value)
        publish_optical_tf = bool(self.get_parameter('publish_optical_tf').value)

        xyz = (
            float(self.get_parameter('tf_x').value),
            float(self.get_parameter('tf_y').value),
            float(self.get_parameter('tf_z').value),
        )
        roll = float(self.get_parameter('tf_roll').value)
        pitch = float(self.get_parameter('tf_pitch').value)
        yaw = float(self.get_parameter('tf_yaw').value)
        if bool(self.get_parameter('tf_degrees').value):
            roll, pitch, yaw = math.radians(roll), math.radians(pitch), math.radians(yaw)

        now = self.get_clock().now().to_msg()
        transforms = [
            make_static_transform(now, parent_frame, camera_link_frame, xyz, (roll, pitch, yaw))
        ]
        # ROS 링크 좌표(+X 전방) -> 카메라 광학 좌표(+Z 전방, +X 우측, +Y 하단) 회전
        if publish_optical_tf and self.frame_id != camera_link_frame:
            transforms.append(
                make_static_transform(
                    now,
                    camera_link_frame,
                    self.frame_id,
                    (0.0, 0.0, 0.0),
                    (-math.pi / 2.0, 0.0, -math.pi / 2.0),
                )
            )

        static_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self._tf_pub = self.create_publisher(TFMessage, tf_topic, static_qos)
        self._tf_pub.publish(TFMessage(transforms=transforms))
        self.get_logger().info(
            f'Static TF 발행 ({tf_topic}): {parent_frame} -> {camera_link_frame}'
            f' (x={xyz[0]:.3f}, y={xyz[1]:.3f}, z={xyz[2]:.3f}, '
            f'rpy=({roll:.3f}, {pitch:.3f}, {yaw:.3f}) rad)'
            + (f' -> {self.frame_id}' if publish_optical_tf and self.frame_id != camera_link_frame else '')
        )

    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisherNode()
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
