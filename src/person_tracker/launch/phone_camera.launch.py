import os
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _to_bool(val, fallback: bool) -> bool:
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in ('true', '1', 'yes', 'on'):
        return True
    if s in ('false', '0', 'no', 'off'):
        return False
    return fallback


def _launch_setup(context, *args, **kwargs):
    config_path = LaunchConfiguration('config_file').perform(context)
    cfg = {}
    if os.path.isfile(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}

    topics_cfg = cfg.get('topics', {})
    frames_cfg = cfg.get('frames', {})
    ext_cfg = cfg.get('extrinsics', {})

    def _pick(arg_name: str, fallback):
        val = LaunchConfiguration(arg_name).perform(context)
        return val if val != '' else str(fallback)

    ns = _pick('namespace', cfg.get('namespace', '/robot4'))
    video_source = _pick('video_source', cfg.get('video_source', '0'))

    raw_topic = _pick('raw_topic', topics_cfg.get('raw_topic', '/top_cam/image_raw'))
    compressed_topic = _pick(
        'compressed_topic',
        topics_cfg.get('compressed_topic', '/top_cam/image_raw/compressed'),
    )
    publish_raw = _to_bool(
        _pick('publish_raw', topics_cfg.get('publish_raw', False)),
        False,
    )
    publish_compressed = _to_bool(
        _pick('publish_compressed', topics_cfg.get('publish_compressed', True)),
        True,
    )

    parent_frame = _pick('parent_frame', frames_cfg.get('parent_frame', 'oakd_rgb_camera_frame'))
    camera_link_frame = _pick('camera_link_frame', frames_cfg.get('camera_link_frame', 'top_cam_link'))
    optical_frame = _pick('optical_frame', frames_cfg.get('optical_frame', 'top_cam_optical_frame'))

    x = float(_pick('x', ext_cfg.get('x', 0.0)))
    y = float(_pick('y', ext_cfg.get('y', 0.0)))
    z = float(_pick('z', ext_cfg.get('z', 0.10)))
    roll = float(_pick('roll', ext_cfg.get('roll', 0.0)))
    pitch = float(_pick('pitch', ext_cfg.get('pitch', 0.0)))
    yaw = float(_pick('yaw', ext_cfg.get('yaw', 0.0)))
    use_deg = _to_bool(_pick('use_degrees', ext_cfg.get('use_degrees', True)), True)
    log_interval = float(_pick('log_interval', cfg.get('log_interval', 2.0)))

    # /robot4/tf, /robot4/tf_static 구독을 위한 리매핑
    tf_remaps = [('/tf', 'tf'), ('/tf_static', 'tf_static')]
    use_sim_time = _to_bool(_pick('use_sim_time', False), False)

    top_cam_tf_node = Node(
        package='person_tracker',
        executable='top_cam_tf',
        name='top_cam_tf_node',
        namespace=ns,
        output='screen',
        remappings=tf_remaps,
        parameters=[{
            'parent_frame': parent_frame,
            'camera_link_frame': camera_link_frame,
            'optical_frame': optical_frame,
            'x': x,
            'y': y,
            'z': z,
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw,
            'use_degrees': use_deg,
            'log_interval': log_interval,
            'use_sim_time': use_sim_time,
        }],
    )

    publish_camera = _to_bool(_pick('publish_camera', True), True)
    camera_pub = Node(
        package='person_tracker',
        executable='camera_pub',
        name='top_camera_publisher_node',
        namespace=ns,
        output='screen',
        parameters=[{
            'video_source': str(video_source),
            'raw_topic': raw_topic,
            'compressed_topic': compressed_topic,
            'publish_raw': publish_raw,
            'publish_compressed': publish_compressed,
            'frame_id': optical_frame,
            'publish_tf': False,  # TF 발행 및 좌표 출력은 top_cam_tf_node가 전담
            'use_sim_time': use_sim_time,
        }],
    )

    return [top_cam_tf_node, camera_pub] if publish_camera else [top_cam_tf_node]


def generate_launch_description():
    src_cfg = os.path.expanduser('~/turtlebot4_ws/src/person_tracker/config/phone_camera.yaml')
    share_cfg = os.path.join(
        get_package_share_directory('person_tracker'),
        'config',
        'phone_camera.yaml',
    )
    default_cfg = src_cfg if os.path.isfile(src_cfg) else share_cfg
    args = [
        DeclareLaunchArgument('config_file', default_value=default_cfg, description='카메라 좌표 설정 YAML 경로'),
        DeclareLaunchArgument('namespace', default_value='', description='비워두면 YAML의 namespace 사용'),
        DeclareLaunchArgument('video_source', default_value='', description='비워두면 YAML의 video_source 사용'),
        DeclareLaunchArgument('raw_topic', default_value='', description='비워두면 YAML의 raw_topic 사용'),
        DeclareLaunchArgument('compressed_topic', default_value='', description='비워두면 YAML의 compressed_topic 사용'),
        DeclareLaunchArgument('publish_raw', default_value='', description='비워두면 YAML의 publish_raw 사용'),
        DeclareLaunchArgument('publish_compressed', default_value='', description='비워두면 YAML의 publish_compressed 사용'),
        DeclareLaunchArgument('parent_frame', default_value='', description='비워두면 YAML의 parent_frame 사용'),
        DeclareLaunchArgument('camera_link_frame', default_value='', description='비워두면 YAML의 camera_link_frame 사용'),
        DeclareLaunchArgument('optical_frame', default_value='', description='비워두면 YAML의 optical_frame 사용'),
        DeclareLaunchArgument('x', default_value='', description='앞(+)/뒤(-) [m]'),
        DeclareLaunchArgument('y', default_value='', description='왼쪽(+)/오른쪽(-) [m]'),
        DeclareLaunchArgument('z', default_value='', description='위(+)/아래(-) [m]'),
        DeclareLaunchArgument('roll', default_value='', description='X축 회전'),
        DeclareLaunchArgument('pitch', default_value='', description='Y축 회전'),
        DeclareLaunchArgument('yaw', default_value='', description='Z축 회전'),
        DeclareLaunchArgument('use_degrees', default_value='', description='true면 각도를 도(deg) 단위로 해석'),
        DeclareLaunchArgument('log_interval', default_value='', description='좌표 로그 출력 주기 [s]'),
        DeclareLaunchArgument('publish_camera', default_value='true', description='camera_pub 실행 여부 (false면 TF 노드만 실행)'),
        DeclareLaunchArgument('use_sim_time', default_value='false', description='시뮬레이션 시간 사용 여부'),
    ]
    return LaunchDescription(args + [OpaqueFunction(function=_launch_setup)])
