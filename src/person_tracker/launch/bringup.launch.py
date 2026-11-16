import os
import yaml
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _to_bool(val, fallback: bool = False) -> bool:
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in ('true', '1', 'yes', 'on'):
        return True
    if s in ('false', '0', 'no', 'off'):
        return False
    return fallback


def _launch_setup(context, *args, **kwargs):
    pkg_share = get_package_share_directory('person_tracker')

    config_path = LaunchConfiguration('config_file').perform(context).strip()
    cfg = {}
    if os.path.isfile(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}

    topics_cfg = cfg.get('topics', {})
    init_cfg = cfg.get('initial_pose', {})
    ext_cfg = cfg.get('extrinsics', {})

    def _pick(arg_name: str, fallback):
        val = LaunchConfiguration(arg_name).perform(context).strip()
        return val if val != '' else str(fallback)

    ns_arg = LaunchConfiguration('namespace').perform(context).strip()
    ns = ns_arg if ns_arg else str(cfg.get('namespace', '/robot4')).strip()
    if not ns.startswith('/'):
        ns = '/' + ns
    ns = ns.rstrip('/') or '/robot4'

    video_arg = LaunchConfiguration('video_source').perform(context).strip()
    video_source = video_arg if video_arg else str(cfg.get('video_source', '0')).strip()

    map_path = LaunchConfiguration('map').perform(context).strip()
    use_localization = _to_bool(LaunchConfiguration('use_localization').perform(context), True)
    run_init_pose = _to_bool(LaunchConfiguration('run_init_pose').perform(context), True)
    run_camera = _to_bool(LaunchConfiguration('run_camera').perform(context), True)
    use_oakd = _to_bool(LaunchConfiguration('use_oakd').perform(context), False)
    run_tracker = _to_bool(LaunchConfiguration('run_tracker').perform(context), True)
    run_rviz = _to_bool(LaunchConfiguration('run_rviz').perform(context), False)
    use_sim_time_str = LaunchConfiguration('use_sim_time').perform(context).strip()
    use_sim_time = _to_bool(use_sim_time_str, False)

    cam_x = float(_pick('x', ext_cfg.get('x', 0.0)))
    cam_y = float(_pick('y', ext_cfg.get('y', 0.0)))
    cam_z = float(_pick('z', ext_cfg.get('z', 0.17)))
    cam_roll = float(_pick('roll', ext_cfg.get('roll', 0.0)))
    cam_pitch = float(_pick('pitch', ext_cfg.get('pitch', 0.0)))
    cam_yaw = float(_pick('yaw', ext_cfg.get('yaw', 0.0)))
    cam_use_deg = _to_bool(_pick('use_degrees', ext_cfg.get('use_degrees', True)), True)

    publish_compressed = _to_bool(topics_cfg.get('publish_compressed', True), True)
    raw_topic = str(topics_cfg.get('raw_topic', '/top_cam/image_raw'))
    compressed_topic = str(topics_cfg.get('compressed_topic', '/top_cam/image_raw/compressed'))

    if use_oakd:
        tracker_image_topic = f'{ns}/oakd/rgb/preview/image_raw/compressed'
    else:
        sub_topic = compressed_topic if publish_compressed else raw_topic
        if not sub_topic.startswith('/'):
            sub_topic = '/' + sub_topic
        tracker_image_topic = f'{ns}{sub_topic}'

    actions = []

    # 1. 저장된 맵 기반 로컬라이제이션 (AMCL + Map Server)
    if use_localization:
        try:
            tb4_nav_share = get_package_share_directory('turtlebot4_navigation')
            loc_launch = IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(tb4_nav_share, 'launch', 'localization.launch.py')
                ),
                launch_arguments={
                    'namespace': ns,
                    'map': map_path,
                    'use_sim_time': 'true' if use_sim_time else 'false',
                }.items(),
            )
            actions.append(loc_launch)
        except PackageNotFoundError:
            print('[WARN] turtlebot4_navigation 패키지를 찾지 못해 localization을 건너뜁니다.')

        # 2. 초기 위치 발행 (phone_camera.yaml의 initial_pose 동일 적용)
        if run_init_pose:
            init_pose_node = Node(
                package='person_tracker',
                executable='init_pose',
                name='init_pose_node',
                namespace=ns,
                output='screen',
                parameters=[{
                    'x': float(init_cfg.get('x', 0.0)),
                    'y': float(init_cfg.get('y', 0.0)),
                    'yaw': float(init_cfg.get('yaw', 0.0)),
                    'use_degrees': _to_bool(init_cfg.get('use_degrees', True), True),
                    'use_sim_time': use_sim_time,
                }],
            )
            actions.append(TimerAction(period=3.0, actions=[init_pose_node]))

    # 3. 상단 카메라(camera_pub) + 상단 캠 TF(top_cam_tf) — 동일 config_file & namespace & 각도 사용
    if run_camera:
        phone_cam_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_share, 'launch', 'phone_camera.launch.py')
            ),
            launch_arguments={
                'config_file': config_path,
                'namespace': ns,
                'video_source': video_source,
                'x': str(cam_x),
                'y': str(cam_y),
                'z': str(cam_z),
                'roll': str(cam_roll),
                'pitch': str(cam_pitch),
                'yaw': str(cam_yaw),
                'use_degrees': 'true' if cam_use_deg else 'false',
                'publish_camera': 'false' if use_oakd else 'true',
                'use_sim_time': 'true' if use_sim_time else 'false',
            }.items(),
        )
        actions.append(phone_cam_launch)

    # 4. YOLO + ReID + 라이다/맵 3D 위치 추적 노드 — 동일 namespace, image_topic, 카메라 각도 연동
    if run_tracker:
        tracker_node = Node(
            package='person_tracker',
            executable='multi_tracker_node',
            name='multi_person_tracker_node',
            namespace=ns,
            output='screen',
            parameters=[{
                'use_compressed': True if use_oakd else publish_compressed,
                'image_topic': tracker_image_topic,
                'use_sim_time': use_sim_time,
                'cam_x': cam_x,
                'cam_y': cam_y,
                'cam_z': cam_z,
                'cam_roll': cam_roll,
                'cam_pitch': cam_pitch,
                'cam_yaw': cam_yaw,
                'cam_use_degrees': cam_use_deg,
                'publish_cam_tf': False,
            }],
        )
        actions.append(TimerAction(period=2.0, actions=[tracker_node]))

    # 5. (선택) RViz2 네비게이션 뷰어 실행
    if run_rviz:
        try:
            tb4_viz_share = get_package_share_directory('turtlebot4_viz')
            rviz_launch = IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(tb4_viz_share, 'launch', 'view_navigation.launch.py')
                ),
                launch_arguments={
                    'namespace': ns,
                    'use_sim_time': 'true' if use_sim_time else 'false',
                }.items(),
            )
            actions.append(rviz_launch)
        except PackageNotFoundError:
            print('[WARN] turtlebot4_viz 패키지를 찾지 못해 RViz2를 건너뜁니다.')

    return actions


def generate_launch_description():
    pkg_share = get_package_share_directory('person_tracker')
    src_cfg = os.path.expanduser('~/turtlebot4_ws/src/person_tracker/config/phone_camera.yaml')
    share_cfg = os.path.join(pkg_share, 'config', 'phone_camera.yaml')
    default_cfg = src_cfg if os.path.isfile(src_cfg) else share_cfg
    default_map = os.path.join(pkg_share, 'maps', 'gallery.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('config_file', default_value=default_cfg, description='공통 카메라/초기위치 YAML 경로'),
        DeclareLaunchArgument('namespace', default_value='', description='비워두면 config_file의 namespace 사용 (기본: /robot4)'),
        DeclareLaunchArgument('map', default_value=default_map, description='저장된 맵 YAML 경로'),
        DeclareLaunchArgument('video_source', default_value='', description='비워두면 config_file의 video_source 사용 (0: 웹캠, URL: DroidCam)'),
        DeclareLaunchArgument('use_oakd', default_value='false', description='true면 camera_pub 대신 터틀봇 기본 OAK-D preview 토픽 사용'),
        DeclareLaunchArgument('run_camera', default_value='true', description='false면 다른 PC가 발행한 top_cam 토픽만 구독하고 로컬 카메라/TF는 생략'),
        DeclareLaunchArgument('x', default_value='', description='비워두면 config_file의 extrinsics.x 사용 [m]'),
        DeclareLaunchArgument('y', default_value='', description='비워두면 config_file의 extrinsics.y 사용 [m]'),
        DeclareLaunchArgument('z', default_value='', description='비워두면 config_file의 extrinsics.z 사용 [m]'),
        DeclareLaunchArgument('roll', default_value='', description='비워두면 config_file의 extrinsics.roll 사용'),
        DeclareLaunchArgument('pitch', default_value='', description='비워두면 config_file의 extrinsics.pitch 사용'),
        DeclareLaunchArgument('yaw', default_value='', description='비워두면 config_file의 extrinsics.yaw 사용 (예: 180.0)'),
        DeclareLaunchArgument('use_degrees', default_value='', description='비워두면 config_file의 extrinsics.use_degrees 사용'),
        DeclareLaunchArgument('use_localization', default_value='true', description='맵 로컬라이제이션(AMCL) 실행 여부'),
        DeclareLaunchArgument('run_init_pose', default_value='true', description='초기 위치(initial_pose) 자동 발행 여부'),
        DeclareLaunchArgument('run_tracker', default_value='true', description='multi_tracker_node 자동 실행 여부'),
        DeclareLaunchArgument('run_rviz', default_value='false', description='RViz2 동시 실행 여부'),
        DeclareLaunchArgument('use_sim_time', default_value='false', description='시뮬레이션/Bag 시간 사용 여부'),
        OpaqueFunction(function=_launch_setup),
    ])
