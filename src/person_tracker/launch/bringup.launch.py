import os
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _to_bool(val: str) -> bool:
    return str(val).strip().lower() in ('true', '1', 'yes', 'on')


def _launch_setup(context, *args, **kwargs):
    pkg_share = get_package_share_directory('person_tracker')

    ns = LaunchConfiguration('namespace').perform(context).strip()
    if not ns.startswith('/'):
        ns = '/' + ns
    ns = ns.rstrip('/') or '/robot4'

    map_path = LaunchConfiguration('map').perform(context).strip()
    video_source = LaunchConfiguration('video_source').perform(context).strip()
    use_localization = _to_bool(LaunchConfiguration('use_localization').perform(context))
    run_init_pose = _to_bool(LaunchConfiguration('run_init_pose').perform(context))
    run_tracker = _to_bool(LaunchConfiguration('run_tracker').perform(context))
    run_rviz = _to_bool(LaunchConfiguration('run_rviz').perform(context))
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context).strip()

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
                    'use_sim_time': use_sim_time,
                }.items(),
            )
            actions.append(loc_launch)
        except PackageNotFoundError:
            print('[WARN] turtlebot4_navigation 패키지를 찾지 못해 localization을 건너뜁니다.')

        # 2. 초기 위치 발행 (AMCL 구독자가 준비될 때까지 자동 대기 후 3회 발행)
        if run_init_pose:
            init_pose_node = Node(
                package='person_tracker',
                executable='init_pose',
                name='init_pose_node',
                namespace=ns,
                output='screen',
            )
            actions.append(TimerAction(period=3.0, actions=[init_pose_node]))

    # 3. 상단 카메라(노트북 웹캠 기본: video_source:=0) + 상단 캠 TF 발행
    phone_cam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share, 'launch', 'phone_camera.launch.py')
        ),
        launch_arguments={
            'namespace': ns,
            'video_source': video_source,
        }.items(),
    )
    actions.append(phone_cam_launch)

    # 4. YOLO + ReID + 라이다/맵 3D 위치 추적 노드
    if run_tracker:
        tracker_node = Node(
            package='person_tracker',
            executable='multi_tracker_node',
            name='multi_person_tracker_node',
            output='screen',
            parameters=[{
                'use_sim_time': _to_bool(use_sim_time),
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
                    'use_sim_time': use_sim_time,
                }.items(),
            )
            actions.append(rviz_launch)
        except PackageNotFoundError:
            print('[WARN] turtlebot4_viz 패키지를 찾지 못해 RViz2를 건너뜁니다.')

    return actions


def generate_launch_description():
    pkg_share = get_package_share_directory('person_tracker')
    default_map = os.path.join(pkg_share, 'maps', 'gallery.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('namespace', default_value='/robot4', description='로봇 네임스페이스 (예: /robot4)'),
        DeclareLaunchArgument('map', default_value=default_map, description='저장된 맵 YAML 경로'),
        DeclareLaunchArgument('video_source', default_value='0', description='카메라 장치 번호(0: 노트북 웹캠) 또는 스트림 URL'),
        DeclareLaunchArgument('use_localization', default_value='true', description='맵 로컬라이제이션(AMCL) 실행 여부'),
        DeclareLaunchArgument('run_init_pose', default_value='true', description='초기 위치(0,0,0) 자동 발행 여부'),
        DeclareLaunchArgument('run_tracker', default_value='true', description='multi_tracker_node 자동 실행 여부'),
        DeclareLaunchArgument('run_rviz', default_value='false', description='RViz2 동시 실행 여부'),
        DeclareLaunchArgument('use_sim_time', default_value='false', description='시뮬레이션/Bag 시간 사용 여부'),
        OpaqueFunction(function=_launch_setup),
    ])
