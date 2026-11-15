import time
from typing import Dict, List, Optional, Tuple
import psutil
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosidl_runtime_py.utilities import get_message


def fmt_rate(bytes_per_sec: float) -> str:
    kb = bytes_per_sec / 1024.0
    mbps = (bytes_per_sec * 8.0) / (1024.0 * 1024.0)
    if kb >= 1024.0:
        return f'{kb / 1024.0:6.2f} MB/s ({mbps:5.2f} Mbps)'
    return f'{kb:6.1f} KB/s ({mbps:5.2f} Mbps)'


class TopicStat:
    def __init__(self, topic: str, msg_type: str, is_baseline: bool):
        self.topic = topic
        self.msg_type = msg_type
        self.is_baseline = is_baseline
        self.bytes_in_window = 0
        self.msgs_in_window = 0
        self.rate_bps = 0.0
        self.hz = 0.0

    def on_msg(self, raw_bytes: bytes):
        self.bytes_in_window += len(raw_bytes)
        self.msgs_in_window += 1

    def step(self, dt: float):
        if dt > 0.0:
            self.rate_bps = self.bytes_in_window / dt
            self.hz = self.msgs_in_window / dt
        self.bytes_in_window = 0
        self.msgs_in_window = 0


class NetMonitorNode(Node):
    """
    1) [1. robot4 기본 토픽]: 터틀봇4가 기본 발행하는 센서/상태 토픽 (/robot4/scan, /robot4/odom, /robot4/tf 등)
    2) [2. 추가 토픽]: person_tracker(camera_pub, multi_tracker_node) 및 Nav2/RViz 실행 시 추가되는 토픽
    3) [3. 최종 사용량]: [1] 소계, [2] 소계, 토픽 총합([1]+[2]), 그리고 실제 Wi-Fi(wlp2s0) RX/TX 통신량 비교
    """

    IGNORE_TOPICS = {'/parameter_events', '/rosout', '/diagnostics'}
    UNUSED_CODEC_SUFFIXES = ('/zstd', '/theora', '/compressedDepth')
    NAV2_AND_TRACKER_KEYWORDS = (
        '/target_person',
        'costmap',
        '/plan',
        '/local_plan',
        '/cmd_vel',
        '/trajectories',
        '/behavior_tree_log',
        '/amcl_pose',
        '/particle_cloud',
        '/map',
        '/waypoint',
        '/navigate_to_pose',
    )

    def __init__(self):
        super().__init__('net_monitor_node')

        self.declare_parameter('iface', 'wlp2s0')
        self.declare_parameter('interval', 2.0)
        self.declare_parameter('active_only', False)
        self.declare_parameter('include_idle_robot_camera', False)

        self.iface = str(self.get_parameter('iface').value)
        self.interval = float(self.get_parameter('interval').value)
        self.active_only = bool(self.get_parameter('active_only').value)
        self.include_idle_robot_camera = bool(self.get_parameter('include_idle_robot_camera').value)

        self.last_step_time = time.monotonic()
        self.stats: Dict[str, TopicStat] = {}
        self.subs = {}

        # 모니터가 토픽 구독을 열기 전 순수 Wi-Fi(wlp2s0) 초기 통신량 측정
        self.baseline_nic_rx_bps, self.baseline_nic_tx_bps = self._measure_initial_nic(0.5)
        self.last_nic = self._read_nic()

        self.timer = self.create_timer(self.interval, self._on_timer)
        self.get_logger().info(
            f'NetMonitor 시작 | 인터페이스: {self.iface}, 주기: {self.interval:.1f}s '
            f'(모니터 구독 전 초기 Wi-Fi RX+TX: {fmt_rate(self.baseline_nic_rx_bps + self.baseline_nic_tx_bps)})'
        )

    def _read_nic(self) -> Optional[Tuple[int, int]]:
        nics = psutil.net_io_counters(pernic=True)
        if self.iface in nics:
            c = nics[self.iface]
            return c.bytes_recv, c.bytes_sent
        return None

    def _measure_initial_nic(self, duration: float) -> Tuple[float, float]:
        t0 = self._read_nic()
        if t0 is None:
            return 0.0, 0.0
        time.sleep(duration)
        t1 = self._read_nic()
        if t1 is None:
            return 0.0, 0.0
        return (t1[0] - t0[0]) / duration, (t1[1] - t0[1]) / duration

    def _is_added_topic(self, topic: str, pubs) -> bool:
        # 1) camera_publisher_node나 tracker 노드가 발행하는 토픽은 무조건 [2. 추가 토픽]
        for p in pubs:
            if any(k in p.node_name for k in ('camera_publisher', 'tracker', 'nav', 'slam', 'rviz')):
                return True
        # 2) /robot4 네임스페이스가 아닌 토픽(/oakd/..., /target_person/... 등)은 [2. 추가 토픽]
        if not topic.startswith('/robot4/'):
            return True
        # 3) /robot4 내부라도 Nav2/제어/맵 관련 토픽은 [2. 추가 토픽]
        return any(k in topic for k in self.NAV2_AND_TRACKER_KEYWORDS)

    def _discover_topics(self):
        for topic, types in self.get_topic_names_and_types():
            if topic in self.IGNORE_TOPICS or topic.startswith('/robot4/_do_not_use') or not types:
                continue
            if topic in self.stats:
                continue

            pubs = self.get_publishers_info_by_topic(topic)
            if not pubs:
                continue

            other_subs = [
                s for s in self.get_subscriptions_info_by_topic(topic)
                if s.node_name != self.get_name()
            ]
            # 아무도 안 쓰는 보조 코덱(zstd/theora) 제외
            if topic.endswith(self.UNUSED_CODEC_SUFFIXES) and not other_subs:
                continue
            # 로봇 자체 OAK-D 카메라 토픽(/robot4/oakd/rgb/...)은 다른 노드가 실제 구독 중일 때만 측정
            # (모니터가 구독해서 잠자던 로봇 카메라 스트림 5MB/s를 강제로 깨우는 현상 방지)
            if (
                not self.include_idle_robot_camera
                and topic.startswith('/robot4/oakd/')
                and ('image' in topic or 'stereo' in topic)
                and not other_subs
            ):
                continue
            if self.active_only and not other_subs:
                continue

            msg_type_str = types[0]
            try:
                msg_cls = get_message(msg_type_str)
            except Exception:
                continue

            is_base = not self._is_added_topic(topic, pubs)
            stat = TopicStat(topic, msg_type_str, is_baseline=is_base)
            sub = self.create_subscription(
                msg_cls,
                topic,
                stat.on_msg,
                qos_profile_sensor_data,
                raw=True,
            )
            self.stats[topic] = stat
            self.subs[topic] = sub

    def _on_timer(self):
        now = time.monotonic()
        dt = now - self.last_step_time
        self.last_step_time = now

        self._discover_topics()

        nic_now = self._read_nic()
        rx_bps = tx_bps = 0.0
        if nic_now is not None and self.last_nic is not None and dt > 0:
            rx_bps = (nic_now[0] - self.last_nic[0]) / dt
            tx_bps = (nic_now[1] - self.last_nic[1]) / dt
            self.last_nic = nic_now

        base_list: List[TopicStat] = []
        added_list: List[TopicStat] = []
        for st in self.stats.values():
            st.step(dt)
            if st.is_baseline:
                base_list.append(st)
            else:
                added_list.append(st)

        base_list.sort(key=lambda s: s.rate_bps, reverse=True)
        added_list.sort(key=lambda s: s.rate_bps, reverse=True)

        base_sum = sum(s.rate_bps for s in base_list)
        added_sum = sum(s.rate_bps for s in added_list)
        total_topic_sum = base_sum + added_sum

        lines = [
            '\033[2J\033[H' + '=' * 88,
            f' 📊 ROS 2 네트워크 사용량 분리 모니터 (인터페이스: {self.iface}, 갱신주기: {dt:.1f}s)',
            '=' * 88,
            f' [1] robot4 기본 토픽 (활성 {sum(1 for s in base_list if s.rate_bps > 0)}개)                 '
            f'소계: {fmt_rate(base_sum)}',
            '-' * 88,
        ]
        active_base = [s for s in base_list if s.rate_bps > 0 or s.hz > 0]
        if not active_base:
            lines.append('   (수신 중인 robot4 기본 토픽 없음)')
        for s in active_base[:12]:
            lines.append(f'   • {s.topic:<46s} | {s.hz:6.1f} Hz | {fmt_rate(s.rate_bps)}')

        lines.extend([
            '-' * 88,
            f' [2] 추가된 토픽 (person_tracker / 웹캠 / Nav2, 활성 {sum(1 for s in added_list if s.rate_bps > 0)}개) '
            f'소계: {fmt_rate(added_sum)}',
            '-' * 88,
        ])
        active_added = [s for s in added_list if s.rate_bps > 0 or s.hz > 0]
        if not active_added:
            lines.append('   (추가 활성 토픽 없음 — camera_pub / multi_tracker_node / Nav2 실행 시 여기에 표시됩니다)')
        for s in active_added[:12]:
            lines.append(f'   • {s.topic:<46s} | {s.hz:6.1f} Hz | {fmt_rate(s.rate_bps)}')

        base_nic_total = self.baseline_nic_rx_bps + self.baseline_nic_tx_bps
        cur_nic_total = rx_bps + tx_bps
        delta_nic = cur_nic_total - base_nic_total
        sign = '+' if delta_nic >= 0 else '-'

        lines.extend([
            '=' * 88,
            ' [3] 최종 사용량 요약 (그룹별 소계 & 실제 Wi-Fi 총합)',
            '-' * 88,
            f'   • [1] robot4 기본 토픽 소계             : {fmt_rate(base_sum)}',
            f'   • [2] 추가 토픽 소계 (Tracker/Cam/Nav2) : {fmt_rate(added_sum)}',
            f'   • ROS 2 토픽 총합 ([1] + [2])           : {fmt_rate(total_topic_sum)}',
            '-' * 88,
            f'   • 실제 Wi-Fi ({self.iface}) 현재 수신(RX)     : {fmt_rate(rx_bps)}',
            f'   • 실제 Wi-Fi ({self.iface}) 현재 송신(TX)     : {fmt_rate(tx_bps)}',
            f'   • 실제 Wi-Fi ({self.iface}) 최종 총합(RX+TX)  : {fmt_rate(cur_nic_total)} '
            f'(모니터 시작 전 대비 {sign}{fmt_rate(abs(delta_nic))})',
            '=' * 88,
        ])
        print('\n'.join(lines), flush=True)


def main(args=None):
    rclpy.init(args=args)
    node = NetMonitorNode()
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
