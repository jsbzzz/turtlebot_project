import glob
import os
import sys


def bootstrap_venv():
    """
    다른 기기나 다른 Python 버전(3.10, 3.12 등)에서도 가상환경의 site-packages를
    자동으로 탐색하여 sys.path 최상단에 등록합니다.
    """
    candidates = []
    if os.environ.get('VIRTUAL_ENV'):
        candidates.append(os.environ['VIRTUAL_ENV'])

    candidates.extend([
        os.path.expanduser('~/venvs/rokey_venv'),
        os.path.expanduser('~/venv/rokey_venv'),
        os.path.expanduser('~/.venv'),
    ])

    for venv_root in candidates:
        if not os.path.isdir(venv_root):
            continue
        for site_dir in sorted(glob.glob(os.path.join(venv_root, 'lib', 'python3*', 'site-packages')), reverse=True):
            if os.path.isdir(site_dir) and site_dir not in sys.path:
                sys.path.insert(0, site_dir)
                return site_dir
    return None


def resolve_model_path(model_name: str) -> str:
    """
    패키지 내부 models/ 폴더에 가중치 파일(.pt)이 있으면 우선 사용하여,
    다른 기기에서 인터넷 없이도 즉시 모델을 로드할 수 있게 합니다.
    """
    if os.path.isabs(model_name) and os.path.isfile(model_name):
        return model_name

    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bundled = os.path.join(pkg_root, 'models', model_name)
    if os.path.isfile(bundled):
        return bundled

    try:
        from ament_index_python.packages import get_package_share_directory
        share_bundled = os.path.join(
            get_package_share_directory('person_tracker'), 'models', model_name
        )
        if os.path.isfile(share_bundled):
            return share_bundled
    except Exception:
        pass

    return model_name


def resolve_image_topic(ns: str, use_compressed: bool, custom_topic: str = '') -> str:
    """
    상단 추가 카메라(top_cam)와 기본 터틀봇 카메라(oakd)가 충돌하지 않도록
    config/phone_camera.yaml 설정을 우선 참조하여 기본 구독 토픽을 결정합니다.
    """
    if custom_topic:
        return custom_topic

    ns_prefix = '' if ns == '/' else ns
    sub_suffix = '/top_cam/image_raw/compressed' if use_compressed else '/top_cam/image_raw'
    if ns_prefix:
        return ns_prefix + sub_suffix

    cfg_candidates = [
        os.path.expanduser('~/turtlebot4_ws/src/person_tracker/config/phone_camera.yaml'),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'phone_camera.yaml'),
    ]
    for cfg_path in cfg_candidates:
        if os.path.isfile(cfg_path):
            try:
                import yaml
                with open(cfg_path, 'r', encoding='utf-8') as f:
                    cfg = yaml.safe_load(f) or {}
                cfg_ns = str(cfg.get('namespace', '')).rstrip('/')
                topics_cfg = cfg.get('topics', {})
                key = 'compressed_topic' if use_compressed else 'raw_topic'
                topic_path = str(topics_cfg.get(key, sub_suffix))
                if not topic_path.startswith('/'):
                    topic_path = '/' + topic_path
                return (cfg_ns + topic_path) if cfg_ns else topic_path
            except Exception:
                pass

    return sub_suffix
