"""Local credential setup for launchers; keys never enter experiment configuration."""
import os
import stat
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KEY_FILE = PROJECT_ROOT / '.secrets' / 'openai_api_key'


def validate_key(value):
    value = value.strip()
    if not value or any(c.isspace() for c in value):
        raise ValueError('키가 비어 있거나 중간에 공백이 있습니다. 키만 입력하세요.')
    return value


def save_key(value, path=KEY_FILE):
    value = validate_key(value)
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    # Exclusive creation also prevents following an existing symlink.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        stream.write(value + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def load_key_into_environment(path=KEY_FILE):
    if os.environ.get('OPENAI_API_KEY', '').strip():
        os.environ['OPENAI_API_KEY'] = validate_key(os.environ['OPENAI_API_KEY'])
        return 'environment'
    path = Path(path)
    if not path.is_file():
        raise RuntimeError('API 키가 없습니다. 먼저 python scripts/configure_api_key.py 를 실행하세요.')
    if stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise PermissionError('키 파일 권한이 너무 넓습니다. chmod 600 .secrets/openai_api_key 를 실행하세요.')
    os.environ['OPENAI_API_KEY'] = validate_key(path.read_text())
    return 'private_file'
