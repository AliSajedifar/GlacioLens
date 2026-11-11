"""P56 research-source projection; selected definitions retain their original bodies."""
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import json, hashlib

def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def hash_file(path, algorithm='sha256'):
    hasher = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def safe_relative(value):
    if not isinstance(value, str) or '\\' in value or '\x00' in value:
        raise ValueError('Invalid archive path')
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or '..' in path.parts or any(':' in p for p in path.parts):
        raise ValueError('Escaping or absolute archive path')
    return path
