"""Durable immutable snapshots and locked, checksummed JSONL journals (Linux)."""
import fcntl
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def atomic_json(path, value, immutable=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock(str(path) + ".lock"):
        if immutable and path.exists():
            if json.loads(path.read_text()) == value:
                return
            raise FileExistsError(f"Refusing to overwrite {path}")
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
        try:
            with os.fdopen(fd, "w") as stream:
                stream.write(canonical(value) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with lock(str(path) + ".lock"):
        for index, line in enumerate(path.read_text().splitlines(), 1):
            try:
                row = json.loads(line)
                checksum = row.pop("_checksum", None)
                if checksum is not None and checksum != digest(row):
                    raise ValueError("checksum mismatch")
                records.append(row)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Corrupt JSONL {path}:{index}; preserve and repair explicitly") from exc
    return records


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(row, _checksum=digest(row))
    with lock(str(path) + ".lock"):
        if path.exists() and path.stat().st_size:
            with path.open('rb') as previous:
                previous.seek(-1, os.SEEK_END)
                if previous.read(1) != b'\n':
                    raise ValueError(f'Incomplete JSONL tail: {path}; repair explicitly before appending')
        with path.open("ab") as stream:
            stream.write((canonical(record) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())
