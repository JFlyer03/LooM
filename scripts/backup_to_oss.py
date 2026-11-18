#!/usr/bin/env python3
"""Create a full project archive and upload a verified, resumable OSS snapshot."""

import argparse
import configparser
import fcntl
import getpass
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import stat
from concurrent.futures import ThreadPoolExecutor, as_completed
from zoneinfo import ZoneInfo
import time
from datetime import datetime, timezone


EXCLUDED_DIRS = {'.cache', '.venv', 'venv', '__pycache__', '.pytest_cache',
                 '.mypy_cache', '.ruff_cache', '.aws'}


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def inventory(source, work_dir, config_path):
    """Walk ignored data too, preserving links without following their targets."""
    entries = []
    credential_paths = {config_path, config_path.resolve()}
    if (source / '__loom_backup__').exists():
        raise RuntimeError('源目录中的 __loom_backup__ 与备份元信息目录冲突')

    def walk(directory):
        visible = []
        for path in sorted(directory.iterdir()):
            if (path.name in EXCLUDED_DIRS or path == work_dir
                    or path in credential_paths or path.name == '.ossutilconfig'):
                continue
            visible.append(path)
        for path in visible:
            info = path.lstat()
            entry = dict(path=path.relative_to(source).as_posix(),
                         mode=stat.S_IMODE(info.st_mode), mtime_ns=info.st_mtime_ns,
                         size=info.st_size)
            if path.is_symlink():
                entry.update(type='symlink', target=os.readlink(path))
                entry['object'] = entry['path'] + '.rclonelink'
            elif stat.S_ISDIR(info.st_mode):
                entry.update(type='directory', object=entry['path'] + '/')
                entry['empty'] = not walk(path)
            elif stat.S_ISREG(info.st_mode):
                entry.update(type='file', object=entry['path'])
            else:
                raise RuntimeError(f'无法备份特殊文件：{path}')
            entries.append(entry)
        return bool(visible)

    walk(source)
    keys = [entry['object'] for entry in entries]
    if len(keys) != len(set(keys)):
        raise RuntimeError('符号链接的 .rclonelink 对象与源文件重名，请先解决冲突')
    return entries


def hashes(path, oss=None):
    sha = hashlib.sha256()
    crc = oss.utils.Crc64() if oss else None
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            sha.update(chunk)
            if crc is not None:
                crc.update(chunk)
    return sha.hexdigest(), crc.crc if crc is not None else None


def configure(path):
    if not sys.stdin.isatty():
        raise RuntimeError('未配置凭据。请先在终端运行 ./backup_to_oss.sh --configure')
    access_id = input('AccessKey ID: ').strip()
    secret = getpass.getpass('AccessKey Secret（输入不回显）: ').strip()
    token = getpass.getpass('STS Token（长期 AccessKey 直接回车）: ').strip()
    if not access_id or not secret:
        raise RuntimeError('AccessKey ID 和 Secret 不能为空')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise RuntimeError('凭据配置路径不能是符号链接')
    config = configparser.ConfigParser(interpolation=None)
    if path.exists():
        config.read(path)
    config['Credentials'] = dict(endpoint='https://oss-cn-hangzhou.aliyuncs.com',
                                 accessKeyID=access_id, accessKeySecret=secret,
                                 stsToken=token)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(descriptor, 0o600)
    with os.fdopen(descriptor, 'w') as stream:
        config.write(stream)
    print(f'凭据已保存到 {path}（权限 600）', flush=True)


def credentials(path):
    access_id = os.getenv('OSS_ACCESS_KEY_ID')
    secret = os.getenv('OSS_ACCESS_KEY_SECRET')
    if access_id or secret:
        if not access_id or not secret:
            raise RuntimeError('必须同时设置 OSS_ACCESS_KEY_ID 和 OSS_ACCESS_KEY_SECRET')
        return access_id, secret, os.getenv('OSS_SESSION_TOKEN') or os.getenv('OSS_STS_TOKEN')
    if not path.exists():
        configure(path)
    config = configparser.ConfigParser(interpolation=None)
    config.read(path)
    section = config['Credentials'] if config.has_section('Credentials') else config['DEFAULT']

    def get(*names):
        return next((section[n] for n in names if section.get(n)), None)

    access_id = get('accesskeyid', 'access-key-id', 'access_key_id')
    secret = get('accesskeysecret', 'access-key-secret', 'access_key_secret')
    if not access_id or not secret:
        raise RuntimeError('配置文件缺少凭据，请运行 ./backup_to_oss.sh --configure')
    return access_id, secret, get('ststoken', 'sts-token', 'sts_token')


def load_sdk(work_dir):
    sdk_dir = work_dir / 'sdk'
    sys.path.insert(0, str(sdk_dir))
    try:
        return importlib.import_module('oss2')
    except ModuleNotFoundError:
        print('首次运行：安装 OSS SDK 到备份缓存目录……', flush=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade',
                        '--target', str(sdk_dir), 'oss2==2.19.1'], check=True)
        importlib.invalidate_caches()
        return importlib.import_module('oss2')


def retry(operation, attempts=5):
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as error:
            if attempt == attempts - 1:
                raise
            # Never print raw SDK request details or credential values.
            print(f'请求失败（{type(error).__name__}），正在重试 {attempt + 1}/{attempts - 1}', flush=True)
            time.sleep(2 ** attempt)


def upload(oss, bucket, state, entries, work_dir, threads):
    source = Path(state['source'])
    prefix = state['key']
    manifest_key = prefix + '__loom_backup__/manifest.json'
    state['status'] = 'in_progress'
    retry(lambda: bucket.put_object(manifest_key, json.dumps(state).encode()))
    oss.defaults.connection_pool_size = max(threads, oss.defaults.connection_pool_size)
    last_update = time.monotonic()
    completed = skipped = 0
    results = []
    errors = []

    def one(entry):
        path = source / entry['path']
        key = prefix + entry['object']
        if entry['type'] == 'directory':
            if entry['empty']:
                retry(lambda: bucket.put_object(key, b''))
            return entry, False
        before = path.lstat()
        if (before.st_mtime_ns != entry['mtime_ns'] or before.st_size != entry['size']
                or stat.S_IMODE(before.st_mode) != entry['mode']
                or (stat.S_ISLNK(before.st_mode) != (entry['type'] == 'symlink'))):
            raise RuntimeError('扫描后源文件发生变化：' + entry['path'])
        if entry['type'] == 'symlink':
            payload = os.readlink(path).encode('utf-8')
            if os.readlink(path) != entry['target']:
                raise RuntimeError('源符号链接发生变化：' + entry['path'])
            digest = oss.utils.Crc64()
            digest.update(payload)
            sha, crc, size = hashlib.sha256(payload).hexdigest(), digest.crc, len(payload)
        else:
            sha, crc = hashes(path, oss)
            size = before.st_size

        def head():
            try:
                return bucket.head_object(key)
            except oss.exceptions.NoSuchKey:
                return None

        remote = retry(head)
        unchanged = remote is not None and remote.content_length == size and remote.server_crc == crc
        if not unchanged:
            if entry['type'] == 'symlink':
                retry(lambda: bucket.put_object(key, payload))
            else:
                retry(lambda: oss.resumable_upload(
                    bucket, key, str(path),
                    store=oss.ResumableStore(root=str(work_dir / 'checkpoints')),
                    multipart_threshold=8 * 1024**2, part_size=8 * 1024**2,
                    num_threads=1, headers={'x-oss-meta-sha256': sha}))
            remote = retry(head)
            if remote is None or remote.content_length != size or remote.server_crc != crc:
                raise RuntimeError('OSS 大小或 CRC64 校验失败：' + entry['path'])
        after = path.lstat()
        if ((after.st_size, after.st_mtime_ns, after.st_mode, after.st_ino)
                != (before.st_size, before.st_mtime_ns, before.st_mode, before.st_ino)):
            raise RuntimeError('上传期间源文件发生变化：' + entry['path'])
        return dict(entry, sha256=sha, crc64=str(crc), object_size=size), unchanged

    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = {pool.submit(one, entry): entry for entry in entries}
        for future in as_completed(futures):
            try:
                entry, unchanged = future.result()
                results.append(entry)
                completed += 1
                skipped += int(unchanged)
            except Exception as error:
                errors.append(futures[future]['path'])
                print(f"文件失败：{futures[future]['path']}（{type(error).__name__}）", flush=True)
            if time.monotonic() - last_update >= 15:
                print(f'已核对 {completed}/{len(entries)} 项，复用 {skipped} 个相同对象，失败 {len(errors)} 项', flush=True)
                last_update = time.monotonic()
    if errors:
        save_json(work_dir / 'failed-files.json', errors)
        raise RuntimeError(f'{len(errors)} 项未完成，请查看缓存目录 failed-files.json，重新运行可续传')
    state.update(status='complete', completed_at=datetime.now(timezone.utc).isoformat(),
                 entries=sorted(results, key=lambda entry: entry['path']), reused=skipped)
    retry(lambda: bucket.put_object(manifest_key, json.dumps(state, ensure_ascii=False, indent=2).encode()))


def main():
    parser = argparse.ArgumentParser(description='按日期目录将完整 LooM 项目逐文件备份到阿里云 OSS')
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--work-dir', type=Path, help='本地清单和续传缓存目录')
    parser.add_argument('--bucket', default='jflyer')
    parser.add_argument('--prefix', default='LooM')
    parser.add_argument('--date', help='备份日期 YYYYMMDD，默认香港时区今天')
    parser.add_argument('--endpoint', default='https://oss-cn-hangzhou.aliyuncs.com')
    parser.add_argument('--config', type=Path, default=Path.home() / '.ossutilconfig')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--configure', action='store_true', help='配置凭据后退出')
    parser.add_argument('--dry-run', action='store_true', help='仅预览文件清单和目标目录，不读取凭据或访问 OSS')
    args = parser.parse_args()
    os.umask(0o077)
    source = args.source.expanduser().resolve()
    config_path = Path(os.path.abspath(args.config.expanduser()))
    if args.configure:
        configure(config_path)
        return
    if not source.is_dir() or source == Path('/'):
        parser.error('--source 必须是项目目录，不能是根目录')
    if args.threads < 1:
        parser.error('--threads 必须大于 0')
    prefix = args.prefix.strip('/')
    if not prefix or '..' in Path(prefix).parts:
        parser.error('--prefix 必须是有效的 OSS 目录前缀')
    work_dir = (args.work_dir or source / '.cache' / 'oss-backup').expanduser().resolve()
    if work_dir == source or work_dir in source.parents:
        parser.error('--work-dir 不能是源目录或源目录的父目录')
    work_dir.mkdir(parents=True, exist_ok=True)
    with (work_dir / 'backup.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('同一缓存目录已有备份任务运行') from None
        day = args.date or datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d')
        if len(day) != 8 or not day.isdigit():
            parser.error('--date 必须为 YYYYMMDD')
        try:
            datetime.strptime(day, '%Y%m%d')
        except ValueError:
            parser.error('--date 必须为有效日期 YYYYMMDD')
        pending = work_dir / 'pending.json'
        state = dict(format='directory-v1', source=str(source), bucket=args.bucket,
                     prefix=prefix, endpoint=args.endpoint, date=day, key=f'{prefix}/{day}/')
        if pending.exists() and not args.dry_run:
            previous = json.loads(pending.read_text())
            if previous.get('format') != 'directory-v1':
                raise RuntimeError('缓存中存在旧版压缩包任务，请先完成旧任务或指定新的 --work-dir')
            for field in ('source', 'bucket', 'prefix', 'endpoint'):
                if state[field] != previous[field]:
                    raise RuntimeError('缓存有不同来源或目标的任务，请使用其他 --work-dir')
            if args.date and day != previous['date']:
                raise RuntimeError('指定日期与待续传任务不同，请使用其他 --work-dir')
            state = previous
            print('继续未完成的目录备份：' + state['key'], flush=True)
        entries = inventory(source, work_dir, config_path)
        state['files'] = sum(entry['type'] != 'directory' for entry in entries)
        state['bytes'] = sum(entry['size'] for entry in entries if entry['type'] == 'file')
        print(f"目标：oss://{args.bucket}/{state['key']}，文件 {state['files']} 个，约 {state['bytes'] / 1024**3:.2f} GiB", flush=True)
        if args.dry_run:
            save_json(work_dir / 'preview.json', dict(state, entries=entries))
            print(f'预览已保存到 {work_dir / "preview.json"}，未访问 OSS。', flush=True)
            return
        creds = credentials(config_path)
        oss = load_sdk(work_dir)
        save_json(pending, state)
        access_id, secret, token = creds
        auth = oss.StsAuth(access_id, secret, token) if token else oss.Auth(access_id, secret)
        bucket = oss.Bucket(auth, args.endpoint, args.bucket, connect_timeout=60, enable_crc=True)
        upload(oss, bucket, state, entries, work_dir, args.threads)
        save_json(work_dir / 'last-success.json', state)
        pending.unlink()
        print(f"备份成功：oss://{args.bucket}/{state['key']}，文件大小与 CRC64 校验通过。", flush=True)
        print('同日备份更新同名对象，不删除旧对象；不同日期分别保留。', flush=True)



if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n已中断；重新运行相同命令可继续目录备份。', file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print(f'备份失败：{type(error).__name__}', file=sys.stderr)
        if isinstance(error, (RuntimeError, ValueError, OSError)):
            print(str(error), file=sys.stderr)
        print('修复配置或网络后，重新运行相同命令；未完成文件会自动续传，相同对象会跳过。', file=sys.stderr)
        sys.exit(1)
