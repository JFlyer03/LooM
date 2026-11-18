"""Offline tests for dated, individually verified OSS directory backups."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zlib

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('oss_backup', ROOT / 'scripts/backup_to_oss.py')
backup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backup)


class MissingKey(Exception):
    pass


class FakeCRC:
    def __init__(self):
        self.crc = 0

    def update(self, data):
        self.crc = zlib.crc32(data, self.crc)


class FakeBucket:
    def __init__(self):
        self.objects = {}
        self.bad_crc = False

    def put_object(self, key, data):
        self.objects[key] = data

    def head_object(self, key):
        if key not in self.objects:
            raise MissingKey()
        data = self.objects[key]
        return SimpleNamespace(content_length=len(data),
                               server_crc=zlib.crc32(data) + int(self.bad_crc))


class BackupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.source = Path(self.temporary.name) / 'LooM'
        self.source.mkdir()
        self.work = self.source / '.cache/oss-backup'
        self.work.mkdir(parents=True)
        self.config = self.source / 'private/oss.conf'
        self.bucket = FakeBucket()
        self.sdk = SimpleNamespace(
            utils=SimpleNamespace(Crc64=FakeCRC),
            defaults=SimpleNamespace(connection_pool_size=10),
            exceptions=SimpleNamespace(NoSuchKey=MissingKey),
            ResumableStore=Mock(),
            resumable_upload=Mock(side_effect=lambda bucket, key, filename, **kw:
                                  bucket.put_object(key, Path(filename).read_bytes())),
            Auth=Mock(), Bucket=Mock(return_value=self.bucket))

    def write(self, name, data='fixture'):
        path = self.source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data)
        return path

    def state(self):
        return dict(format='directory-v1', source=str(self.source), bucket='jflyer',
                    prefix='LooM', date='20271104', key='LooM/20271104/',
                    endpoint='https://oss-cn-hangzhou.aliyuncs.com')

    def entries(self):
        return backup.inventory(self.source, self.work, self.config)

    def test_dry_run_preserves_data_git_and_links_and_uses_date_directory(self):
        included = ['datasets/train.json', '.git/objects/object',
                    'oss-backup-current/datasets/old.json', '.env']
        excluded = ['.venv/excluded', '.cache/excluded', '.aws/credentials',
                    '.ossutilconfig', 'private/oss.conf']
        for name in included + excluded:
            self.write(name)
        (self.source / 'outside-link').symlink_to('/nonexistent/external-data')
        result = subprocess.run([
            'bash', str(ROOT / 'backup_to_oss.sh'), '--source', str(self.source),
            '--dry-run', '--date', '20271104', '--config',
            str(self.source / 'private/../private/oss.conf')],
            capture_output=True, text=True, cwd=self.source.parent)
        self.assertEqual(result.returncode, 0, result.stderr)
        preview = json.loads((self.work / 'preview.json').read_text())
        self.assertEqual(preview['key'], 'LooM/20271104/')
        entries = {item['path']: item for item in preview['entries']}
        for name in included:
            self.assertIn(name, entries)
        for name in excluded:
            self.assertNotIn(name, entries)
        self.assertEqual(entries['outside-link']['target'], '/nonexistent/external-data')
        self.assertFalse(list(self.work.glob('*.tar*')))
        self.assertFalse((self.work / 'pending.json').exists())
        self.assertFalse((self.work / 'sdk').exists())

    def test_upload_retains_paths_links_empty_dirs_and_skips_identical_objects(self):
        self.write('datasets/data.json', 'first content')
        (self.source / 'link').symlink_to('datasets/data.json')
        (self.source / 'empty').mkdir()
        state = self.state()
        backup.upload(self.sdk, self.bucket, state, self.entries(), self.work, 2)
        self.assertEqual(self.bucket.objects['LooM/20271104/datasets/data.json'], b'first content')
        self.assertEqual(self.bucket.objects['LooM/20271104/link.rclonelink'], b'datasets/data.json')
        self.assertEqual(self.bucket.objects['LooM/20271104/empty/'], b'')
        self.assertEqual(state['status'], 'complete')
        self.sdk.resumable_upload.reset_mock()
        backup.upload(self.sdk, self.bucket, self.state(), self.entries(), self.work, 2)
        self.sdk.resumable_upload.assert_not_called()
        self.write('datasets/data.json', 'new content')
        backup.upload(self.sdk, self.bucket, self.state(), self.entries(), self.work, 2)
        self.assertEqual(self.bucket.objects['LooM/20271104/datasets/data.json'], b'new content')

    def test_crc_failure_cannot_leave_previous_success_marker(self):
        self.write('data', 'content')
        key = 'LooM/20271104/__loom_backup__/manifest.json'
        self.bucket.objects[key] = b'{"status":"complete"}'
        self.bucket.bad_crc = True
        with self.assertRaisesRegex(RuntimeError, '未完成'):
            backup.upload(self.sdk, self.bucket, self.state(), self.entries(), self.work, 1)
        self.assertEqual(json.loads(self.bucket.objects[key])['status'], 'in_progress')
        self.assertEqual(json.loads((self.work / 'failed-files.json').read_text()), ['data'])

    def test_resume_retains_original_date_even_after_midnight(self):
        self.write('data')
        state = self.state()
        backup.save_json(self.work / 'pending.json', state)
        argv = ['backup', '--source', str(self.source)]
        with patch.object(sys, 'argv', argv), \
                patch.object(backup, 'credentials', return_value=('test-id', 'test-secret', None)), \
                patch.object(backup, 'load_sdk', return_value=self.sdk), \
                patch.object(backup, 'upload', side_effect=RuntimeError('network interrupted')):
            with self.assertRaisesRegex(RuntimeError, 'network interrupted'):
                backup.main()
        self.assertTrue((self.work / 'pending.json').exists())
        with patch.object(sys, 'argv', argv), \
                patch.object(backup, 'credentials', return_value=('test-id', 'test-secret', None)), \
                patch.object(backup, 'load_sdk', return_value=self.sdk):
            backup.main()
        self.assertFalse((self.work / 'pending.json').exists())
        report = json.loads((self.work / 'last-success.json').read_text())
        self.assertEqual(report['key'], 'LooM/20271104/')
        self.assertEqual(report['status'], 'complete')

    def test_link_collision_is_rejected_before_upload(self):
        (self.source / 'link').symlink_to('target')
        self.write('link.rclonelink')
        with self.assertRaisesRegex(RuntimeError, '重名'):
            self.entries()

    def test_custom_work_directory_is_excluded(self):
        work = self.source / 'local-backups'
        work.mkdir()
        (work / 'old.tar.gz').write_text('old')
        entries = backup.inventory(self.source, work, self.config)
        self.assertFalse(any(e['path'].startswith('local-backups') for e in entries))

    def test_incomplete_environment_credentials_do_not_mix_with_file(self):
        with patch.dict(os.environ, {'OSS_ACCESS_KEY_ID': 'test-id'}, clear=True):
            with self.assertRaisesRegex(RuntimeError, '同时设置'):
                backup.credentials(self.config)


if __name__ == '__main__':
    unittest.main()
