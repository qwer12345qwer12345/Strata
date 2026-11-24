import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import threading
import unittest
import urllib.request
from pathlib import Path

from l2 import startup_benchmark as bench

FAKE = '''
import argparse, json, os, time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
p = argparse.ArgumentParser()
p.add_argument('command'); p.add_argument('--profile'); p.add_argument('--ready-file'); p.add_argument('--shutdown-timeout')
a = p.parse_args(); cfg = json.loads(Path(a.profile).read_text())
root = Path(__file__).parent
class API(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_GET(self):
        self.send_response(200); self.end_headers()
        self.wfile.write(json.dumps({'status': 'ok', 'images': bool(cfg.get('vision'))}).encode())
    def do_POST(self):
        assert self.path == '/v1/chat/completions'
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        assert body['stream'] and body['temperature'] == 0 and body['seed'] == 123
        (root / 'bench-entered').touch()
        for _ in range(250):
            if (root / 'release').exists(): break
            time.sleep(.02)
        self.send_response(200); self.send_header('content-type', 'text/event-stream'); self.end_headers()
        self.wfile.write(b'data: {"choices":[{"delta":{"content":"dummy"}}]}\\n\\n')
        if (root / 'fail').exists(): return
        self.wfile.write(b'data: {"choices":[{"finish_reason":"length"}],"timings":{"prompt_ms":10,"predicted_ms":100,"predicted_n":8,"predicted_per_second":80,"draft_n_accepted":4}}\\n\\n')
        self.wfile.write(b'data: [DONE]\\n\\n')
s = ThreadingHTTPServer(('127.0.0.1', cfg['port']), API)
Path(a.ready_file).write_text(json.dumps({'api_state':'ready','pid':os.getpid()}))
s.serve_forever()
'''


def ports():
    with socket.socket() as a, socket.socket() as b:
        a.bind(('127.0.0.1', 0)); b.bind(('127.0.0.1', 0))
        return a.getsockname()[1], b.getsockname()[1]


def wait_until(check, process, seconds=10):
    until = time.monotonic() + seconds
    while not check():
        if process.poll() is not None or time.monotonic() >= until:
            raise AssertionError(f'Condition not met; subprocess exit {process.poll()}')
        time.sleep(.02)


class StartupBenchmark(unittest.TestCase):
    def launch(self, root, fail, delayed_save=False, vision=False):
        here = Path(__file__).resolve().parent
        (root / 'l2').mkdir()
        for name in ('managed_server.py', 'reference_controller.py', 'startup_benchmark.py', 'runtime_config.py'):
            shutil.copyfile(here / name, root / 'l2' / name)
        if delayed_save:
            with (root / 'l2/reference_controller.py').open('a') as f:
                f.write("\n_test_root = Path(__file__).resolve().parents[1]\n"
                        "def after_turn(body, response_ok):\n"
                        " if not response_ok: return\n"
                        " (_test_root / 'save-entered').touch()\n"
                        " deadline = time.monotonic() + 5\n"
                        " while not (_test_root / 'save-release').exists():\n"
                        "  if time.monotonic() > deadline: raise TimeoutError('test save')\n"
                        "  time.sleep(.01)\n"
                        " (_test_root / 'save-done').touch()\n")
        (root / 'strata.py').write_text(FAKE)
        if fail:
            (root / 'fail').touch()
        front, back = ports()
        config = root / 'profile.json'
        config.write_text(json.dumps({'port': front, 'l2_backend_port': back, 'args': [], 'host': '127.0.0.1',
                                      'slot_save_path': str(root / 'archives'), 'drafting': 'dflash2',
                                      'startup_benchmark': {'output_dir': str(root / 'results')},
                                      **({'vision': {'expert_lease': True}} if vision else {})}))
        output = (root / 'console.log').open('w')
        self.addCleanup(output.close)
        p = subprocess.Popen([sys.executable, str(root / 'l2/managed_server.py'), '--profile', str(config),
                              '--ready-file', str(root / 'ready.json'), '--shutdown-timeout', '1'],
                             stdout=output, stderr=subprocess.STDOUT)
        self.addCleanup(self.stop, p)
        wait_until(lambda: (root / 'ready.json').exists(), p)
        (root / 'release').touch()
        return front, p

    def stop(self, p):
        if p.poll() is None:
            p.send_signal(signal.SIGINT)
            try: p.wait(timeout=6)
            except subprocess.TimeoutExpired:
                p.kill(); p.wait()

    def assert_closed(self, port):
        with self.assertRaises(ConnectionRefusedError):
            socket.create_connection(('127.0.0.1', port), timeout=.5).close()

    def test_normal_startup_serves_without_model_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'results').mkdir()
            # Old pending requests cannot cause automatic comparisons anymore.
            (root/'results'/'spec-shape-lookup.request.json').write_text('{}')
            port,p=self.launch(root,False)
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as response:
                self.assertEqual(response.status,200)
            self.assertFalse((root/'bench-entered').exists())
            self.assertFalse((root/'results'/'latest.json').exists())
            self.assertFalse((root/'archives'/'index.json').exists())
            self.stop(p)

    def test_ready_record_reports_backend_image_support(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            port, p = self.launch(root, False, vision=True)
            record = json.loads((root / 'ready.json').read_text())
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2) as response:
                self.assertTrue(json.load(response)['images'])
            self.assertEqual(record['vision'], {'enabled': True, 'on_demand': True})
            self.assertIn('视觉：已启用，首张图片时加载', (root / 'console.log').read_text())
            self.stop(p)

    def test_shutdown_drains_completed_stream_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            port, p = self.launch(root, False, delayed_save=True)
            wait_until(lambda: (root / 'ready.json').exists(), p)
            errors = []
            def request():
                try:
                    body = {'messages': [{'role': 'user', 'content': 'hi'}],
                            'stream': True, 'temperature': 0, 'seed': 123}
                    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',
                        data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
                    with urllib.request.urlopen(req, timeout=5) as response:
                        for line in response:
                            if line.strip() == b'data: [DONE]': break
                except Exception as exc:
                    errors.append(exc)
            t = threading.Thread(target=request)
            t.start()
            wait_until(lambda: (root / 'save-entered').exists(), p)
            p.send_signal(signal.SIGINT)
            time.sleep(.2)
            self.assertIsNone(p.poll(), 'supervisor exited while a completed stream was being archived')
            (root / 'save-release').touch()
            self.assertEqual(p.wait(timeout=6), 0)
            t.join(timeout=5)
            self.assertFalse(t.is_alive())
            self.assertEqual(errors, [])
            self.assertTrue((root / 'save-done').exists())

    def test_bad_model_stream_does_not_turn_startup_into_benchmark(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);port,p=self.launch(root,True)
            self.assertFalse((root/'bench-entered').exists())
            self.assertIsNone(p.poll())
            self.assertFalse((root/'results'/'latest.json').exists())
            self.stop(p)

    def test_first_baseline_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench.save_report(root, {'status': 'passed', 'drafting': 'dflash2', 'run': 'first'})
            bench.save_report(root, {'status': 'passed', 'drafting': 'dflash2', 'run': 'second'})
            self.assertEqual(json.loads((root / 'baseline-dflash2.json').read_text())['run'], 'first')
            self.assertEqual(json.loads((root / 'latest.json').read_text())['run'], 'second')

if __name__ == '__main__':
    unittest.main()
