"""Live Memory (PR4): the MEMORY protocol's server half - the engine command's ACK parsing, the CAPACITY
telemetry's routing away from the token stream, the /v1/memory route, and Vision's shrink riding the same
serialized owner (Phase D)."""
import io
import json
import queue
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, EngineDied, Service, StrataEngine, serve
from serve.test_lifecycle import ResidentEngine


class EngineCommand(unittest.TestCase):
    def engine(self, acks):
        e = StrataEngine("missing-executable", [], lazy=True)
        e.proc = mock.Mock()
        e.proc.poll.return_value = None
        e.proc.stdin = io.StringIO()
        e.ended = False
        e.lines = queue.Queue()
        e.ack_q = queue.Queue()
        e._memory_mu = threading.Lock()   # (the lazy init never ran the post-READY block that makes these)
        e._memory_seq = 0
        for a in acks:
            e.ack_q.put(a)
        return e

    def test_the_line_and_the_ack(self):
        e = self.engine(["ACK 1 OK ram_resident_mib=4096 vram_committed_mib=3584 vram_reserve_mib=1536\n",
                         "ACK 2 PARTIAL ram_resident_mib=2048 note=the_RAM_release_stopped_early\n"])
        out = e.request_memory(4096, 1536)
        self.assertEqual(e.proc.stdin.getvalue(), "MEMORY 1 4096 1536\n")
        self.assertEqual(out["state"], "OK")
        self.assertEqual(out["ram_resident_mib"], 4096)
        self.assertEqual(e.info["vram_committed_mib"], 3584)
        # the ids advance; a PARTIAL rides its note back
        out = e.request_memory(2048, 1536)
        self.assertEqual(e.proc.stdin.getvalue(), "MEMORY 1 4096 1536\nMEMORY 2 2048 1536\n")
        self.assertEqual(out["state"], "PARTIAL")
        self.assertEqual(out["ram_resident_mib"], 2048)

    def test_refused_and_died_and_partial_note(self):
        e = self.engine(["ACK 1 REFUSED ram_resident_mib=0 note=a_capacity_change_is_already_waiting\n"])
        with self.assertRaisesRegex(ValueError, "already waiting"):
            e.request_memory(0, 0)
        e = self.engine([])
        with self.assertRaises(EngineDied):
            e.request_memory(0, 0, timeout=0.2)

    def test_the_serialized_owner(self):
        # request_memory holds _memory_mu: a second caller waits instead of interleaving its line
        e = self.engine(["ACK 1 OK ram_resident_mib=1\n"])
        order = []

        def first():
            with e._memory_mu:
                order.append("policy-holder")
                e.request_memory(1, 1)

        t = threading.Thread(target=first)
        t.start()
        e._memory_mu.acquire(timeout=5)
        order.append("vision-after")
        t.join(timeout=5)
        e._memory_mu.release()
        self.assertEqual(order, ["policy-holder", "vision-after"])


class PumpRouting(unittest.TestCase):
    def pump(self, lines):
        e = StrataEngine("missing-executable", [], lazy=True)
        e.proc = mock.Mock()
        e.proc.poll.return_value = None
        e.proc.stdout = iter(lines)
        e.ended = False
        e.lines = queue.Queue()
        e.ack_q = queue.Queue()
        e.slot_q = []
        e.slot_busy = []        # (the lazy engine's batch setup never ran: the pump only reads it)
        e.log = None
        e._pump()
        return e, e.lines

    def test_capacity_never_reaches_a_request(self):
        e, lines = self.pump(["CAPACITY ram_resident_mib=1024 ram_available_mib=8192 vram_committed_mib=3584 "
                              "vram_free_mib=6144 vram_reserve_mib=1536\n",
                              "CAPACITY ram_resident_mib=512 ram_available_mib=8448 vram_committed_mib=3584 "
                              "vram_free_mib=6400 vram_reserve_mib=1536\n"])
        self.assertEqual(lines.get_nowait(), None)      # EOF: nothing else was queued
        self.assertEqual(e.terminal_capacity["ram_resident_mib"], 512)
        self.assertEqual(e.terminal_capacity["ram_available_mib"], 8448)
        self.assertLess(e.terminal_capacity["observed_monotonic"], 1e6)   # stamped

    def test_ack_waits_for_its_caller(self):
        e, lines = self.pump(["ACK 4 OK ram_resident_mib=8\n"])
        self.assertEqual(lines.get_nowait(), None)
        self.assertEqual(e.ack_q.get_nowait(), "ACK 4 OK ram_resident_mib=8")


class MemoryEngine(ResidentEngine):
    """The resident test engine with the MEMORY protocol (PR4): an ACK for whatever was asked."""
    def __init__(self, tok):
        super().__init__(tok)
        self.calls = []
        self.info["live_memory"] = 1
        self.terminal_capacity = {"ram_resident_mib": 2048, "ram_available_mib": 8192,
                                  "vram_committed_mib": 3584, "vram_free_mib": 6144,
                                  "vram_reserve_mib": 1536, "observed_monotonic": 0.0}

    def has_live_memory(self):
        return True

    def latest_capacity(self):
        return dict(self.terminal_capacity)

    def request_memory(self, resident_mib, vram_reserve_mib, timeout=300.0):
        self.calls.append((resident_mib, vram_reserve_mib))
        return {"state": "OK", "id": len(self.calls), "ram_resident_mib": resident_mib,
                "vram_committed_mib": 3584, "vram_reserve_mib": vram_reserve_mib}

    def vram(self, reserve_mib, timeout=120.0):
        raise AssertionError("a live-memory engine's Vision reserve must ride the MEMORY owner, not VRAM")


class VisionOwner(unittest.TestCase):
    """Phase D: the image job's shrink and give-back go through the same owner the policy uses."""

    def setUp(self):
        tok = ByteTokenizer()
        self.engine = MemoryEngine(tok)
        self.svc = Service(self.engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def request(self, path, body=None, headers=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            response = urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.status, json.loads(response.read().decode())

    def test_make_room_and_give_back_use_the_owner(self):
        from serve import images as images_api
        gen = mock.Mock()
        restore = images_api._make_room(self.svc, gen, "shrink", 2600)
        self.assertEqual(restore[0], "shrunk-memory")
        self.assertEqual(restore[1], 1536)                  # the reserve before the job, for the give-back
        self.assertEqual(self.engine.calls, [(2048, 2600)])  # the RAM tier rides; the reserve carries the need
        images_api._give_back(self.svc, gen, "shrink", restore)
        self.assertEqual(self.engine.calls[-1], (2048, 1536))

    def test_status_route_reports_the_owner(self):
        code, body = self.request("/v1/memory")
        self.assertEqual(code, 200, body)
        self.assertTrue(body["live_memory"])
        self.assertEqual(body["capacity"]["ram_resident_mib"], 2048)

    def test_memory_route(self):
        self.engine.loaded = True
        code, body = self.request("/v1/memory", {"resident_mib": 4096, "vram_reserve_mib": 1536})
        self.assertEqual(code, 200, body)
        self.assertEqual(body["status"], "ok")
        self.assertEqual(self.engine.calls, [(4096, 1536)])
        code, body = self.request("/v1/memory", {"resident_mib": -1})
        self.assertEqual(code, 400)
        code, body = self.request("/v1/memory", {"resident_mib": True})
        self.assertEqual(code, 400)


if __name__ == "__main__":
    unittest.main()
