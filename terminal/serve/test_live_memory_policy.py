"""Live Memory (PR4 Phase 4): the server's automatic policy - pure decisions, no engine.

Pressure detection with hysteresis, the debounce, the cooldown (and its REFUSED backoff), the bounded
steps, and that a grow-back only ever undoes one of this policy's own shrinks."""
import unittest

from serve.live_memory_policy import CapacitySample, MemoryPolicy, MemoryRequest, PolicyConfig, parse_ack

NOW = 1000.0


def sample(ram_available, ram_resident=8192, vram_reserve=1536, at=NOW):
    return CapacitySample(ram_resident_mib=ram_resident, ram_available_mib=ram_available,
                          vram_committed_mib=3584, vram_free_mib=6144, vram_reserve_mib=vram_reserve,
                          observed_monotonic=at)


class Parsing(unittest.TestCase):
    def test_capacity_line(self):
        got = CapacitySample.parse("CAPACITY ram_resident_mib=1024 ram_available_mib=8192 "
                                   "vram_committed_mib=3584 vram_free_mib=6144 vram_reserve_mib=1536", now=NOW)
        self.assertEqual(got, sample(8192, 1024))

    def test_incomplete_and_foreign_lines(self):
        self.assertIsNone(CapacitySample.parse("VRAM reserve_mib=1"))
        self.assertIsNone(CapacitySample.parse("CAPACITY ram_available_mib=1"))
        self.assertIsNone(CapacitySample.parse(""))

    def test_ack_line(self):
        self.assertEqual(parse_ack("ACK 7 OK ram_resident_mib=4096 vram_committed_mib=3584"),
                         (7, "OK", {"ram_resident_mib": 4096, "vram_committed_mib": 3584}, ""))
        self.assertEqual(parse_ack("ACK 9 PARTIAL note=the_RAM_release_stopped_early"),
                         (9, "PARTIAL", {}, "the_RAM_release_stopped_early"))
        self.assertEqual(parse_ack("ACK 3 REFUSED")[1], "REFUSED")
        self.assertIsNone(parse_ack("VRAM reserve_mib=1"))
        self.assertIsNone(parse_ack("ACK x OK"))


class Pressure(unittest.TestCase):
    def setUp(self):
        self.cfg = PolicyConfig(enabled=True, low_watermark_mib=1024, high_watermark_mib=3072,
                                debounce_n=2, cooldown_s=120.0, max_shrink_mib=4096, max_grow_mib=512)

    def test_off_is_never_anything(self):
        p = MemoryPolicy(PolicyConfig(enabled=False))
        self.assertIsNone(p.observe(sample(0)))

    def test_stale_report_is_not_a_reading(self):
        p = MemoryPolicy(self.cfg)
        self.assertIsNone(p.observe(sample(0, at=NOW - 91.0), now=NOW))
        self.assertEqual(p._direction, "")                            # a stale report does not move the detector
        self.assertIsNone(p.observe(sample(0, at=NOW - 89.0), now=NOW))
        self.assertEqual(p._direction, "shrink")                      # a fresh one does

    def test_debounce_then_bounded_shrink(self):
        p = MemoryPolicy(self.cfg)
        self.assertIsNone(p.observe(sample(512), now=NOW))            # one crossing is not a direction
        request = p.observe(sample(512), now=NOW + 1)
        self.assertIsInstance(request, MemoryRequest)
        self.assertEqual(request.resident_mib, 8192 - 4096)           # the bounded step, not the whole tier
        self.assertEqual(request.vram_reserve_mib, 1536)              # the reserve rides unchanged: no growth
        self.assertEqual(request.reason, "ram_pressure")

    def test_a_flip_resets_the_streak(self):
        p = MemoryPolicy(self.cfg)
        self.assertIsNone(p.observe(sample(512), now=NOW))
        self.assertIsNone(p.observe(sample(4096), now=NOW + 1))       # relief: the pressure streak resets
        self.assertIsNone(p.observe(sample(512), now=NOW + 2))
        request = p.observe(sample(512), now=NOW + 3)                 # two fresh crossings again
        self.assertEqual(request.resident_mib, 4096)

    def test_cooldown_holds_requests_back(self):
        p = MemoryPolicy(self.cfg)
        p.observe(sample(512, at=NOW), now=NOW)
        first = p.observe(sample(512, at=NOW + 1), now=NOW + 1)
        self.assertIsNotNone(first)
        p.observe(sample(512, at=NOW + 60), now=NOW + 60)             # still pressured, still debounced
        self.assertIsNone(p.observe(sample(512, at=NOW + 61), now=NOW + 61))   # ... but inside the cooldown
        second = p.observe(sample(512, at=NOW + 121), now=NOW + 121)
        self.assertEqual(second.resident_mib, 4096)   # the sample's level (8192) less the bounded step
        self.assertEqual(first.resident_mib, 4096)

    def test_the_empty_tier_is_not_asked_lower(self):
        p = MemoryPolicy(self.cfg)
        p.observe(sample(512, ram_resident=0), now=NOW)
        self.assertIsNone(p.observe(sample(512, ram_resident=0), now=NOW + 1))


class GrowBack(unittest.TestCase):
    def setUp(self):
        self.cfg = PolicyConfig(enabled=True, low_watermark_mib=1024, high_watermark_mib=3072,
                                debounce_n=1, cooldown_s=0.0, max_shrink_mib=4096, max_grow_mib=512)

    def test_no_grow_back_without_a_shrink_of_its_own(self):
        p = MemoryPolicy(self.cfg)
        self.assertIsNone(p.observe(sample(8192), now=NOW))           # comfortable RAM, no episode: nothing
        self.assertIsNone(p._episode)

    def test_grow_back_is_bounded_by_what_the_shrink_took(self):
        p = MemoryPolicy(self.cfg)
        p.observe(sample(512), now=NOW)                               # the shrink (8192 -> 4096)
        grow = p.observe(sample(4096, ram_resident=4096), now=NOW + 1)
        self.assertEqual(grow.resident_mib, 4096 + 512)               # the smaller step, capped by the episode
        grow2 = p.observe(sample(4096, ram_resident=4608), now=NOW + 2)
        self.assertEqual(grow2.resident_mib, 4096 + 512 + 512)
        grow3 = p.observe(sample(4096, ram_resident=8192), now=NOW + 3)
        self.assertIsNone(grow3)                                      # back at the level the pressure took
        self.assertIsNone(p._episode)

    def test_a_tiny_shrink_is_not_worth_undoing(self):
        p = MemoryPolicy(self.cfg, _cooldown_s=0.0)
        p.config = PolicyConfig(enabled=True, low_watermark_mib=1024, high_watermark_mib=3072, debounce_n=1,
                                cooldown_s=0.0, max_shrink_mib=4096, max_grow_mib=512, min_pressure_gain_mib=256)
        p.observe(sample(1000, ram_resident=128), now=NOW)            # freed 0 MiB (the tier was empty)
        p.observe(sample(1024, ram_resident=128), now=NOW + 1)
        self.assertIsNone(p.observe(sample(4096, ram_resident=128), now=NOW + 2))


class AckFeedback(unittest.TestCase):
    def test_refused_doubles_the_cooldown(self):
        cfg = PolicyConfig(enabled=True, cooldown_s=120.0, refuse_backoff=2.0)
        p = MemoryPolicy(cfg)
        self.assertEqual(p._cooldown_s, 120.0)
        p.note_ack("REFUSED", now=NOW)
        self.assertEqual(p._cooldown_s, 240.0)
        p.note_ack("REFUSED", now=NOW)
        self.assertEqual(p._cooldown_s, 480.0)

    def test_backoff_is_capped(self):
        cfg = PolicyConfig(enabled=True, cooldown_s=120.0, refuse_backoff=2.0)
        p = MemoryPolicy(cfg)
        for _ in range(10):
            p.note_ack("REFUSED", now=NOW)
        self.assertEqual(p._cooldown_s, 1200.0)

    def test_ok_resets_the_backoff_and_ends_the_episode(self):
        cfg = PolicyConfig(enabled=True, cooldown_s=120.0)
        p = MemoryPolicy(cfg)
        p._direction = "grow"
        p._episode = object()
        p.note_ack("OK", now=NOW)
        self.assertEqual(p._cooldown_s, 120.0)
        self.assertIsNone(p._episode)


if __name__ == "__main__":
    unittest.main()
