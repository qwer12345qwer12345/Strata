"""Live Memory (PR4 Phase 4): the server's conservative, OPT-IN capacity policy.

The ENGINE remains the final admission authority: this module only decides WHEN to send a
`MEMORY <id> <resident-mib> <vram-reserve-mib>` request and what target to ask for.  The engine answers
every request with one terminal ACK (OK / PARTIAL / REFUSED) reporting the ACTUAL capacity, and refuses
what its own floors refuse - a PARTIAL or REFUSED here is a normal outcome, not an error to retry in a
loop (a REFUSED doubles the cooldown instead).

What the policy does (PORT_PLAN Phase C, all of it opt-in):
  - fresh samples: a CAPACITY report older than `max_age_s` is stale and never acted on;
  - pressure detection with hysteresis: pressure is RAM available below `low_watermark_mib`, relief only
    above `high_watermark_mib` (default low + 2048) - one crossing never flips the direction twice;
  - debounce: a direction must hold for `debounce_n` consecutive fresh samples before a request goes out;
  - cooldown: at most one request every `cooldown_s`, doubled (up to 10x) by every REFUSED;
  - bounded steps: a shrink asks for at most `max_shrink_mib` at once, a grow-back at most `max_grow_mib`
    (growth is deliberately slower than relief);
  - growth is never a side effect: the VRAM reserve rides every request at its CURRENT reported value -
    the policy relieves RAM pressure and grows the RAM tier back after a shrink of its own, it never
    touches the GPU reserve another owner (Vision, the operator's POST /v1/vram) set.

A pressure episode: the resident level when the first shrink went out is remembered, and the grow-backs
aim at it - never above it, never when no shrink of this policy happened.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class PolicyConfig:
    enabled: bool = False
    interval_s: float = 30.0            # how often the policy thread looks at the latest CAPACITY report
    low_watermark_mib: int = 1024       # pressure: RAM available below this
    high_watermark_mib: int = 3072      # relief: RAM available above this (a fresh episode's grow-back bar)
    debounce_n: int = 2                 # consecutive fresh samples in one direction before acting
    cooldown_s: float = 120.0           # at most one request in this window
    max_shrink_mib: int = 4096          # one shrink's bounded step
    max_grow_mib: int = 512             # one grow-back's (smaller) step
    max_age_s: float = 90.0             # a CAPACITY report older than this is stale
    refuse_backoff: float = 2.0         # the cooldown multiplier per REFUSED (capped at 10x)
    min_pressure_gain_mib: int = 256    # grow-backs stop when the first shrink freed less than this


@dataclass
class CapacitySample:
    ram_resident_mib: int = 0
    ram_available_mib: int = 0
    vram_committed_mib: int = 0
    vram_free_mib: int = 0
    vram_reserve_mib: int = 0
    observed_monotonic: float = 0.0

    @staticmethod
    def parse(line: str, now: float | None = None) -> "CapacitySample | None":
        """The engine's `CAPACITY k=v ...` line (the same key=value shape as the VRAM answer)."""
        parts = line.split()
        if not parts or parts[0] != "CAPACITY":
            return None
        out = CapacitySample(observed_monotonic=time.monotonic() if now is None else now)
        seen = {}
        for kv in parts[1:]:
            key, _, value = kv.partition("=")
            if value.lstrip("-").isdigit():
                seen[key] = int(value)
        out.ram_resident_mib = seen.get("ram_resident_mib", 0)
        out.ram_available_mib = seen.get("ram_available_mib", 0)
        out.vram_committed_mib = seen.get("vram_committed_mib", 0)
        out.vram_free_mib = seen.get("vram_free_mib", 0)
        out.vram_reserve_mib = seen.get("vram_reserve_mib", 0)
        if "ram_resident_mib" not in seen:      # not a complete report: treat as unusable
            return None
        return out


@dataclass
class MemoryRequest:
    resident_mib: int
    vram_reserve_mib: int
    reason: str


@dataclass
class _Episode:
    """One pressure episode: from this policy's first shrink until the RAM is back above the high bar."""
    resident_before_mib: int = 0        # what the RAM tier held when the first shrink went out
    shrunk_total_mib: int = 0           # what this policy's shrinks asked away (for a real gain check)


@dataclass
class MemoryPolicy:
    """The pure decision half: feed `observe` a fresh CapacitySample, send the MemoryRequest it returns
    (None: nothing to do now), and tell it the ACK with `note_ack`."""
    config: PolicyConfig
    _direction: str = ""                # "", "shrink", "grow": the streak's direction
    _streak: int = 0
    _last_request_at: float = -1.0e30   # -inf: no request yet
    _cooldown_s: float = field(default=0.0)
    _episode: _Episode | None = None

    def __post_init__(self) -> None:
        if self._cooldown_s == 0.0:
            self._cooldown_s = self.config.cooldown_s

    # ---- the decision ----
    def observe(self, sample: CapacitySample | None, now: float | None = None) -> MemoryRequest | None:
        cfg = self.config
        if not cfg.enabled or sample is None:
            return None
        now = time.monotonic() if now is None else now
        if now - sample.observed_monotonic > cfg.max_age_s:
            return None                                  # a stale report is not a reading
        direction = ""
        if sample.ram_available_mib < cfg.low_watermark_mib:
            direction = "shrink"
        elif sample.ram_available_mib >= cfg.high_watermark_mib and self._episode is not None:
            direction = "grow"                           # only this policy's own shrinks are undone
        if direction != self._direction:
            self._direction, self._streak = direction, 1 if direction else 0
        elif direction:
            self._streak += 1
        if not direction or self._streak < cfg.debounce_n:
            return None
        if now - self._last_request_at < self._cooldown_s:
            return None

        if direction == "shrink":
            step = min(cfg.max_shrink_mib, max(sample.ram_resident_mib, 0))
            target = max(sample.ram_resident_mib - step, 0)
            if target >= sample.ram_resident_mib:
                return None                              # the tier is already empty; the engine must find RAM elsewhere
            if self._episode is None:
                self._episode = _Episode(resident_before_mib=sample.ram_resident_mib)
            self._episode.shrunk_total_mib += sample.ram_resident_mib - target
            request = MemoryRequest(resident_mib=target, vram_reserve_mib=sample.vram_reserve_mib,
                                    reason="ram_pressure")
        else:
            episode = self._episode
            assert episode is not None                   # direction "grow" requires one
            if episode.shrunk_total_mib < cfg.min_pressure_gain_mib:
                self._episode = None                     # the shrink freed too little to be worth undoing
                return None
            step = min(cfg.max_grow_mib, episode.resident_before_mib - sample.ram_resident_mib)
            if step <= 0:
                self._episode = None                     # already back at the level the pressure took it from
                return None
            target = sample.ram_resident_mib + step
            request = MemoryRequest(resident_mib=target, vram_reserve_mib=sample.vram_reserve_mib,
                                    reason="ram_relief")

        self._last_request_at = now
        self._direction, self._streak = "", 0            # one request per direction decision
        return request

    # ---- the ACK feeds back ----
    def note_ack(self, state: str, now: float | None = None) -> None:
        """`state` is the ACK's second token: OK, PARTIAL or REFUSED."""
        if state == "REFUSED":
            now = time.monotonic() if now is None else now
            self._cooldown_s = min(self._cooldown_s * self.config.refuse_backoff,
                                   self.config.cooldown_s * 10.0)
            self._last_request_at = now                  # the backoff counts from the refusal
        else:
            self._cooldown_s = self.config.cooldown_s    # a landed request resets the backoff
        if state in ("OK", "PARTIAL") and self._direction == "grow":
            self._episode = None


def parse_ack(line: str) -> tuple[int, str, dict[str, int], str] | None:
    """The engine's `ACK <id> <OK|PARTIAL|REFUSED> k=v ...` line -> (id, state, numbers, note); None
    otherwise.  The note is the engine's limitation as one note= token ('_' for spaces, else '')."""
    parts = line.split()
    if len(parts) < 3 or parts[0] != "ACK" or not parts[1].isdigit():
        return None
    numbers: dict[str, int] = {}
    note = ""
    for kv in parts[3:]:
        key, _, value = kv.partition("=")
        if key == "note":
            note = value
        elif value.lstrip("-").isdigit():
            numbers[key] = int(value)
    return int(parts[1]), parts[2], numbers, note
