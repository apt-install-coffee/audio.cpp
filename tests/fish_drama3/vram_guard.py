#!/usr/bin/env python3
"""VRAM gate and watchdog for Drama 3 benchmark runs.

Drama 3 does not fit in 16 GB, so a run near the ceiling can wedge the GPU
ring: on gfx1101 that kills the desktop compositor and has twice needed a
reboot. This module keeps a benchmark away from the edge.

  * `preflight` refuses to start when the card is already busy.
  * `Watchdog` samples the card every 200 ms and terminates the model process
    before usage reaches the wedge zone.

Both read the amdgpu sysfs counters, so they cost nothing, need no HIP context
and work before the backend is initialised. The card is the one reporting the
largest VRAM total unless `--card` names it, which on a hybrid system is the
discrete GPU the benchmark wants.

Examples:

    # what the card looks like now
    python3 tests/fish_drama3/vram_guard.py

    # refuse to run below 13.5 GiB free
    python3 tests/fish_drama3/vram_guard.py --min-free-mb 13500

    # wrap any command: gate, then kill it at 15.4 GiB
    python3 tests/fish_drama3/vram_guard.py --min-free-mb 13500 \
        --kill-used-mb 15400 -- python3 tests/fish_drama3/bench_drama3_hip.py ...
"""

from __future__ import annotations

import argparse
import glob
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

DRM_ROOT = Path("/sys/class/drm")
MIB = 1024.0 * 1024.0


@dataclass(frozen=True)
class VramState:
    card: str
    used_mb: int
    total_mb: int

    @property
    def free_mb(self) -> int:
        return self.total_mb - self.used_mb

    def __str__(self) -> str:
        return (f"{self.card}: {self.used_mb} MiB used, {self.free_mb} MiB free "
                f"of {self.total_mb} MiB")


def _device(card: str) -> Path:
    return DRM_ROOT / card / "device"


def _read_mb(device: Path, name: str) -> int:
    try:
        return int((device / name).read_text().strip()) // int(MIB)
    except (OSError, ValueError):
        return 0


def available_cards() -> list[str]:
    """DRM card names that expose amdgpu VRAM counters."""
    cards = []
    for path in sorted(glob.glob(str(DRM_ROOT / "card[0-9]*"))):
        card = Path(path).name
        if (_device(card) / "mem_info_vram_total").exists():
            cards.append(card)
    return cards


def read_vram(card: str | None = None) -> VramState:
    """Current usage of `card`, else of the card with the largest VRAM total."""
    names = available_cards()
    if not names:
        raise RuntimeError("no DRM card exposes mem_info_vram_* (not an amdgpu host?)")
    if card is not None:
        if card not in names:
            raise RuntimeError(f"card {card!r} not found; available: {', '.join(names)}")
        names = [card]
    states = [VramState(name, _read_mb(_device(name), "mem_info_vram_used"),
                        _read_mb(_device(name), "mem_info_vram_total"))
              for name in names]
    return max(states, key=lambda state: state.total_mb)


def preflight(min_free_mb: int, card: str | None = None) -> VramState:
    """Return the card state, or raise SystemExit when below `min_free_mb`."""
    state = read_vram(card)
    if min_free_mb <= 0:
        return state
    if state.free_mb < min_free_mb:
        raise SystemExit(
            f"[vram] refusing to start: {state} is below the "
            f"--min-free-mb gate of {min_free_mb} MiB; close GPU applications first")
    return state


class Watchdog:
    """Terminate attached processes when VRAM usage reaches `kill_used_mb`.

    The poller runs on its own thread and only reads sysfs, so it cannot slow a
    measurement. `stop()` returns the peak usage seen while it ran.
    """

    def __init__(self, kill_used_mb: int, card: str | None = None,
                 interval_s: float = 0.2) -> None:
        self.kill_used_mb = kill_used_mb
        self.card = card
        self.interval_s = interval_s
        self.peak_mb = 0
        self.killed = False
        self._procs: list[subprocess.Popen] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def attach(self, proc: subprocess.Popen) -> None:
        self._procs.append(proc)

    def start(self) -> "Watchdog":
        if self.kill_used_mb <= 0:
            return self
        self._thread = threading.Thread(target=self._run, name="vram-watchdog", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> "Watchdog":
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                used_mb = read_vram(self.card).used_mb
            except RuntimeError:
                return
            self.peak_mb = max(self.peak_mb, used_mb)
            if used_mb < self.kill_used_mb:
                continue
            for proc in self._procs:
                # SIGTERM first: an orderly exit releases the device without
                # leaving the ring half-submitted.
                if proc.poll() is None:
                    print(f"[vram] {used_mb} MiB >= {self.kill_used_mb} MiB, "
                          f"terminating pid {proc.pid}", file=sys.stderr)
                    proc.send_signal(signal.SIGTERM)
            self.killed = True
            time.sleep(1.0)
            for proc in self._procs:
                if proc.poll() is None:
                    proc.kill()
            return


def run_guarded(argv: list[str], min_free_mb: int, kill_used_mb: int,
                card: str | None = None) -> int:
    """Run `argv` between the gate and the watchdog; returns its exit code."""
    state = preflight(min_free_mb, card)
    print(f"[vram] start: {state}", file=sys.stderr)
    watchdog = Watchdog(kill_used_mb, card).start()
    proc = subprocess.Popen(argv)
    watchdog.attach(proc)
    try:
        code = proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        code = proc.wait()
    watchdog.stop()
    peak = f", peak {watchdog.peak_mb} MiB" if watchdog.peak_mb else ""
    print(f"[vram] exit {code}{peak}{', watchdog fired' if watchdog.killed else ''}",
          file=sys.stderr)
    return code


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-free-mb", type=int, default=0,
                        help="refuse to start below this much free VRAM (0 = no gate)")
    parser.add_argument("--kill-used-mb", type=int, default=0,
                        help="terminate the command at this much used VRAM (0 = no watchdog)")
    parser.add_argument("--card", default=None,
                        help="DRM card name, e.g. card1 (default: largest VRAM total)")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="command to run under the guard, after --")
    args = parser.parse_args(argv)

    if not args.command:
        try:
            state = (preflight(args.min_free_mb, args.card) if args.min_free_mb > 0
                     else read_vram(args.card))
        except SystemExit as refusal:
            print(refusal, file=sys.stderr)
            return 1
        except RuntimeError as error:
            print(error, file=sys.stderr)
            return 1
        print(state)
        return 0

    command = args.command[1:] if args.command[0] == "--" else args.command
    if not command:
        parser.error("no command after --")
    return run_guarded(command, args.min_free_mb, args.kill_used_mb, args.card)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
