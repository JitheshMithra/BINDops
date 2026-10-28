"""Keep this process's GPU well under a utilization ceiling (default 68%).

Two uses:
  * wait_for_headroom(gpu) -- block before a heavy op until util < ceiling
  * Throttle(gpu)          -- call .tick() once per training step; it duty-cycles
                             the loop with short sleeps so measured util stays low.

Rationale: the box's other GPU may still be running MD; the user's hard rule is that
the GPU WE touch never sustains >70%. Pausing MD (mdctl.sh) frees the SMs; this module
stops our own job from then pinning them at 100%.
"""

# repo root on sys.path so `python bindops/<pkg>/<file>.py` works directly
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os, subprocess, time

CEIL = float(os.environ.get("GPU_UTIL_CEIL", "68"))


def _util(gpu):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits",
             "-i", str(gpu)], text=True, timeout=10)
        return float(out.strip().splitlines()[0])
    except Exception:
        return 0.0


def wait_for_headroom(gpu=0, ceil=CEIL, poll=2.0, quiet=False):
    n = 0
    while _util(gpu) >= ceil:
        if not quiet and n % 15 == 0:
            print(f"[gpu_budget] gpu{gpu} util >= {ceil}%, waiting...", flush=True)
        time.sleep(poll); n += 1


class Throttle:
    """Target a duty cycle so sustained util stays ~ (duty * peak). duty=0.55 -> ~55-65%."""
    def __init__(self, gpu=0, duty=0.55, ceil=CEIL, check_every=5):
        self.gpu, self.duty, self.ceil, self.check_every = gpu, duty, ceil, check_every
        self._t = time.time(); self._i = 0

    def tick(self):
        self._i += 1
        dt = time.time() - self._t
        # base duty-cycle sleep
        if self.duty < 1.0:
            time.sleep(dt * (1.0 - self.duty) / self.duty)
        # hard backstop: if util still creeping over ceiling, sleep until it drops
        if self._i % self.check_every == 0:
            while _util(self.gpu) >= self.ceil:
                time.sleep(1.0)
        self._t = time.time()


if __name__ == "__main__":
    import sys
    g = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    print(f"gpu{g} util = {_util(g)}%  (ceil {CEIL}%)")
