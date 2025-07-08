import hashlib
import random
import numpy as np
import torch
from typing import Optional


# ---------- helper functions ----------
def _stable_int_from_str(s: str, mod: int = 10_000) -> int:
    """Return a stable int (0–mod) from a string via MD5; same result every run."""
    h = hashlib.md5(s.encode()).hexdigest()[:8]
    return int(h, 16) % mod


def _set_all_rng(seed: int):
    """Seed NumPy, Python‐random and PyTorch (CPU + CUDA if present)."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _get_torch_cuda_state():
    """Grab full CUDA RNG state list (None if CUDA unavailable)."""
    return torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None


def _set_torch_cuda_state(state):
    """Restore CUDA RNG state list (no-op if CUDA unavailable / None)."""
    if state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state)


# ---------- main class ----------
class SeedManager:
    """Handle reproducible RNG for multiple algorithms and training rounds."""

    def __init__(self, global_seed: int = 42):
        """Create manager with one global seed; empty per-algo tables."""
        self.global_seed = global_seed
        self.algorithm_seeds = {}      # algo → base seed
        self.saved_states = {}         # (algo, round) → RNG snapshot

    def set_global_seed(self):
        """Apply global seed (dataset split etc.)."""
        _set_all_rng(self.global_seed)
        print(f"[INFO] Global seed set -> {self.global_seed}")

    def register_algorithm(self, algo: str, seed: Optional[int] = None):
        """
        Assign a fixed base seed to an algorithm.
        If seed omitted, derive a stable one from name + global_seed.
        """
        if seed is None:
            seed = self.global_seed + _stable_int_from_str(algo)
        self.algorithm_seeds[algo] = seed
        print(f"[INFO] Registered {algo} base seed = {seed}")

    def prepare_round(self, algo: str, round_num: int, jump: int = 100):
        """
        Set per-round seed (base + round*jump), then save full RNG state.
        Return the seed used.
        """
        if algo not in self.algorithm_seeds:
            raise KeyError(f"{algo} not registered.")
        seed = self.algorithm_seeds[algo] + round_num * jump
        _set_all_rng(seed)
        self.saved_states[(algo, round_num)] = {
            "np": np.random.get_state(),
            "py": random.getstate(),
            "th": torch.random.get_rng_state(),
            "cu": _get_torch_cuda_state()
        }
        return seed

    def restore_round(self, algo: str, round_num: int):
        """
        Reload the exact RNG snapshot for <algo, round>; enables
        deterministic re-runs of a single algorithm in isolation.
        """
        key = (algo, round_num)
        if key not in self.saved_states:
            raise KeyError(f"No state saved for {algo} round {round_num}")
        st = self.saved_states[key]
        np.random.set_state(st["np"])
        random.setstate(st["py"])
        torch.random.set_rng_state(st["th"])
        _set_torch_cuda_state(st["cu"])


# ----------------- Demo -----------------
#
# def generate_list():
#     return [random.randint(0, 99) for _ in range(5)]
#
#
# if __name__ == "__main__":
#     sm = SeedManager(42)
#     sm.set_global_seed()
#
#     # register algorithms with fixed seeds
#     sm.register_algorithm("FedAvg", 1234)
#     sm.register_algorithm("FedProx", 56718)
#     sm.register_algorithm("FedOpt")        # auto-derive from global seed
#
#     print("\n=== full experiment 3 rounds ===")
#     for r in range(3):
#         print(f"\n[ROUND {r}]")
#         for algo in ("FedAvg", "FedProx", "FedOpt"):
#             seed = sm.prepare_round(algo, r)        # set seed and save state
#             print(f"{algo:7s} seed {seed:6d} ->", generate_list())
#
#     # ----------------- replay single algo rounds -----------------
#     print("\n=== replay FedProx rounds ===")
#     for r in range(3):
#         sm.restore_round("FedProx", r)              # restore state without re-setting
#         print(f"FedProx round {r} ->", generate_list())
#
#     print("\n=== replay FedOpt rounds ===")
#     for r in range(3):
#         sm.restore_round("FedOpt", r)
#         print(f"FedOpt  round {r} ->", generate_list())