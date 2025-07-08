'''
📌 目标
你希望能够：

在 完整实验中（所有算法一起运行），保证每轮各算法的随机性不同但可复现。
在 单独运行某个算法（如 FedProx）时，保证它的随机性 和完整实验中的相同。
✅ 问题分析
目前的 SeedManager 方案：

让 FedAvg、FedProx、FedOpt 在每一轮都有不同的随机性，但各算法之间互不影响。
然而，如果你只运行 FedProx，round_seed = base_seed + round_num * 100 仍然会起作用，但它和完整实验中的 FedProx 可能不同，因为完整实验中所有算法按顺序执行，随机状态可能被前面算法的随机操作影响。
✅ 解决方案
我们需要确保 FedProx 在单独运行时，其随机状态与完整实验中一致。
关键点：

不能让 FedProx 的随机状态被 FedAvg、FedOpt 影响。
需要显式存储 每轮的随机状态，这样即使单独运行 FedProx 也能恢复和完整实验时的状态一致。


'''

import hashlib
import random
import numpy as np
import torch
from typing import Optional

# ----------------- 工具函数 -----------------
def _stable_int_from_str(s: str, mod: int = 10_000) -> int:
    """取字符串 md5 前 8 位转 int，保证跨进程稳定"""
    h = hashlib.md5(s.encode()).hexdigest()[:8]
    return int(h, 16) % mod


def _set_all_rng(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _get_torch_cuda_state():
    return torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None


def _set_torch_cuda_state(state):
    if state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state)


# ----------------- SeedManager -----------------
class SeedManager:
    """多算法、多轮次 RNG 管理器"""

    def __init__(self, global_seed: int = 42):
        self.global_seed = global_seed
        self.algorithm_seeds = {}          # algo -> base seed
        self.saved_states = {}             # (algo, round) -> RNG states

    # ---------- 全局 ----------
    def set_global_seed(self):
        _set_all_rng(self.global_seed)
        print(f"[INFO] Global seed set -> {self.global_seed}")

    # ---------- 注册 ----------
    def register_algorithm(self, algo: str, seed: Optional[int] = None):
        if seed is None:
            seed = self.global_seed + _stable_int_from_str(algo)
        self.algorithm_seeds[algo] = seed
        print(f"[INFO] Registered {algo} base seed = {seed}")

    # ---------- 设种子 & 立即保存 ----------
    def prepare_round(self, algo: str, round_num: int, jump: int = 100):
        if algo not in self.algorithm_seeds:
            raise KeyError(f"{algo} not registered.")
        seed = self.algorithm_seeds[algo] + round_num * jump
        _set_all_rng(seed)
        self.saved_states[(algo, round_num)] = {
            "np":  np.random.get_state(),
            "py":  random.getstate(),
            "th":  torch.random.get_rng_state(),
            "cu":  _get_torch_cuda_state()
        }
        return seed

    # ---------- 恢复 ----------
    def restore_round(self, algo: str, round_num: int):
        key = (algo, round_num)
        if key not in self.saved_states:
            raise KeyError(f"No state saved for {algo} round {round_num}")
        st = self.saved_states[key]
        np.random.set_state(st["np"])
        random.setstate(st["py"])
        torch.random.set_rng_state(st["th"])
        _set_torch_cuda_state(st["cu"])


# ----------------- Demo -----------------
def generate_list():
    return [random.randint(0, 99) for _ in range(5)]


if __name__ == "__main__":
    sm = SeedManager(42)
    sm.set_global_seed()

    # 注册算法
    sm.register_algorithm("FedAvg", 1234)
    sm.register_algorithm("FedProx", 56718)
    sm.register_algorithm("FedOpt")        # 自动派生

    print("\n=== 完整实验 3 轮 ===")
    for r in range(3):
        print(f"\n[ROUND {r}]")
        for algo in ("FedAvg", "FedProx", "FedOpt"):
            seed = sm.prepare_round(algo, r)        # 设种子并保存
            print(f"{algo:7s} seed {seed:6d} ->", generate_list())

    # ----------------- 单独复现实验 -----------------
    print("\n=== 单独复现 FedProx ===")
    for r in range(3):
        sm.restore_round("FedProx", r)              # 只恢复，不再重新 set
        print(f"FedProx round {r} ->", generate_list())

    print("\n=== 单独复现 FedOpt ===")
    for r in range(3):
        sm.restore_round("FedOpt", r)
        print(f"FedOpt  round {r} ->", generate_list())