"""Balanced randomization: Williams squares, AB/BA periods, and reproducible seeds.

Arms must be interleaved, not run one whole arm then the next, so a drift in the engine over time
cannot masquerade as an arm effect. A Williams square gives each arm order the balance property
that every arm immediately follows every other arm equally often, which cancels first-order
carryover across a TraceBlock's replays. Paired tasks use AB/BA period balancing for the same
reason at the pair level.

Every seed is derived from a stable logical id (protocol + node coordinates), never from a clock
or a call counter, so the same logical unit draws the same schedule on a re-run -- a shared-path
seed the plan (§5.4) requires for reproducibility.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

__all__ = ["derive_seed", "williams_square", "seeded_permutation", "ab_ba_order",
           "arm_order_for_block"]


def derive_seed(*parts: str) -> int:
    """A deterministic 32-bit seed from stable string coordinates."""
    h = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big")


def _williams_first_row(n: int) -> list[int]:
    # 0, 1, n-1, 2, n-2, 3, n-3, ...
    row = [0]
    lo, hi = 1, n - 1
    take_lo = True
    while len(row) < n:
        if take_lo:
            row.append(lo)
            lo += 1
        else:
            row.append(hi)
            hi -= 1
        take_lo = not take_lo
    return row


def williams_square(n: int) -> list[list[int]]:
    """A Williams design over ``n`` treatments (0..n-1).

    Each row is a treatment order; across the returned rows every ordered adjacent pair is balanced.
    For even ``n`` a single n×n square suffices; for odd ``n`` the square and its row-reverses
    together achieve balance, so 2n rows are returned.
    """
    if n < 2:
        raise ValueError("need at least 2 treatments")
    first = _williams_first_row(n)
    square = [[(first[j] + i) % n for j in range(n)] for i in range(n)]
    if n % 2 == 1:
        square += [row[::-1] for row in square]
    return square


def seeded_permutation(items: Sequence, seed: int) -> list:
    """A deterministic permutation of ``items`` from ``seed`` (Fisher-Yates with a hashed stream,
    so it needs no global RNG state and reproduces exactly)."""
    result = list(items)
    n = len(result)
    for i in range(n - 1, 0, -1):
        # draw an index in [0, i] deterministically from (seed, i)
        h = hashlib.sha256(f"{seed}:{i}".encode()).digest()
        j = int.from_bytes(h[:8], "big") % (i + 1)
        result[i], result[j] = result[j], result[i]
    return result


def ab_ba_order(pair: tuple, *, replicate: int, seed: int) -> tuple:
    """AB or BA for one paired replicate, alternating by replicate parity with a seeded phase, so
    across replicates each element leads equally often."""
    a, b = pair
    phase = seed % 2
    return (a, b) if (replicate + phase) % 2 == 0 else (b, a)


def arm_order_for_block(arms: Sequence[str], *, block_index: int, seed: int) -> list[str]:
    """The interleaved arm order for one TraceBlock, taken from the Williams square row selected
    by the block index (rotated by the seed), mapping treatment indices back to arm labels."""
    n = len(arms)
    square = williams_square(n)
    row = square[(block_index + seed) % len(square)]
    return [arms[i] for i in row]
