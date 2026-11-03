"""`hdf5_rows.rank_of`, the one hash that picks the rank a row of integers meets at, and
`rank_order`, the sort that groups rows by rank.

    python3 -m pytest tests/test_routing.py

numpy only.
"""
import numpy as np

from femtools.hdf5_rows import rank_of


def test_rows_of_any_size_meet_at_one_rank_each_spread_evenly():
    """Rows of ids past 2^40 (a face's three vertex ids at h0.012 broke the join's check when
    it borrowed the slab identities' 40- and 21-bit packing, 2026-10-16): the same row to the
    same rank, every rank in range and used about equally, the columns' order mattering."""
    rng = np.random.default_rng(0)
    rows = rng.integers(1 << 40, 1 << 62, size=(64000, 3))
    dest = rank_of(rows, 64)
    assert dest.dtype == np.int64 and dest.min() >= 0 and dest.max() < 64
    assert np.array_equal(dest, rank_of(rows.copy(), 64))
    counts = np.bincount(dest, minlength=64)
    assert counts.min() > 0.85 * counts.mean() and counts.max() < 1.15 * counts.mean()
    assert not np.array_equal(dest, rank_of(rows[:, ::-1], 64))
    small = np.arange(3000).reshape(1000, 3)            # consecutive small ids spread too
    assert len(np.unique(rank_of(small, 7))) == 7
    assert rank_of(np.zeros((0, 3), np.int64), 4).shape == (0,)
    assert np.array_equal(rank_of(rows, 1), np.zeros(len(rows), np.int64))


def test_rank_order_is_the_stable_argsort():
    """`rank_order` sorts by 16-bit ranks where they fit: the permutation of the 64-bit
    stable argsort, at one rank, a few, a node's and the 16-bit edge, and past it."""
    from femtools.hdf5_rows import rank_order
    rng = np.random.default_rng(1)
    for size in (1, 3, 256, 32768, 40000):
        dest = rng.integers(0, size, 20000)
        assert np.array_equal(rank_order(dest, size), np.argsort(dest, kind="stable"))
    assert rank_order(np.zeros(0, np.int64), 4).shape == (0,)
