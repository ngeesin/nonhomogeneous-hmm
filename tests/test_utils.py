import numpy as np

from nhmm import state_durations


def test_state_durations_basic():
    out = state_durations([0, 0, 1, 1, 1, 0])
    assert list(out) == [1, 2, 1, 2, 3, 1]


def test_state_durations_single_and_constant():
    assert list(state_durations([5])) == [1]
    assert list(state_durations([2, 2, 2, 2])) == [1, 2, 3, 4]


def test_state_durations_accepts_arrays():
    states = np.array([1, 0, 0])
    out = state_durations(states)
    assert isinstance(out, np.ndarray)
    assert out.dtype.kind == "i"
    assert list(out) == [1, 1, 2]
