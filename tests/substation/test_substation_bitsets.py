
import pytest

from pandapower_env.substation.substation_bitsets import fully_connected_hexset, hexset_to_closed_switch_list


def test_fully_connected_bitset() -> None:
    fully_connected_8elements = "0x11111111"
    assert fully_connected_hexset(8) == fully_connected_8elements

def test_bitset_to_open_switch_list() -> None:
    # Asking about busbar 0:
    assert hexset_to_closed_switch_list("0x11010", busbar=0, nbits=5, nbusbars=2) == [False, False, True, False, True]
    # Asking about busbar 1:
    assert hexset_to_closed_switch_list("0x11010", busbar=1, nbits=5, nbusbars=2) == [True, True, False, True, False]

    with pytest.raises(RuntimeError):
        hexset_to_closed_switch_list("0x11010", busbar=1, nbits=6, nbusbars=2) # Leading bit is missing:
    with pytest.raises(RuntimeError):
        hexset_to_closed_switch_list("0x11010", busbar=1, nbits=4, nbusbars=2)  # hexset is larger than number of bits.
    with pytest.raises(RuntimeError):
        hexset_to_closed_switch_list("0x12010", busbar=1, nbits=5, nbusbars=2)  # digit used greater than nbusbars
