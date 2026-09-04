import copy

import pandas as pd

from pandapower_env.substation.create_double_busbar_substation import (
    can_convert_to_n_busbar_substation,
    create_3bb_with_pst_substation,
    create_double_busbar_substation,
)
from pandapower_env.substation.detect_substations import (
    detect_substations,
)
from pandapower_env.substation.double_busbar_substation import (
    _element_switch_columns,
)


def test_detect_substations(test_grid) -> None:
    """
    Test the function to detect substations (including ones with PSTs).

    :param test_grid: test_grid fixture (see conftest.py)
    :type test_grid: pandapowerNet
    """
    net = copy.deepcopy(test_grid)

    bus_indices = net.bus.index[2:]
    create_3bb_with_pst_substation(net, 1)
    for bus in bus_indices:
        if not can_convert_to_n_busbar_substation(net, bus):
            continue
        create_double_busbar_substation(net, bus)

    df_final = detect_substations(net)

    # Comparing "nan" values in some list columns results in issues. Convert nan to []
    el_sw_cols = _element_switch_columns(df_final)
    df_final[el_sw_cols] = df_final[el_sw_cols].map(lambda x: x if isinstance(x, list) else [])

    # Comparing "nan" values in some list columns results in issues. Convert nan to []
    mbb_sub = net.multi_bb_substation
    el_sw_cols = _element_switch_columns(mbb_sub)
    mbb_sub[el_sw_cols] = mbb_sub[el_sw_cols].map(lambda x: x if isinstance(x, list) else [])

    assert (df_final.index == mbb_sub.index).all(), "Indices of recreated substation DataFrame are not the same."
    assert (df_final.columns == mbb_sub.columns).all(), "Columns of recreated substation DataFrame are not the same."
    assert pd.Series((df_final == mbb_sub).to_numpy().ravel()).dropna().all(), "Did not exactly recreate the mbb df."
