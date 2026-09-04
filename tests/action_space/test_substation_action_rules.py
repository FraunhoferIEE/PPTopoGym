from pandapower_env.action_space.substation_action_rules import (
    passes_islanded_elements_rule,
    passes_n_elements_rule,
    passes_two_bus_symmetry_rule,
)


def test_rules(test_grid_multi_bb_substations) -> None:
    """
    Test various functions in double_busbar_substation.py.

    :param test_grid_multi_bb_substations: net fixture with dbb_substation Dataframe (see conftest.py)
    :type test_grid_multi_bb_substations: pandapowerNet
    """
    net = test_grid_multi_bb_substations

    n_sym, n_elements, n_island = 0, 0, 0

    for _, sub in net.multi_bb_substation.iterrows():
        n_bits = len(sub.connected_elements)

        for state in range(pow(2, n_bits)):
            state_str = bin(state)[2:].zfill(n_bits)
            n_sym += passes_two_bus_symmetry_rule(state_str)
            pass_test = passes_two_bus_symmetry_rule(state_str)
            # 0-indexed busbar convention: the first element must sit on bus 0, so a
            # configuration passes the symmetry rule iff its first digit is "0".
            should_pass_test = state_str[0] == "0"
            assert should_pass_test == pass_test, f"{state_str} symmetry rule was not passed"
            n_elements += passes_n_elements_rule(state_str)
            n_island += passes_islanded_elements_rule(sub, state_str)

    n_sym_correct = 144
    assert n_sym == n_sym_correct, "Different number of actions passing two-busbar symmetry rule."

    n_elements_correct = 216
    assert n_elements == n_elements_correct, "Different number of actions passing n-elements rule."

    n_island_correct = 262
    assert n_island == n_island_correct, "Different number of actions passing islanded elements rule."
