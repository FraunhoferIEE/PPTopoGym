from pandapower.auxiliary import pandapowerNet

from pandapower_env.action_space.action_space import create_unitary_substation_action_one_subs, return_donothing_action


def actions_list_stations_dn(net: pandapowerNet, substation: list[int]) -> list:
    connection_lengths = [len(net.multi_bb_substation.loc[sub, "connected_buses"]) for sub in substation]
    number_busbars: list[int] = [net.multi_bb_substation.loc[sub, "n_busbars_in_substation"] for sub in substation]
    unitary_actions: list= []
    all_bus_connection_tuples = list(
        zip(substation, connection_lengths, number_busbars),
    )
    for isub, connections, n_busbars in all_bus_connection_tuples:
        actions = create_unitary_substation_action_one_subs(
            n_connections=connections,
            bus=isub,
            number_busbars=n_busbars,
        )
        unitary_actions.extend(actions)
    action_offset = 1
    for action_counter, entry in enumerate(unitary_actions):
        entry["action"] = action_counter + action_offset
    unitary_actions.insert(0, return_donothing_action())
    return unitary_actions
