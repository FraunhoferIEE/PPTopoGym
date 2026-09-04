from itertools import chain, combinations

from pandapower import pandapowerNet


def get_all_substation_switches(net: pandapowerNet, i_sub: int) -> list[int]:
    """
    Get a list of all the substation switches in a given substation.

    :param net: The pandapower net (must have net.multi_bb_substation)
    :type net: pandapowerNet
    :param i_sub: index of the substation in the net.multi_bb_substation Dataframe
    :type i_sub: int
    :return: The index of each of the switches in the substation
    :rtype: list of int
    """
    sub = net.multi_bb_substation.loc[i_sub]
    max_number_busbars = sub.n_busbars_in_substation
    return list(
        chain(
            (
                switch
                for i in range(max_number_busbars)
                for switch in sub[f"b{i}_switches"]
            ),
            (
                sub[f"b{i}{j}_switch"]
                for i, j in combinations(range(max_number_busbars), 2)
            ),
        ),
    )


def get_list_of_closed_substation_switches(
    net: pandapowerNet,
    i_sub: int,
    switch_hex: str,  # hex-str
    open_switches: list[int] | None = None,
) -> list:
    """
    Given a bitset corresponding to a busbar assignment in a substation, return the switches to be closed.

    :param net: The pandapower net (must have net.multi_bb_substation)
    :type net: pandapowerNet
    :param i_sub: index of the substation in the net.multi_bb_substation Dataframe
    :type i_sub: int
    :param switch_bits: binary bitset containing busbar assignment
    :type switch_bits: int
    :return: List of closed switches corresponding to the bitset
    :rtype: list of int
    """
    # check if the hexstr has a 0x prefix
    if switch_hex.startswith("0x"):
        switch_hex = switch_hex[2:]
    if open_switches is None:
        open_switches = get_list_of_open_substation_switches(net, i_sub, switch_hex)
    all_switches = get_all_substation_switches(net, i_sub)

    return list(set(all_switches).difference(open_switches))


def get_list_of_open_substation_switches(
    net: pandapowerNet,
    i_sub: int,
    switch_hex: str,  # hex-int, starting with 0x
) -> list[int]:
    """
    Given a hexset representing a substation switch, return the switches to be opened.

    Output is a list of switch indices (idx from net.switch) in the substation,
    for both switches to elements and busbars.
    list[int] is used instead of np.ndarray, as we don't do any operations upon.
    Workflow:
    0. Check if the substation is fully connected -> Nothing is opened
    1. Call function to get switches to busbars
    2. Call function to get switches to elements

    :param net: The pandapower net (must have net.multi_bb_substation)
    :type net: pandapowerNet
    :param i_sub: index of the substation in the net.multi_bb_substation Dataframe
    :type i_sub: int
    :param switch_bits: hexadecimal charset containing busbar assignment
    :type switch_bits: int
    :return open_switches: List of open switches corresponding to the bitset
    :rtype: list of int
    """
    # check if the hexstr has a 0x prefix
    if switch_hex.startswith("0x"):
        switch_hex = switch_hex[2:]
    # check if the len of the hexstr is equal to the number of busbars
    n_busbars = net.multi_bb_substation.loc[i_sub, "n_busbars_in_substation"]
    n_sub_elements = len(net.multi_bb_substation.loc[i_sub, "connected_buses"])
    if len(str(switch_hex)) != n_sub_elements:
        msg = f"Hex-String encodes {len(str(switch_hex))} busbars, but the substation has {n_sub_elements} elements."
        raise ValueError(msg)
    if _is_fully_connected(
        switch_hex=switch_hex,
    ):
        return []
    # open switches to busbars.
    all_open_switches = _get_open_switches_to_busbars(
        net=net,
        i_sub=i_sub,
        n_busbars=n_busbars,
    )
    # open switches to elements.
    for i in range(n_busbars):
        all_open_switches.extend(
            _get_open_switches_to_elements(
                net=net,
                i_sub=i_sub,
                i_busbar=i,
                hex_str=switch_hex,
            ),
        )
    return all_open_switches


def _is_fully_connected(
    switch_hex: str,
) -> bool:
    """
    Check if the substation is fully connected.

    Use number-repr. of the hex for checks using bitshifts.
    Workflow:
    1. Get the hexset corresponding to the busbar assignment
    2. Checks:
        - check if the hexset does not start with 0x
    3. Is the hexset fully connected?
        - create a string of the hexset
        - check if all characters in the string are the same, via converting it to a set & checking the size.

    :param net: The pandapower net (must have net.multi_bb_substation)
    :type net: pandapowerNet
    :param i_sub: index of the substation in the net.multi_bb_substation Dataframe
    :type i_sub: int
    :param n_busbars: number of busbars in the substation
    :type n_busbars: int
    :param switch_bits: hexadecimal charset containing busbar assignment
    :type switch_bits: int
    :return: True if the substation is fully connected, False otherwise
    :rtype: bool
    """
    # Check if the hexset does not start with 0x
    if switch_hex.startswith("0x"):
        switch_hex = switch_hex[2:]
    # Is the hexset fully connected?
    return len(set(switch_hex)) == 1


def _get_open_switches_to_busbars(
    net: pandapowerNet,
    i_sub: int,
    n_busbars: int,
) -> list[int]:
    """
    Given a hexset representing a substation switch, return the switches to be opened.

    Output is a list of switch indices (idx from net.switch) in the substation,
    for both switches to elements and busbars.
    Workflow:
    1. Call function to get switches to busbars
    2. Call function to get switches to elements

    :param net: The pandapower net (must have net.multi_bb_substation)
    :type net: pandapowerNet
    :param i_sub: index of the substation in the net.multi_bb_substation Dataframe
    :type i_sub: int
    :param switch_bits: hexadecimal charset containing busbar assignment
    :type switch_bits: int
    :return open_switches: List of open switches corresponding to the bitset
    :rtype: list of int
    """
    open_switches = []
    for i, j in combinations(range(n_busbars), 2):
        switch = net.multi_bb_substation.loc[i_sub, f"b{i}{j}_switch"]
        if switch in net.switch.index:
            open_switches.append(switch)
    return open_switches


def _get_open_switches_to_elements(
    net: pandapowerNet,
    i_sub: int,
    i_busbar: int,
    hex_str: str,
) -> list[int]:
    """
    Given a hexset representing a substation switch, return the switches to be opened.

    Workflow:
    - Input: hex_str as string
    - Goal: For the busbar i, get all digits in hex_str eq. to i_busbar
    - Idea: Use char-wise comparation of the hex-str.
    - Lookout! we read the switches in reverse order (does not matter, as all actions are symmetric)
    2. Extract the switches from the substation DF

    :param net: The pandapower net (must have net.multi_bb_substation)
    :type net: pandapowerNet
    :param i_sub: index of the substation in the net.multi_bb_substation Dataframe
    :type i_sub: int
    :param switch_bits: hexadecimal charset containing busbar assignment
    :type switch_bits: int
    :return open_switches: List of open switches corresponding to the bitset
    :rtype: list of int
    """
    all_switches = net.multi_bb_substation.loc[i_sub, f"b{i_busbar}_switches"]
    return [
        switch
        for hex_id, switch in zip(hex_str, all_switches)
        if hex_id == str(i_busbar)
    ]
