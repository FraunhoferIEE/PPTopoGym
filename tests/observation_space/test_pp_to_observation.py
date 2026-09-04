import pandapower as pp
import pandapower.contingency
import pytest

from pandapower_env.observation_space.pp_to_observation import (
    clipped_line_loading_observation_space,
    line_loading_mean,
    line_overloading_percentage,
    line_overloading_sum,
    line_power_losses_sum,
    nminus1_bus_overloading_percentage,
    nminus1_line_loading_max,
    nminus1_line_loading_mean,
    nminus1_line_overloading_percentage,
    nminus1_line_overloading_sum,
    open_busbar_coupler_percentage,
    overload_energy,
)


def test_clipped_line_loading_observation_space(test_grid) -> None:
    """
    Test the clipped_line_loading_observation_space function.

    :param test_grid: test_grid fixture (see conftest.py)
    :type test_grid: pandapowerNet
    """
    net = test_grid
    pp.runpp(net)

    observation = clipped_line_loading_observation_space(net)

    assert(len(observation) == len(net.line)), "Mismatch in length of observation and net.line length."


def test_line_overloading_percentage(test_grid) -> None:
    """
    Test the line_overloading_percentage function.

    :param test_grid: test_grid fixture (see conftest.py)
    :type test_grid: pandapowerNet
    """
    net = test_grid
    del net.res_line
    del net.res_bus

    pp.runpp(net)
    reward = line_overloading_percentage(net, 80.0)

    assert isinstance(reward, float) is True
    assert reward >= 0.0
    assert reward <= 1.0


def test_nminus1_line_overloading_percentage(test_grid) -> None:
    """
    Test the nminus1_line_overloading_percentage function.

    :param test_grid: test_grid fixture (see conftest.py)
    :type test_grid: pandapowerNet
    """
    net = test_grid
    del net.res_line
    del net.res_bus

    nminus1_cases = {"line": {"index": net.line.index.to_numpy()}}
    pp.contingency.run_contingency(net, nminus1_cases)
    reward = nminus1_line_overloading_percentage(net, 80.0)

    assert isinstance(reward, float) is True
    assert reward >= 0.0
    assert reward <= 1.0


def test_line_overloading_sum(test_grid) -> None:
    """
    Test the line_overloading_sum function.

    :param test_grid: test_grid fixture (see conftest.py)
    :type test_grid: pandapowerNet
    """
    net = test_grid
    del net.res_line
    del net.res_bus

    pp.runpp(net)
    reward = line_overloading_sum(net, 0.5)

    assert isinstance(reward, float) is True

def test_nminus1_line_overloading_sum(test_grid) -> None:
    """
    Test the nminus1_line_overloading_sum function.

    :param test_grid: test_grid fixture (see conftest.py)
    :type test_grid: pandapowerNet
    """
    net = test_grid
    del net.res_line
    del net.res_bus

    nminus1_cases = {"line": {"index": net.line.index.to_numpy()}}
    pp.contingency.run_contingency(net, nminus1_cases)
    reward = nminus1_line_overloading_sum(net, 0.5)

    assert isinstance(reward, float) is True

    # Check if error raises when n-1 loadflow has not been calculated
    net = test_grid
    del net.res_line
    del net.res_bus

    pp.runpp(net)
    with pytest.raises(ValueError):  # noqa: PT011
        nminus1_line_overloading_sum(net, 0.5)


def test_nminus1_bus_overloading_percentage(test_grid) -> None:
    """
    Test the nminus1_bus_overloading_percentage function.

    :param test_grid: test_grid fixture (see conftest.py)
    :type test_grid: pandapowerNet
    """
    net = test_grid
    del net.res_line
    del net.res_bus

    nminus1_cases = {"line": {"index": net.line.index.to_numpy()}}
    pp.contingency.run_contingency(net, nminus1_cases)
    reward = nminus1_bus_overloading_percentage(net)

    assert reward >= 0.0
    assert reward <= 1.0
    assert isinstance(reward, float) is True

    # Check if error raises when n-1 loadflow has not been calculated
    net = test_grid
    del net.res_line
    del net.res_bus

    pp.runpp(net)
    with pytest.raises(ValueError):  # noqa: PT011
        reward = nminus1_bus_overloading_percentage(net)


def test_line_loading_mean(test_grid) -> None:
    """
    Test the line_loading_mean function.

    :param test_grid: test_grid fixture (see conftest.py)
    :type test_grid: pandapowerNet
    """
    net = test_grid
    del net.res_line
    del net.res_bus

    pp.runpp(net)
    reward = line_loading_mean(net)

    assert isinstance(reward, float) is True


def test_nminus1_line_loading_mean(test_grid) -> None:
    """
    Test the nminus1_line_loading_mean function.

    :param test_grid: test_grid fixture (see conftest.py)
    :type test_grid: pandapowerNet
    """
    net = test_grid
    del net.res_line
    del net.res_bus

    nminus1_cases = {"line": {"index": net.line.index.to_numpy()}}
    pp.contingency.run_contingency(net, nminus1_cases)
    reward = nminus1_line_loading_mean(net)

    assert isinstance(reward, float) is True

    # Check if error raises when n-1 loadflow has not been calculated
    net = test_grid
    del net.res_line
    del net.res_bus

    pp.runpp(net)
    with pytest.raises(ValueError):  # noqa: PT011
        reward = nminus1_line_loading_mean(net)


def test_open_busbar_coupler_percentage(test_grid_multi_bb_substations) -> None:
    """
    Test the open_busbar_coupler_percentage function.

    :param test_grid_multi_bb_substations: test_grid_multi_bb_substations fixture (see conftest.py)
    :type test_grid_multi_bb_substations: pandapowerNet
    """
    net = test_grid_multi_bb_substations
    del net.res_line
    del net.res_bus

    reward = open_busbar_coupler_percentage(net)

    assert reward >= 0.0
    assert reward <= 1.0
    assert isinstance(reward, float) is True


def test_line_power_losses_sum(test_grid) -> None:
    """
    Test the line_power_losses_sum function.

    :param test_grid: test_grid fixture (see conftest.py)
    :type test_grid: pandapowerNet
    """
    net = test_grid
    del net.res_line
    del net.res_bus

    pp.runpp(net)
    reward = line_power_losses_sum(net)

    assert isinstance(reward, float) is True


def test_nminus1_line_loading_max(test_grid) -> None:
    """
    Test the nminus1_line_loading_max function.

    :param test_grid: test_grid fixture (see conftest.py)
    :type test_grid: pandapowerNet
    """
    net = test_grid
    del net.res_line
    del net.res_bus

    nminus1_cases = {"line": {"index": net.line.index.to_numpy()}}
    pp.contingency.run_contingency(net, nminus1_cases)
    reward = nminus1_line_loading_max(net)

    assert isinstance(reward, float) is True

    # Check if error raises when n-1 loadflow has not been calculated
    net = test_grid
    del net.res_line
    del net.res_bus

    pp.runpp(net)
    with pytest.raises(ValueError):  # noqa: PT011
        reward = nminus1_line_loading_max(net)


def test_overload_energy(test_grid) -> None:
    """
    Test the overload_energy function.

    :param test_grid: test_grid fixture (see conftest.py)
    :type test_grid: pandapowerNet
    """
    net = test_grid
    del net.res_line
    del net.res_bus

    pp.runpp(net)
    reward = overload_energy(net)

    assert isinstance(reward, float) is True
