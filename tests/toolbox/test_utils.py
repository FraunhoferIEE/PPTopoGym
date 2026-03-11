

from pandapower_env.toolbox.utils import (
    run_nminus1_powerflow,
    run_powerflow,
)


def test_run_nminus1_powerflow(test_grid_dbb_plus_simbench) -> None:
    """
    Test the run_powerflow function.

    :param test_grid_dbb_plus_simbench: test_grid_dbb_plus_simbench fixture (see conftest.py)
    :type test_grid_dbb_plus_simbench: pandapowerNet
    """
    net = test_grid_dbb_plus_simbench

    del net.res_line
    del net.res_bus

    run_nminus1_powerflow(net=net, pf_type="ac")

    assert hasattr(net, "res_line"), "Error: net.res_line does not exist!"
    assert hasattr(net, "res_bus"), "Error: net.res_bus does not exist!"

    del net.res_line
    del net.res_bus

    run_nminus1_powerflow(net=net, pf_type="dc")

    assert hasattr(net, "res_line"), "Error: net.res_line does not exist!"
    assert hasattr(net, "res_bus"), "Error: net.res_bus does not exist!"

    del net.res_line
    del net.res_bus
    del net._options

    run_nminus1_powerflow(net=net, pf_type="ac", use_ls2g=True)

    assert hasattr(net, "res_line"), "Error: net.res_line does not exist!"
    assert hasattr(net, "res_bus"), "Error: net.res_bus does not exist!"
    assert net._options["lightsim2grid"] is True, "Error: lightsim2grid wasn't used!"


def test_run_powerflow(test_grid_dbb_plus_simbench) -> None:
    """
    Test the run_powerflow function.

    :param test_grid_dbb_plus_simbench: test_grid_dbb_plus_simbench fixture (see conftest.py)
    :type test_grid_dbb_plus_simbench: pandapowerNet
    """
    net = test_grid_dbb_plus_simbench

    del net.res_line
    del net.res_bus

    run_powerflow(net=net, pf_type="ac")

    assert hasattr(net, "res_line"), "Error: net.res_line does not exist!"
    assert hasattr(net, "res_bus"), "Error: net.res_bus does not exist!"

    del net.res_line
    del net.res_bus

    run_powerflow(net=net, pf_type="dc")

    assert hasattr(net, "res_line"), "Error: net.res_line does not exist!"
    assert hasattr(net, "res_bus"), "Error: net.res_bus does not exist!"

    del net.res_line
    del net.res_bus
    del net._options

    run_powerflow(net=net, pf_type="ac", use_ls2g=True)

    assert hasattr(net, "res_line"), "Error: net.res_line does not exist!"
    assert hasattr(net, "res_bus"), "Error: net.res_bus does not exist!"
    assert net._options["lightsim2grid"] is True, "Error: lightsim2grid wasn't used!"

