

import copy

import numpy as np
import pandapower as pp
import pandas as pd
import pytest
from pandapower import pandapowerNet

from pandapower_env.toolbox.utils import (
    _WARM_OPTIONS_KEY,
    _bus_vn_kv,
    run_nminus1_powerflow,
    run_powerflow,
    total_active_overload_mva,  # Change 'your_module' to your filename
)

# A line is loaded beyond its rating above this loading -- pandapower's own definition.
FULL_LOAD_PERCENT = 100.0


def test_total_active_overload_calculation(overloaded_net) -> None:
    # 1. Run the function on the overloaded network
    overload_val = total_active_overload_mva(overloaded_net)

    # Assert that overload is detected (> 0)
    assert overload_val > 0.0

    # 2. Verify results are added to net (the function calls runpp internally)
    assert overloaded_net.res_line is not None

    # 3. Test the "No Overload" case
    # Remove the load or set to zero
    overloaded_net.load.p_mw = 0.0
    overloaded_net.load.q_mvar = 0.0

    # Re-run power flow to update results
    pp.runpp(overloaded_net)

    no_overload_val = total_active_overload_mva(overloaded_net)
    assert no_overload_val == 0.0


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


def _cold_run(net: pandapowerNet, pf_type: str = "ac") -> np.ndarray:
    """Run a power flow with the warm-options marker cleared, and return the line loadings.

    Dropping the marker forces :func:`run_powerflow` down the full ``pp.runpp`` path, so this
    is the reference the warm fast path must reproduce.

    :param net: a pandapower network, solved in place
    :type net: pandapowerNet
    :param pf_type: the powerflow type, either 'ac' or 'dc'
    :type pf_type: str
    :return: ``res_line.loading_percent`` after the cold solve.
    :rtype: np.ndarray
    """
    net.pop(_WARM_OPTIONS_KEY, None)
    run_powerflow(net=net, pf_type=pf_type)
    return net.res_line["loading_percent"].to_numpy().copy()


def test_run_powerflow_warm_matches_cold(test_grid_dbb_plus_simbench) -> None:
    """The warm-options fast path must reproduce the cold ``pp.runpp`` result exactly."""
    net = test_grid_dbb_plus_simbench
    cold = _cold_run(copy.deepcopy(net))

    run_powerflow(net=net)  # parses options and sets the marker
    run_powerflow(net=net)  # takes the warm path
    warm = net.res_line["loading_percent"].to_numpy()

    np.testing.assert_array_equal(warm, cold)


def test_run_powerflow_warm_reflects_topology_change(test_grid_dbb_plus_simbench) -> None:
    """A topology change must be picked up on the warm path, not masked by a stale ppc."""
    net = test_grid_dbb_plus_simbench
    run_powerflow(net=net)  # warm the options up on the unchanged topology

    switch = net.switch.index[0]
    net.switch.loc[switch, "closed"] = not net.switch.loc[switch, "closed"]

    cold = _cold_run(copy.deepcopy(net))
    run_powerflow(net=net)  # warm path, changed topology

    np.testing.assert_allclose(net.res_line["loading_percent"].to_numpy(), cold, equal_nan=True)


def test_run_powerflow_reparses_on_pf_type_and_ls2g_change(test_grid_dbb_plus_simbench) -> None:
    """Options must be re-parsed whenever pf_type, use_ls2g, or ``_options`` itself changes."""
    net = test_grid_dbb_plus_simbench

    run_powerflow(net=net, pf_type="ac")
    ac_mode = bool(net._options["ac"])
    run_powerflow(net=net, pf_type="dc")
    dc_mode = bool(net._options["ac"])
    run_powerflow(net=net, pf_type="ac")
    ac_again = bool(net._options["ac"])
    assert (ac_mode, dc_mode, ac_again) == (True, False, True), (
        "a pf_type change must re-parse the options"
    )

    run_powerflow(net=net, pf_type="ac", use_ls2g=False)
    ls2g_off = bool(net._options["lightsim2grid"])
    run_powerflow(net=net, pf_type="ac", use_ls2g=True)
    ls2g_on = bool(net._options["lightsim2grid"])
    assert (ls2g_off, ls2g_on) == (False, True), "a use_ls2g change must re-parse the options"

    del net._options
    run_powerflow(net=net, pf_type="ac")
    assert "_options" in net, "missing _options must fall back to the full path"


def test_run_powerflow_reparses_on_gen_vm_pu_change(test_grid_dbb_plus_simbench) -> None:
    """Changing generator voltage setpoints must re-parse, keeping results bit-exact.

    ``init_vm_pu`` is derived from the in-service gen/ext_grid setpoints and would otherwise
    stay frozen at its warm value, changing the Newton-Raphson starting point.
    """
    net = test_grid_dbb_plus_simbench
    run_powerflow(net=net)

    net.gen["vm_pu"] = net.gen["vm_pu"] * 1.01
    net.ext_grid["vm_pu"] = net.ext_grid["vm_pu"] * 1.01

    cold = _cold_run(copy.deepcopy(net))
    run_powerflow(net=net)

    np.testing.assert_array_equal(net.res_line["loading_percent"].to_numpy(), cold)


def test_warm_marker_does_not_survive_json(test_grid_dbb_plus_simbench) -> None:
    """The marker must die with ``_options`` across the JSON roundtrip used by workers.

    ``greedy_worker`` and ``nminus1_parallel`` ship nets into child processes as JSON. If the
    marker survived while ``_options`` did not, a child would take the fast path with no
    options parsed.
    """
    net = test_grid_dbb_plus_simbench
    run_powerflow(net=net)
    assert _WARM_OPTIONS_KEY in net

    rebuilt = pp.from_json_string(pp.to_json(net))
    assert _WARM_OPTIONS_KEY not in rebuilt
    assert "_options" not in rebuilt

    run_powerflow(net=rebuilt)
    assert hasattr(rebuilt, "res_line")


def test_total_active_overload_respects_parallel_and_derating(overloaded_net) -> None:
    """The MVA rating must include ``df`` and ``parallel``, as pandapower's loading does.

    ``total_active_overload_mva`` rated a line at ``sqrt(3) * vn_kv * max_i_ka`` alone, ignoring
    the derating factor and the number of parallel systems -- the very terms pandapower divides
    by when it computes ``res_line.loading_percent``. A derated line was therefore reported as
    less overloaded than it is, and a double circuit as more. Both default to 1, which is why
    the shipped case30 grid never exposed it.
    """
    pp.runpp(overloaded_net)
    single_circuit = total_active_overload_mva(overloaded_net)
    assert single_circuit > 0.0

    # Doubling the circuits doubles the rating, so the overload must shrink.
    overloaded_net.line["parallel"] = 2
    overloaded_net.trafo["parallel"] = 2
    pp.runpp(overloaded_net)
    double_circuit = total_active_overload_mva(overloaded_net)
    assert double_circuit < single_circuit

    # Halving the derating factor halves the rating again, so the overload must grow back.
    overloaded_net.line["df"] = 0.5
    overloaded_net.trafo["df"] = 0.5
    pp.runpp(overloaded_net)
    derated = total_active_overload_mva(overloaded_net)
    assert derated > double_circuit


def test_total_active_overload_matches_pandapower_loading(overloaded_net) -> None:
    """An element pandapower calls overloaded must contribute a positive overload, and vice versa.

    This ties the hand-rolled MVA rating to ``res_line.loading_percent``: the two must agree on
    *whether* each line is over its limit, which is what the ``df`` / ``parallel`` terms decide.
    """
    overloaded_net.line["parallel"] = 3
    pp.runpp(overloaded_net)

    line_overloaded = (overloaded_net.res_line["loading_percent"] > FULL_LOAD_PERCENT).to_numpy()
    net_lines_only = copy.deepcopy(overloaded_net)
    net_lines_only.trafo = net_lines_only.trafo.iloc[0:0]
    net_lines_only.res_trafo = net_lines_only.res_trafo.iloc[0:0]
    line_only_overload = total_active_overload_mva(net_lines_only)

    assert (line_only_overload > 0.0) == bool(line_overloaded.any())


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("run_pf", [run_powerflow, run_nminus1_powerflow])
def test_unknown_pf_type_raises(run_pf, test_grid_dbb_plus_simbench) -> None:
    """Only 'ac' and 'dc' are power flow types; anything else is a typo, not a fallback."""
    with pytest.raises(ValueError, match="pf_type must be 'ac' or 'dc'"):
        run_pf(test_grid_dbb_plus_simbench, pf_type="acdc")


@pytest.mark.parametrize("run_pf", [run_powerflow, run_nminus1_powerflow])
def test_non_bool_use_ls2g_raises(run_pf, test_grid_dbb_plus_simbench) -> None:
    """``use_ls2g`` is a tri-state (True/False/'auto'); other strings must not pass silently."""
    with pytest.raises(ValueError, match="use_ls2g must be bool or 'auto'"):
        run_pf(test_grid_dbb_plus_simbench, use_ls2g="yes")


def test_dc_powerflow_with_ls2g_warns(test_grid_dbb_plus_simbench, caplog) -> None:
    """lightsim2grid has no DC backend here, so asking for both is downgraded with a warning."""
    net = test_grid_dbb_plus_simbench
    net.line["max_loading_percent"] = 100.0
    net.trafo["max_loading_percent"] = 100.0

    with caplog.at_level("WARNING"):
        run_nminus1_powerflow(net, pf_type="dc", use_ls2g=True)

    assert "can't be used for DC powerflow" in caplog.text


# ---------------------------------------------------------------------------
# Bus voltage lookup
# ---------------------------------------------------------------------------


def test_bus_vn_kv_positional_and_label_paths_agree(test_grid_dbb_plus_simbench) -> None:
    """The fast positional path is only valid on a 0..n-1 index; both paths must return the same kV."""
    net = test_grid_dbb_plus_simbench
    bus_labels = net.line["from_bus"].to_numpy()

    positional = _bus_vn_kv(net, bus_labels)

    # Re-index the bus table off the RangeIndex to force the .loc fallback.
    relabelled = copy.deepcopy(net)
    offset = 1000
    relabelled.bus.index = relabelled.bus.index + offset
    by_label = _bus_vn_kv(relabelled, bus_labels + offset)

    np.testing.assert_allclose(positional, by_label)


# ---------------------------------------------------------------------------
# Three-winding transformers
# ---------------------------------------------------------------------------


def test_total_active_overload_includes_trafo3w(test_grid_dbb_plus_simbench) -> None:
    """A loaded 3W transformer contributes its per-side excess apparent power to the total."""
    net = test_grid_dbb_plus_simbench
    run_powerflow(net)
    baseline = total_active_overload_mva(net)

    # Fabricate one 3W result row whose HV side sits 5 MVA above its rating.
    net.trafo3w = pd.DataFrame(
        {"sn_hv_mva": [10.0], "sn_mv_mva": [10.0], "sn_lv_mva": [10.0]},
        index=[0],
    )
    net.res_trafo3w = pd.DataFrame(
        {
            "p_hv_mw": [15.0], "q_hv_mvar": [0.0],
            "p_mv_mw": [4.0], "q_mv_mvar": [3.0],
            "p_lv_mw": [0.0], "q_lv_mvar": [0.0],
        },
        index=[0],
    )

    # HV: sqrt(15^2) - 10 = 5. MV: sqrt(4^2 + 3^2) = 5 -> under the 10 MVA rating. LV: 0.
    assert total_active_overload_mva(net) == pytest.approx(baseline + 5.0)
