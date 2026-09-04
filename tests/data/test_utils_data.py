import numpy as np

from pandapower_env.data import utils_data


def test_poly_fit_error_perfect_fit() -> None:
    # quadratic signal, should be fit perfectly by degree=2 polynomial
    y = np.array([i**2 for i in range(10)])
    err = utils_data.poly_fit_error(y, degree=2)
    assert np.isclose(err, 0.0), f"Expected zero error, got {err}"


def test_poly_fit_error_noise_increases_error() -> None:
    y_clean = np.linspace(0, 10, 50)
    rng = np.random.default_rng(0)  # create a Generator with a seed
    y_noisy = y_clean + rng.normal(0, 1, size=len(y_clean))
    err_clean = utils_data.poly_fit_error(y_clean, degree=1)
    err_noisy = utils_data.poly_fit_error(y_noisy, degree=1)
    assert err_noisy > err_clean, "Noisy signal should have higher error"


def test_run_single_scenario_returns_list(env_config) -> None:
    episode_length = env_config["episode_length"]
    result = utils_data.run_single_scenario([0, env_config])
    assert isinstance(result, list)
    assert all(isinstance(x, float) for x in result)
    assert len(result) == episode_length  # should always run 96 steps


def test_find_smooth_scenario_indices_sorted(env_config) -> None:
    indices = utils_data.find_smooth_scenario(env_config, threshold=0)
    # should return indices [0]
    assert sorted(indices) == [np.int64(0)]
    assert isinstance(indices, list)
    assert all(isinstance(i, np.integer) for i in indices)
