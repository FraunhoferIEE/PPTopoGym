from pandapower_env.environments.gym_env_pp import BaseEnvPP


def test_gym_env_pp(test_grid_dbb_plus_simbench) -> None:
    """
    Test the base gym environment.

    :param test_grid_dbb_plus_simbench: Grid with 14 buses, with a double-busbar Dataframe
        already created, plus simbench-style profiles.
    :type test_grid_dbb_plus_simbench: pandapowerNet
    """
    env_config = {
        "n_episodes": 10,
        "episode_length": 5,
        "net": test_grid_dbb_plus_simbench,
        "nminus1": False,
    }

    class CustomEnv(BaseEnvPP):

        def __init__(self, env_config) -> None:
            super().__init__(env_config)

        def load_action(self, action) -> None:
            pass

        def create_observation(self, run_pf: bool | None = None) -> list[float]:  #noqa: ARG002
            return [0.0] * 5

        def calculate_reward(self) -> float:
            return 0.0

    my_env = CustomEnv(env_config=env_config)
    my_env.render()
    my_env.reset(options={"index": 4})
    last_value_res = my_env.df_profiles_load_p.iloc[-1, -1]
    my_env.reset(options={"index": 0})
    my_env.load_profile_timestep_into_net(4)
    last_value = my_env.df_profiles_load_p.iloc[-1, -1]
    assert last_value == last_value_res
    a, b = my_env.reset()
    assert isinstance(a, dict)
    assert isinstance(b, dict)
    my_env.step(0)
    my_env.load_action({0: 0})
    my_env.create_observation()
    my_env.calculate_reward()
    # test the run powerflow method
    assert my_env.run_pf(nminus1=True)
# test if profiles are loaded correctly
def test_setup_profiles(test_grid_dbb_plus_simbench) -> None:
    net = test_grid_dbb_plus_simbench
    assert "load" in net.profiles
    net.load.p_mw = 1
    net.gen.p_mw = 1
    net.sgen.p_mw = 1
    net.gen.q_mvar = 1
    env_config = {
        "n_episodes": 10,
        "episode_length": 5,
        "net": net,
        "nminus1": False,
    }
    class CustomEnv(BaseEnvPP):

        def __init__(self, env_config) -> None:
            super().__init__(env_config)

        def load_action(self) -> None:
            pass

        def create_observation(self) -> list[float]:
            return [0.0] * 5

        def calculate_reward(self) -> float:
            return 0.0
    env = CustomEnv(env_config=env_config)
    number_loads = len(net.load)
    len_profiles = len(net.profiles["powerplants"])
    assert number_loads == len(env.df_profiles_load_p.columns)
    assert len_profiles == len(env.df_profiles_load_p)
    # are the values correct?
    load0_values = env.df_profiles_load_p.iloc[:, 0]
    assert load0_values.shape == (len_profiles,)
    assert (load0_values == [3,3,3,3,2]).all(), load0_values
    # last value should be 3.0
    last_value = 3
    assert env.df_profiles_load_p.iloc[-1, -1] == last_value, env.df_profiles_load_p
    #test all shapes
    assert len(env.df_profiles_sgen_q) == len_profiles
    assert len(env.df_profiles_gen_vm) == len_profiles

    # also test the load function
    env.load_action()
    assert env.net.converged is None



