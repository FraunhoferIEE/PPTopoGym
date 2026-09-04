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

    my_env.reset()
    my_env.step(0)
    my_env.load_action({0: 0})
    my_env.create_observation()
    my_env.calculate_reward()
    # test the run powerflow method
    assert my_env.run_pf(nminus1=True)
