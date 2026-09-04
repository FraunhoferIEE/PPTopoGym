import numpy as np

from pandapower_env.environments.simulation_env import PPTopoGym

# A line is "overloaded" from this loading upwards; below it the grid is healthy and the agent
# gets full reward regardless of how the DoNothing baseline did.
OVERLOAD_THRESHOLD_PERCENT = 100.0

# Marks that a DoNothing rollout is in progress on this environment. The rollout drives
# ``env.step()``, which calls the reward function again -- without this flag that recursion
# never terminates (see _worst_donothing_loading).
_ROLLOUT_FLAG = "_donothing_rollout_active"


def reward_normalized(env: PPTopoGym, max_loading: int = 150, min_loading: int = 50) -> float:
    """Calc a normalized reward."""
    max_line_loading =  env.net.res_line["loading_percent"].max()
    if np.isnan(max_line_loading):
        return env.worst_reward
    reward = (max_loading - max_line_loading) / (max_loading - min_loading)
    if reward >= 0:
        return np.clip(reward, 0, 1)
    return np.clip(reward, -0.5, 0)


def reward_better_than_donothing(env: PPTopoGym) -> float:
    """Reward the agent for keeping the grid below what DoNothing would have done.

    Once per episode the worst line loading a pure DoNothing policy reaches is measured by
    rolling the episode out (see :func:`_worst_donothing_loading`) and cached on
    ``env.cache``. The agent is then scored on its own worst loading so far this episode:
    a healthy grid scores 1.0, and an overloaded one scores a sigmoid of how far below the
    DoNothing baseline it stayed -- 0.5 for matching it, above 0.5 for beating it.

    :param env: the environment being scored; read after its power flow has converged.
    :type env: PPTopoGym
    :return: the reward, in ``(0, 1]``.
    :rtype: float
    """
    if env.cache.get(_ROLLOUT_FLAG):
        # Called from inside the DoNothing rollout below, whose rewards are never read.
        return 0.0

    worst_donothing_loading = env.cache.setdefault("DoNothing_worst_loading", {})
    worst_agent_loading = env.cache.setdefault("max_line_loading", {})
    if env.episode_step_counter == 0:
        # First step of this episode: the running agent maximum must not carry over from a
        # previous visit to the same episode index. The DoNothing baseline may -- it is a
        # property of the episode's start state, which is identical on every visit.
        worst_agent_loading.pop(env.episode_index, None)

    if env.episode_index not in worst_donothing_loading:
        worst_donothing_loading[env.episode_index] = _worst_donothing_loading(env)
    baseline = worst_donothing_loading[env.episode_index]

    agent_loading = max(
        _max_line_loading(env),
        worst_agent_loading.get(env.episode_index, 0.0),
    )
    worst_agent_loading[env.episode_index] = agent_loading

    if agent_loading < OVERLOAD_THRESHOLD_PERCENT or baseline <= 0.0:
        return 1.0
    decrease = baseline - agent_loading
    return float(_sigmoid(2 * decrease / baseline))


def _worst_donothing_loading(env: PPTopoGym) -> float:
    """Roll the rest of the episode out with DoNothing and report the worst line loading seen.

    The environment is driven forward in place and put back afterwards. State is saved and
    restored with :meth:`PPTopoGym.save_state` / :meth:`PPTopoGym.restore_state` rather than
    ``start_simulation`` / ``end_simulation``: this runs from *inside* ``step()``, before the
    current action has been appended to ``log_actions``, so the action-log replay that
    ``end_simulation`` performs would restore the grid to the state *before* the action being
    scored and the caller's observation would then describe the wrong topology.
    ``save_state`` captures the live topology instead, so the action survives the round trip.

    ``episode_step_counter`` is saved separately because ``restore_state`` deliberately does not
    touch it, and every rolled-out step raises it.

    :param env: the environment to roll out; left exactly as it was found, results included.
    :type env: PPTopoGym
    :return: the highest line loading (percent) reached over the remainder of the episode.
    :rtype: float
    """
    state = env.save_state()
    episode_step_counter = env.episode_step_counter
    env.cache[_ROLLOUT_FLAG] = True
    worst_loading = _max_line_loading(env)
    try:
        while True:
            _, _, terminated, truncated, _ = env.step(0)
            if terminated:  # power flow failed -- the loadings are meaningless
                break
            worst_loading = max(worst_loading, _max_line_loading(env))
            if truncated:
                break
    finally:
        env.cache[_ROLLOUT_FLAG] = False
        env.restore_state(state, run_pf=True)
        env.episode_step_counter = episode_step_counter
    return worst_loading


def _max_line_loading(env: PPTopoGym) -> float:
    """Highest line loading in percent, with a missing/NaN result read as 0.0."""
    loading = env.net.res_line["loading_percent"].max()
    return 0.0 if np.isnan(loading) else float(loading)


def _sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))
