# PPTopoGym

## Getting Started

The module is built for distribution using poetry and we recommend the same for usage and development.

### Installation

The following steps should help you install gridgenerate in a virtual environment. Note that Python and poetry must be installed totally independent of Anaconda to avoid conflicts!

Step 1. Download and install Python 3.10.11 from [python.org](https://www.python.org/downloads/release/python-31011/)

Step 2. Install Poetry using the official installer and add Poetry to your PATH. See [python-poetry.org](https://python-poetry.org/docs/#installing-with-the-official-installer) for instructions.

Step 3. Create a virtual environment: `poetry env use <path_to_python_executable>`

Step 4. Install: `poetry install`

## The code

The code basis are the environments `GymEnvPP` and `SimulationEnv`, saved in `pandapower_env/environments`.

### Overview of the code

For demonstration, we provide several notebooks:

- `getting_started.ipynb` shows how to get started with the environment.
- `pandapower_nets_demo.ipynb` shows how to generate custom pandapower power grids.
- `action_space.ipynb` shows how the action space is built and how unrealistic grid states are filtered out.
- `observations.ipynb` shows how observations are made in the current environment `SimulationEnv`.
- `scenarios_scaling.ipynb` shows how to select profiles based on the renewable energy generation, and scale them to the net such that overloads occur.
- `environments_demonstration.ipynb` Puts this all together and shows how to initialize an environment with pre-defined parameters for the power grid, profiles and action space.
- `agents.ipynb` Shows how the provided benchmark agents can be used and evaluated on a provided power grid.


### Details about the code

- In `action_space`, actions for switching double busbars, and switching lines, are created. Further, unrealistic actions are filtered out.
- In `agents`, the agents are created object-oriented: There exist base-agents with core functionalities and built upon benchmark-agents that can be used by users. The greedy agents simulate every action and evaluate it using a feedback function, and the greedy rollout agents simulate for each action additional rollouts, and evaluate these rollouts using a feedback function for the rollouts, and then again aggregate the rollout feedback to generate an overall evaluation for each action.
- In `environments`, the two basis environments are stored. The more sophisticated `SimulationEnv` is able to simulate actions and restore the power grid state. It implements simple observations, e.g., graph observations and maximal line loadings.
- In `metrics`, a class for evaluating a sequence of actions is provided.
- In `observation_space` functions to create observations for the agent are gathered
- In `substation` functionalities to create and plot double busbars are gathered.
- In `toolbox` additional utility functions, like scaling the power grid elements, or running a N-1 power flow, are gathered.

## Reproducing the results
The figures and results in the paper have been made with the notebooks `getting_started.ipynb` and `agents.ipynb`. The agent evaluation is stored in `notebooks/plots`.
