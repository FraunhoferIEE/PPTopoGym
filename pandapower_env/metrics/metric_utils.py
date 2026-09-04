from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, cast

from pandapower_env.environments.simulation_env import PPTopoGym

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    import pandapower as pp


class FloatMetric:
    """Descriptor for evaluating metrics in a simulation environment.

    : param func: A callable that computes the metric. It must accept two arguments:
            - instance: the object holding metric state (e.g., a MetricContainer)
            - cache: a dictionary for storing persistent per-metric state
    : type func: Callable[[Any, dict[str, Any]], float]
    """

    def __init__(self, func: Callable[[Any, dict[str, Any]], float]) -> None:
        self.func: Callable[[Any, dict[str, Any]], float] = func
        self._name: str | None = None

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = name

    def __get__(
        self,
        instance: MetricContainer | None,
        owner: type | None = None,
    ) -> float | FloatMetric:
        if instance is None:
            return self

        name = self._name
        if name is None:
            msg = "FloatMetric name was not set via __set_name__"
            raise AttributeError(msg)
        return instance.__dict__.get(name, 0.0)

    def __set__(self, instance: MetricContainer, value: float) -> None:
        name = self._name
        if name is None:
            msg = "FloatMetric name was not set via __set_name__"
            raise AttributeError(msg)
        instance.__dict__[name] = value

    def get_cache(self, instance: MetricContainer) -> dict[str, Any]:
        if self._name is None:
            msg = "FloatMetric name was not set via __set_name__"
            raise AttributeError(msg)
        return instance.metric_state.setdefault(self._name, {})

    def get_name(self) -> str:
        if self._name is None:
            msg = "FloatMetric name was not set via __set_name__"
            raise AttributeError(msg)
        return self._name

    def format_with_instance(self, instance: MetricContainer) -> str:
        """
        Format the metric with its current value and cache for a specific instance.

        This method does not perform any calculations.
        """
        if self._name is None:
            msg = "FloatMetric name was not set via __set_name__"
            raise AttributeError(msg)

        # Get the current value (if it exists) without calculation
        value = instance.__dict__.get(self._name, "Not Evaluated")

        # Get the current cache (if it exists)
        cache = instance.metric_state.get(self._name, {})

        return f"{self._name}: value={value}, cache={cache}"

    def evaluate(self, instance: MetricContainer) -> float:
        if self._name is None:
            msg = "FloatMetric name was not set via __set_name__"
            raise AttributeError(msg)

        cache: dict[str, Any] = instance.metric_state.setdefault(self._name, {})
        result = self.func(instance, cache)
        result = cast(float, result)
        # assert test to insure float
        assert isinstance(  # noqa: S101
            result,
            float,
        ), f"Metric '{self._name}' must return a float, got {type(result).__name__}"

        instance.__dict__[self._name] = result
        return result

    def __str__(self) -> str:
        """Return a string representation of the metric, including name, value, and cache contents."""
        if self._name is None:
            msg = "FloatMetric name was not set via __set_name__"
            raise AttributeError(msg)

        return f"FloatMetric(name={self._name})"

    def __repr__(self, instance: MetricContainer | None = None) -> str:
        """Return a detailed representation of the metric.

        Including current value and cache if an instance is provided.
        """
        if self._name is None:
            msg = "FloatMetric name was not set via __set_name__"
            raise AttributeError(msg)

        if instance is None:
            return f"FloatMetric(name={self._name}, value=Not Evaluated, cache=Not Available)"

        value = instance.__dict__.get(self._name, "Not Evaluated")
        cache = instance.metric_state.get(self._name, {})

        return f"FloatMetric(name={self._name}, value={value}, cache={cache})"


class MetricBase:
    """Automatically adds the descriptor FloatMetric to all non-private methods in the class."""

    def __init_subclass__(cls, **kwargs: dict) -> None:
        super().__init_subclass__(**kwargs)
        for name, obj in cls.__dict__.items():
            if (
                callable(obj)
                and not name.startswith("_")
                and not isinstance(obj, FloatMetric)
            ):
                descriptor = FloatMetric(obj)
                setattr(cls, name, descriptor)
                descriptor.__set_name__(cls, name)


class MetricContainer:
    """
    Container that holds all state needed for metric evaluation.

    All sub-classes use the FloatMetric descriptor to define metrics.
    All sub-classes should also inherit from MetricBase to automatically add the descriptor.
    """

    def __init__(self) -> None:
        self.net: None | pp.pandapowerNet = None
        self.metric_state: dict[str, dict[str, Any]] = {}
        self.last_action: Any = None
        self.last_reward: float = 0.0
        self.last_obs: Any = None
        self.last_info: dict[str, Any] = {}

    def __init_subclass__(cls, **kwargs: dict) -> None:
        """Ensure that all subclasses also inherit from MetricBase."""
        super().__init_subclass__(**kwargs)

        # Check if the class inherits from MetricBase
        if MetricBase not in cls.__mro__:
            msg = f"Class {cls.__name__} must inherit from MetricBase to automatically "
            raise TypeError(msg)


class EvaluateMetrics:
    """
    Runs a list of actions in an RL environment and evaluates metrics after each step.

    Input:
    env_config: To build an own environment for evaluation.
    actions: The actions an agent has taken, which should be evaluated.
    index: The profile index, the environment should be reset to (default: 0)
    """

    def __init__(
        self,
        env_config: dict,
        actions: Sequence[int | np.integer],
        index: int = 0,
    ) -> None:
        self.env = PPTopoGym(env_config)
        self.metrics = MetricContainer()
        self.actions = actions
        self.index = index

    def _retrieve_all_metrics(self) -> list[FloatMetric]:
        """Retrieve all FloatMetric functions to be evaluated."""

        def all_subclasses(cls: type[MetricContainer]) -> set[type[MetricContainer]]:
            return set(cls.__subclasses__()).union(
                [s for c in cls.__subclasses__() for s in all_subclasses(c)],
            )

        def method_from_class(cls: type[MetricContainer]) -> list[str]:
            return [func_ for func_ in dir(cls) if not func_.startswith("__")]

        all_subclasses_list = all_subclasses(self.metrics.__class__)
        method_dict_name_class = {}
        for subclass in all_subclasses_list:
            for func in method_from_class(subclass):
                method_dict_name_class[func] = subclass
        self.all_methods_to_eval = []
        for method, class_ in method_dict_name_class.items():
            method_callable = getattr(class_, method)
            if isinstance(method_callable, FloatMetric):
                self.all_methods_to_eval.append(method_callable)
        return self.all_methods_to_eval

    def _evaluate_metrics(self, metrics_to_eval: list[FloatMetric] | None) -> None:
        """
        Evaluate all FloatMetric descriptors attached to the metrics instance.

        This function is here because it is only needed for this class.
        """
        if metrics_to_eval is None:
            metrics_to_eval = self._retrieve_all_metrics()
        for method in metrics_to_eval:
            if isinstance(method, FloatMetric):
                method.evaluate(self.metrics)

    def run(self, metrics_to_eval: list[FloatMetric] | None = None) -> None:
        """
        Run all FloatMetric descriptors attached to the metrics instance.

        The results are stored in the functions descriptor class.
        They can be seen with print(<class-instance>).

        Parameters
        ----------
        metrics_to_eval: list of floatmetrics (default: all)
        """
        self.env.reset(options = {"index": self.index})
        for action in self.actions:
            obs, reward, terminated, truncated, info = self.env.step(action)

            # Share state with metrics
            self.metrics.net = self.env.net
            self.metrics.last_action = action
            self.metrics.last_reward = reward
            self.metrics.last_obs = obs
            self.metrics.last_info = info

            # Evaluate all metrics
            self._evaluate_metrics(metrics_to_eval)

    def __str__(self) -> str:
        """Collect all metrics that have been evaluated."""
        result: list[str] = []

        # Find all metric descriptors in the class hierarchy
        metrics = self._retrieve_all_metrics()
        # Check which metrics have values in the instance
        result = []
        for method in metrics:
            if isinstance(method, FloatMetric):
                result.append(method.__str__())
                result.append(method.format_with_instance(self.metrics))
        if not result:
            return "No metrics were evaluated"
        return "\n".join(result)

    def gather_results(self) -> dict[str, dict[str, Any]]:
        """
        Gather all metric results into a list of dictionaries.

        Each dictionary contains:
        - 'name': The name of the metric function
        - 'value': The current value of the metric (without recalculating)
        - 'cache': The current cache state for the metric

        Returns
        -------
            dict[dict[str, Any]]: Dictionary of metric information
            dict[dict[str, Any]]: List of dictionaries containing metric information
        """
        results: dict = {}

        # Find all FloatMetric descriptors in the class hierarchy
        metrics = self._retrieve_all_metrics()
        for attr in metrics:
            if isinstance(attr, FloatMetric):
                # Get the current value without calculation
                name = attr.get_name()
                value = self.metrics.__dict__.get(name, "Not Evaluated")

                # Get the current cache
                cache = self.metrics.metric_state.get(name, {})

                # Add to results
                results[name] = {"value": value, "cache": cache}
        return results


def make_metric_container(name: str, metric_funcs: dict) -> type:
    """
    Dynamically create a MetricContainer subclass with given metric functions.

    This showcases how to create new metrics, but might not work perfectly!
    Args:
        name (str): Name of the class.
        metric_funcs (dict): Dict of metric_name -> function(instance, cache) -> float

    Returns
    -------
        A subclass of MetricContainer with added FloatMetric descriptors.
    """
    if not metric_funcs:
        msg = "No metric functions provided"
        raise ValueError(msg)

    class_dict = {}
    for metric_name, func in metric_funcs.items():
        if not callable(func):
            msg = f"Metric '{metric_name}' is not callable"
            raise TypeError(msg)
        descriptor = FloatMetric(func)
        class_dict[metric_name] = descriptor

    # Create the class with both MetricContainer and MetricBase
    new_class = type(name, (MetricContainer, MetricBase), class_dict)

    # Manually call __set_name__ for each FloatMetric descriptor
    for metric_name, descriptor in class_dict.items():
        if isinstance(descriptor, FloatMetric):
            descriptor.__set_name__(new_class, metric_name)

    return new_class
