from pandapower_env.metrics.metric_utils import (
    EvaluateMetrics,
    FloatMetric,
    MetricContainer,
    make_metric_container,
)


def test_float_metric_initialization() -> None:
    def dummy_func(instance, cache: dict) -> float:  # noqa: ARG001
        return 42.0

    metric = FloatMetric(dummy_func)
    assert metric.func == dummy_func
    assert metric._name is None


def test_float_metric_set_name() -> None:
    def dummy_func(instance, cache) -> float:  # noqa: ARG001
        return 42.0

    metric = FloatMetric(dummy_func)
    metric.__set_name__(MetricContainer, "test_metric")
    assert metric._name == "test_metric"


def test_float_metric_get_and_set() -> None:
    def dummy_func(instance, cache) -> float:  # noqa: ARG001
        return 42.0

    metric = FloatMetric(dummy_func)
    metric.__set_name__(MetricContainer, "test_metric")

    container = MetricContainer()
    set_value = 10
    metric.__set__(container, set_value)
    assert metric.__get__(container) == set_value


def test_float_metric_evaluate() -> None:
    def dummy_func(instance, cache) -> float:  # noqa: ARG001
        return 42.0

    metric = FloatMetric(dummy_func)
    metric.__set_name__(MetricContainer, "test_metric")

    container = MetricContainer()
    result = metric.evaluate(container)
    set_value = 42
    assert result == set_value
    assert container.__dict__["test_metric"] == set_value


def test_make_metric_container() -> None:
    def metric_func_1(instance, cache) -> float:  # noqa: ARG001
        return 1.0

    def metric_func_2(instance, cache) -> float:  # noqa: ARG001
        return 2.0

    CustomMetricContainer = make_metric_container(  # noqa: N806
        "CustomMetricContainer",
        {"metric_1": metric_func_1, "metric_2": metric_func_2},
    )
    assert isinstance(CustomMetricContainer.__dict__["metric_1"], FloatMetric)
    assert isinstance(CustomMetricContainer.__dict__["metric_2"], FloatMetric)


def test_evaluatemetrics(env_config: dict) -> None:
    """
    Test the AllMetrics class.

    :param env_config: Environment configuration.
    :type env_config: dict
    """
    log_actions = [0, 0, 0]
    eval_ = EvaluateMetrics(env_config, log_actions)
    eval_.run()
    assert eval_.index == 0
    assert eval_.env.index == len(log_actions)
    start_index = 1
    eval_2 = EvaluateMetrics(env_config, log_actions, start_index)
    eval_2.run()
    assert eval_2.index == start_index
    assert eval_2.env.index == (len(log_actions)+start_index)
    # a lot is tested in test_evaluation_metrics.py
