from typing import Dict

from stable_baselines3.common.type_aliases import Schedule


def linear_schedule(initial_value: float) -> Schedule:
    """
    from:
    https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#learning-rate-schedule
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


def step_schedule(initial_value: float, steps_map: Dict[float, float]) -> Schedule:
    cur_value = [initial_value]  # use immutable type for pointer capabilities
    step_points = sorted(steps_map.keys())

    def func(progress_remaining: float) -> float:
        if step_points and step_points[-1] > progress_remaining:
            cur_value[0] = steps_map[step_points.pop()]

        return cur_value[0]

    return func
