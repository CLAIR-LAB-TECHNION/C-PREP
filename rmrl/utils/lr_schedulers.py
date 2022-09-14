from abc import ABC, abstractmethod
from typing import Dict

from rmrl.utils.misc import repr_all_public


class LearningRateSchedule(ABC):
    @abstractmethod
    def __call__(self, progress_remaining: float) -> float:
        pass

    def __repr__(self):
        return repr_all_public(self)

    def __eq__(self, other):
        return(isinstance(other, self.__class__) and
               vars(self) == vars(other))

    def __hash__(self):
        return hash(tuple(vars(self).items()))


class LinearSchedule(LearningRateSchedule):
    def __init__(self, initial_value: float):
        self.initial_value = initial_value

    def __call__(self, progress_remaining: float) -> float:
        return progress_remaining * self.initial_value


class StepSchedule(LearningRateSchedule):
    def __init__(self, initial_value: float, steps_map: Dict[float, float]):
        self.initial_value = initial_value
        self.steps_map = steps_map
        self.__step_points = sorted(steps_map.keys(), reverse=True)

    def __call__(self, progress_remaining: float) -> float:
        prev_p = 1
        for p in self.__step_points:
            if p < progress_remaining:
                return self.steps_map.get(prev_p, self.initial_value)
            prev_p = p

        return self.steps_map[self.__step_points[-1]]
