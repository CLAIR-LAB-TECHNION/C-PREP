from abc import ABC, abstractmethod
from typing import Dict

from stable_baselines3.common.type_aliases import Schedule


class LearningRateSchedule(ABC):
    @abstractmethod
    def __call__(self, progress_remaining: float) -> float:
        pass

    def __repr__(self):
        vars_join = ', '.join(f'{k}={v}'
                              for k, v in vars(self).items()
                              if not k.startswith(f'_{self.__class__.__name__}'))
        return f'{self.__class__.__name__}({vars_join})'

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
