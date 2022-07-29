from functools import reduce

from multi_taxi import single_taxi_v0, wrappers
from multi_taxi.world.entities import PASSENGER_NOT_IN_TAXI
from multi_taxi.world.domain_map import DomainMap

from rmrl.context.multitask_env import MultiTaskWrapper


def fixed_entities_env(initial_task=None, change_task_on_reset=False, **env_kwargs):
    env = single_taxi_v0.gym_env(**env_kwargs)
    env = FixedLocsWrapper(env, initial_task=initial_task, change_task_on_reset=change_task_on_reset)

    return env


def changing_map_env(initial_task=None, change_task_on_reset=False, **env_kwargs):
    env = single_taxi_v0.gym_env(**env_kwargs, domain_map=EMPTY_MAP)
    env = ChangeMapWrapper(env, initial_task=initial_task, change_task_on_reset=change_task_on_reset)

    return env


class FixedLocsWrapper(MultiTaskWrapper):
    def __init__(self, env, initial_task=None, change_task_on_reset=True):
        super().__init__(env, initial_task, change_task_on_reset)
        taxi_loc, passenger_loc, passenger_dst = self.task

        # set constant locations and destinations wrappers
        # this will allow us to control taxi locations and passenger locations and destinations on the fly.
        self.fixed_env = wrappers.FixedTaxiStartLocationsWrapper(self.env, *taxi_loc)
        self.fixed_env = wrappers.FixedPassengerStartLocationsWrapper(self.fixed_env, *passenger_loc)
        self.fixed_env = wrappers.FixedPassengerDestinationsWrapper(self.fixed_env, *passenger_dst)

    def sample_task(self, n):
        fixed_locs = []
        for _ in range(n):
            # call private method. BAD PRACTICE
            taxis = self.unwrapped._MultiTaxiEnv__random_taxis()
            passengers = self.unwrapped._MultiTaxiEnv__random_passengers()

            # concat all locations and all destinations
            taxi_locs = reduce(lambda locs, t: locs + t.location, taxis, tuple())
            passenger_locs = reduce(lambda locs, p: locs + p.location, passengers, tuple())

            if self.unwrapped.pickup_only:
                passenger_dsts = tuple()
            else:
                passenger_dsts = reduce(lambda dsts, p: dsts + p.destination, passengers, tuple())

            # save locations and destinations
            fixed_locs.append((taxi_locs, passenger_locs, passenger_dsts))

        return fixed_locs

    def set_task(self, task):
        # override `set_task to update env wrappers' locations
        # new task will only take effect on reset due to uncertainty regarding reset

        # set new task property
        super(FixedLocsWrapper, self).set_task(task)

        # set wrappers for deterministic locations on reset
        self.__set_wrappers(task)

        # set locations in environment
        # we do this to enable changing the environment in the middle of an episode
        # self.__set_locations()

    def reset(self, **kwargs):
        # override `reset` task setting and for efficiency

        # want to set the task only for the wrappers, which will take effect on reset
        # no need to set the locations as well (as done in `set_state`)
        if self.change_task_on_reset:
            new_task = self.sample_task(1)[0]
            self.__set_wrappers(new_task)  # only set wrappers

        # use wrapped env for reset to set the entities in the right place
        # wrapper returns a dictionary for one taxi
        return self.fixed_env.reset(**kwargs)

    def __set_wrappers(self, task):
        taxi_loc, passenger_loc, passenger_dst = task

        # outer wrapper controls destinations
        # set locations in destinations wrapper
        self.fixed_env.locs = [passenger_dst]

        # next wrapper controls passenger locations
        # set locations in locations wrapper
        self.fixed_env.env.locs = [passenger_loc]

        # next next wrapper controls taxi locations
        # set locations in locations wrapper
        self.fixed_env.env.env.locs = [taxi_loc]

    def __set_locations(self):
        # outer wrapper controls destinations
        # set destinations using dst wrapper methods
        self.fixed_env.set_locations()

        # next wrapper controls passenger locations
        # set locations using loc wrapper method
        self.fixed_env.env.set_locations()

        # next next wrapper controls taxi locations
        # set locations using loc wrapper method
        self.fixed_env.env.env.set_locations()

        # remove passengers from carrying taxis
        s = self.fixed_env.state()
        for p in s.passengers:
            p.carrying_taxi = PASSENGER_NOT_IN_TAXI
        for t in s.taxis:
            t.passengers = set()
        self.fixed_env.unwrapped.set_state(s)


EMPTY_MAP = [
    "+-----------------------+",
    "| : : : : : : : : : : : |",
    "| : : : : : : : : : : : |",
    "| : : : : : : : : : : : |",
    "| : : : : : : : : : : : |",
    "| : : : : : : : : : : : |",
    "| : : : : : : : : : : : |",
    "| : : : : : : : : : : : |",
    "+-----------------------+",
]
WALL_LOCS = [(i, j) for i in range(len(EMPTY_MAP)) for j in range(len(EMPTY_MAP[0])) if EMPTY_MAP[i][j] == ':']

class ChangeMapWrapper(MultiTaskWrapper):
    def sample_task(self, n):
        # sample number of walls to add
        num_locs = len(WALL_LOCS)

        wall_pos_list = []
        for _ in range(n):
            num_locs_to_sample = self._task_np_random.integers(0, num_locs)

            # sample wall positions
            walls_idx = self._task_np_random.choice(range(num_locs), num_locs_to_sample, replace=False)
            walls_pos = [WALL_LOCS[idx] for idx in walls_idx]

            # assert walls do not make some areas unreachable
            for j in range(1, len(EMPTY_MAP[0]) - 1):
                # check if entire column is walls (i.e. blocking)
                column_locs = [(i, j) for i in range(1, len(EMPTY_MAP) - 1)]
                if all(loc in walls_pos for loc in column_locs):
                    # column is full of walls
                    # choose one to remove
                    pos_idx_to_remove = self._task_np_random.choice(range(len(column_locs)))
                    wall_loc_to_remove = column_locs[pos_idx_to_remove]
                    walls_pos.remove(wall_loc_to_remove)

            # save sample
            wall_pos_list.append(tuple(walls_pos))

        return wall_pos_list

    def set_task(self, task):
        super().set_task(task)

        map_arr = [[v for v in row] for row in EMPTY_MAP]
        for i, j in task:
            map_arr[i][j] = '|'

        self.unwrapped.domain_map = DomainMap([''.join(row) for row in map_arr])
