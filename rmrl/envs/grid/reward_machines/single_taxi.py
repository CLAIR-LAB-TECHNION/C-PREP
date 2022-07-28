import numpy as np

from ....reward_machines.reward_machine import *
from multi_taxi.world.domain_map import DomainMap
from multi_taxi.utils.types import SymbolicObservation as sym_obs


class TaxiEnvRM(RewardMachine):
    def _init(self, goal_state_reward=1, grid_resolution=None, fuel_resolution=None):
        # get env info
        self.dm = self.env.unwrapped.domain_map
        self.num_passengers = self.env.unwrapped.num_passengers

        # goal state info
        self.goal_state_reward = goal_state_reward

        # set default resolution info
        self.grid_resolution = grid_resolution or (self.dm.map_height, self.dm.map_width)
        self.fuel_resolution = fuel_resolution or 1  # TODO handle fuel

        # check resolution info
        sector_res_h, sector_res_w = self.grid_resolution
        assert sector_res_h <= self.dm.map_height
        assert sector_res_w <= self.dm.map_width

        # set sector helper values and mappings
        num_sector_rows = (self.dm.map_height // sector_res_h) + (self.dm.map_height % sector_res_h != 0)
        num_sector_cols = (self.dm.map_width // sector_res_w) + (self.dm.map_width % sector_res_w != 0)
        self.num_sectors = num_sector_rows * num_sector_cols
        self.loc_to_sec = {
            (r, c): (r // sector_res_h) * num_sector_rows + (c // sector_res_w)
            for r, c in self.dm.iter_free_locations()
        }
        self.sec_to_locs = {i: [] for i in range(self.num_sectors)}
        [self.sec_to_locs[self.loc_to_sec[loc]].append(loc) for loc in self.loc_to_sec]

        # important observation info
        obs_meanings = self.env.unwrapped.get_observation_meanings()
        self.taxi_loc_idx = [obs_meanings.index(sym_obs.LOCATION_ROW.value),
                             obs_meanings.index(sym_obs.LOCATION_COL.value)]

        # TODO enable multiple passengers
        self.pass_loc_idx = [obs_meanings.index(sym_obs.PASSENGER_LOCATION_ROW.value.format(index=0)),
                             obs_meanings.index(sym_obs.PASSENGER_LOCATION_COL.value.format(index=0))]
        self.picked_up_idx = obs_meanings.index(sym_obs.PASSENGER_PICKED_UP.value.format(index=0))

        # self.pass_loc_idx = []
        # try:
        #     for i in count():
        #         pass_row_obs = sym_obs.PASSENGER_LOCATION_ROW.value.format(index=i)
        #         pass_col_obs = sym_obs.PASSENGER_LOCATION_COL.value.format(index=i)
        #         self.pass_loc_idx.append((obs_meanings.index(pass_row_obs),
        #                                   obs_meanings.index(pass_col_obs)))
        # except ValueError:
        #     pass

    def _delta(self) -> TransitionMap:
        delta = {}

        num_props = self.num_sectors + self.num_passengers * 2
        single_props = [tuple(v) for v in np.eye(num_props)]  # make tuple for hashing

        # passenger location
        # TODO multiple passenger locations
        _, passenger_loc, _ = self.env.task
        passenger_sec = self.loc_to_sec[passenger_loc]
        on_passenger_prop = tuple(np.array(single_props[self.__get_sector_prop_idx(passenger_sec)]) +
                                  np.array(single_props[self.__get_on_pass_prop_idx(0)]))
        pickup_prop = tuple(np.array(single_props[self.__get_sector_prop_idx(passenger_sec)]) +
                            np.array(single_props[self.__get_on_pass_prop_idx(0)]) +
                            np.array(single_props[self.__get_pickup_prop_idx(0)]))

        # sectors
        # iterate all sectors
        for sec in range(self.num_sectors):
            sec_prop = single_props[self.__get_sector_prop_idx(sec)]

            # check adjacency to other sectors
            for other_sec in range(self.num_sectors):
                if self.__adjacent_secs(sec, other_sec):
                    other_sec_prop = single_props[self.__get_sector_prop_idx(other_sec)]
                    delta.setdefault(sec_prop, {})[other_sec_prop] = 0

            # check adjacency to passenger
            if self.__adjacent_sec_to_loc(sec, passenger_loc):
                delta.setdefault(sec_prop, {})[on_passenger_prop] = 0
                delta.setdefault(on_passenger_prop, {})[sec_prop] = 0

        # stuck on passenger
        delta.setdefault(on_passenger_prop, {})[on_passenger_prop] = 0

        # final pickup
        delta.setdefault(on_passenger_prop, {})[pickup_prop] = self.goal_state_reward

        return delta

    def L(self, s):
        props = np.zeros(self.num_propositions,)

        # start with current sector
        taxi_loc = tuple(s[self.taxi_loc_idx])
        taxi_sector = self.loc_to_sec[taxi_loc]
        props[self.__get_sector_prop_idx(taxi_sector)] = 1

        # TODO handle multiple passengers
        # continue to check if on passenger or picked up passenger
        if np.all(s[self.taxi_loc_idx] == s[self.pass_loc_idx]):
            props[self.__get_on_pass_prop_idx(0)] = 1

        if s[self.picked_up_idx]:
            props[self.__get_pickup_prop_idx(0)] = 1

        return tuple(props)

    def __get_sector_prop_idx(self, sector_num):
        return sector_num

    def __get_on_pass_prop_idx(self, pass_num):
        return self.num_sectors + pass_num * 2

    def __get_pickup_prop_idx(self, pass_num):
        return self.num_sectors + pass_num * 2 + 1

    def __adjacent_locs(self, l1, l2):
        # unpack rows and columns
        r1, c1 = l1
        r2, c2 = l2

        # check 1D adjacency
        same_row = r1 == r2
        same_col = c1 == c2
        adj_row = abs(r1 - r2) == 1
        adj_col = abs(c1 - c2) == 1

        # check 2d adjacency
        adjacent_cells = (adj_row and same_col) or (adj_col and same_row)

        # check transition
        transitionable = not self.dm.hit_obstacle(l1, l2)

        # adjacent locs iff cells are adjacent and can transition from one to the other
        return adjacent_cells and transitionable

    def __adjacent_loc_to_sec(self, l, sec):
        return any(self.__adjacent_locs(l, ll) for ll in self.sec_to_locs[sec])

    def __adjacent_sec_to_loc(self, sec, l):
        return any(self.__adjacent_locs(ll, l) for ll in self.sec_to_locs[sec])

    def __adjacent_secs(self, sec1, sec2):
        return any(self.__adjacent_loc_to_sec(l1, sec2) for l1 in self.sec_to_locs[sec1])


class SingleFixedPassengerHighResRM(RewardMachine):
    def _init(self, taxi_loc_idx: list, pass_loc_idx: list, pickup_ind_idx: int, resolution: tuple, dm: DomainMap,
              use_prop_vector_attr=False):
        self.taxi_loc_idx = taxi_loc_idx
        self.pass_loc_idx = pass_loc_idx
        self.pickup_ind_idx = pickup_ind_idx
        self.use_prop_vector_attr = use_prop_vector_attr

        # divide all locations into sectors according to the resolution
        self.res_y, self.res_x = resolution
        self.num_sectors = self.res_x * self.res_y
        cell_res_x = dm.map_width // self.res_x + (dm.map_width % self.res_x != 0)
        cell_res_y = dm.map_height // self.res_y + (dm.map_height % self.res_y != 0)
        self.loc_to_sec = {
            (r, c): (r // cell_res_y) * self.res_x + (c // cell_res_x)
            for r, c in dm.iter_free_locations()
        }
        self.sec_to_locs = {i: [] for i in range(self.num_sectors)}
        [self.sec_to_locs[self.loc_to_sec[loc]].append(loc) for loc in self.loc_to_sec]

        # find edge connections
        self.sector_connections = {}
        for sec1, locs1 in self.sec_to_locs.items():
            self.sector_connections[sec1] = set()
            for sec2, locs2 in self.sec_to_locs.items():
                if any(self.__adjacent(l1, l2) and not dm.hit_obstacle(l1, l2)
                       for l1 in locs1 for l2 in locs2):
                    self.sector_connections[sec1].add(sec2)

        self.__dm = dm  # save dm for task choosing

        self.__passenger_sector = None

    @staticmethod
    def __adjacent(l1, l2):
        return (abs(l1[0] - l2[0]) == 1 and l1[1] == l2[1]) or (abs(l1[1] - l2[1]) == 1 and l1[0] == l2[0])

    @property
    def P(self):
        return ([f'in sector {i}' for i in range(self.res_x * self.res_y)] +
                ['on pass', 'picked up'])

    def u0(self, s0) -> int:
        # labeling function only needs the current state
        props = self.L(s0)

        # get max idx proposition as state id
        max_prop = max(np.where(self.prop_list_to_bitmap(props))[0])

        return max_prop

    def delta(self, u, props):
        next_u = max(np.where(self.prop_list_to_bitmap(props))[0])
        return next_u, self.G[u][next_u][REWARD_ATTR]

    def L(self, s):
        # start with current sector
        taxi_loc = tuple(s[self.taxi_loc_idx])
        taxi_sector = self.loc_to_sec[taxi_loc]
        props = [self.P[taxi_sector]]

        # continue to check if on passenger or picked up passenger
        if np.all(s[self.taxi_loc_idx] == s[self.pass_loc_idx]):
            props.append(self.P[self.num_sectors])

        if s[self.pickup_ind_idx]:
            props.append(self.P[self.num_sectors + 1])

        return props

    def new_task(self, task):
        self.reset_sm()

        # expecting single passenger, no destination
        _, passenger_loc, _ = task

        for sec, locs in self.sec_to_locs.items():
            if any(self.__adjacent(loc, passenger_loc) and not self.__dm.hit_obstacle(loc, passenger_loc)
                   for loc in locs):
                # at `num_sectors` we find the "on pass" symbol
                self.G.add_edge(sec, self.num_sectors, **{REWARD_ATTR: 0})
                self.G.add_edge(self.num_sectors, sec, **{REWARD_ATTR: 0})

        self.__passenger_sector = self.loc_to_sec[passenger_loc]

        if self.use_prop_vector_attr:
            self._add_prop_vector()

    def create_state_machine(self):
        # adds nodes and edges for sector states
        sm = nx.DiGraph(self.sector_connections)

        # adds nodes and edges for "on passenger" and "picked up" states
        sm.add_edge(self.num_sectors, self.num_sectors)  # self loop if staying on the passenger
        sm.add_edge(self.num_sectors, self.num_sectors + 1, **{REWARD_ATTR: 1000})

        return sm

    def _add_prop_vector(self):
        for u, u_data in self.G.nodes(data=True):
            pv = [0] * self.num_propositions
            pv[u] = 1

            if u >= self.num_sectors:  # at passenger. add containing sector
                pv[self.__passenger_sector] = 1

            u_data[PROPS_VECTOR_ATTR] = pv

