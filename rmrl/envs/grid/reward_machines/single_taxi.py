import networkx as nx

from ....reward_machines.reward_machine import *
from multi_taxi.world.domain_map import DomainMap


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
        props = self.L(None, None, s0)

        # get max idx proposition as state id
        max_prop = max(np.where(self.prop_list_to_bitmap(props))[0])

        return max_prop

    def delta(self, u, props):
        next_u = max(np.where(self.prop_list_to_bitmap(props))[0])
        return next_u, self.G[u][next_u][REWARD_ATTR]

    def L(self, s, a, s_prime):
        # start with current sector
        taxi_loc = tuple(s_prime[self.taxi_loc_idx])
        taxi_sector = self.loc_to_sec[taxi_loc]
        props = [self.P[taxi_sector]]

        # continue to check if on passenger or picked up passenger
        if np.all(s_prime[self.taxi_loc_idx] == s_prime[self.pass_loc_idx]):
            props.append(self.P[self.num_sectors])

        if s_prime[self.pickup_ind_idx]:
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

