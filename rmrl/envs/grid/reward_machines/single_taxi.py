from itertools import product

import numpy as np

from ....reward_machines.reward_machine import *
from multi_taxi.world.domain_map import DomainMap
from multi_taxi.utils.types import SymbolicObservation as sym_obs


class TaxiEnvRM(RewardMachine):
    def _init(self, goal_state_reward=1, grid_resolution=None, fuel_resolution=None):
        # get env info
        self.dm = self.env.unwrapped.domain_map
        self.num_passengers = self.env.unwrapped.num_passengers
        self.pickup_only = self.env.unwrapped.pickup_only  # check "pickup only" setting where there is no dropoff
        self.pickup_order = self.env.unwrapped.pickup_order
        self.dropoff_order = self.env.unwrapped.dropoff_order

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
            (r, c): (r // sector_res_h) * num_sector_cols + (c // sector_res_w)
            for r, c in self.dm.iter_free_locations()
        }
        self.sec_to_locs = {i: [] for i in range(self.num_sectors)}
        [self.sec_to_locs[self.loc_to_sec[loc]].append(loc) for loc in self.loc_to_sec]

    def _delta(self) -> TransitionMap:
        # output transitions dict
        delta = {}

        # the number of passenger info propositions
        # - one for each passenger picked up. true iff corresponding passenger was picked up
        # - one for each passenger dropped off. true iff corresponding passenger has arrived at the destination. only
        num_passenger_statuses = self.num_passengers + self.num_passengers * (not self.pickup_only)

        # the number of propositions is the number of sectors and statuses combined
        num_props = self.num_sectors + num_passenger_statuses

        # a container for single proposition vectors
        single_props = np.eye(num_props)

        # iterate all possible passenger statuses
        for status in product([0, 1], repeat=num_passenger_statuses):
            if all(status):
                continue  # this is a goal status
            if not self.pickup_only and any(status[i + 1] and not status[i] for i in range(0, 2, len(status))):
                continue  # this is an invalid status of dropoff before pickup

            # pad status with zeros in the sector propositions
            status_prop = np.concatenate([np.zeros(self.num_sectors), status])

            # iterate all sectors at current status
            for sec in range(self.num_sectors):
                # get proposition vector for sector at current passenger status
                sec_prop = single_props[self.__get_sector_prop_idx(sec)] + status_prop

                # self edge in all sectors (e.g., bad pickup / dropoff)
                delta.setdefault(tuple(sec_prop), {})[tuple(sec_prop)] = 0

                # add edges to adjacent sectors with same status
                for other_sec in range(self.num_sectors):
                    if self.__adjacent_secs(sec, other_sec):
                        # proposition at current status of adjacent sector
                        other_sec_prop = single_props[self.__get_sector_prop_idx(other_sec)] + status_prop

                        # add edge
                        delta.setdefault(tuple(sec_prop), {})[tuple(other_sec_prop)] = 0

                # add edge to possible status changes for passengers
                for i in range(self.num_passengers):
                    p = self.env.unwrapped.state().passengers[i]
                    passenger_pickup_status = sec_prop[self.__get_pickup_prop_idx(i)]

                    # add pickup status change only if not picked up
                    if passenger_pickup_status == 0:
                        passenger_sec = self.loc_to_sec[p.location]  # get passenger section

                        # check that preceding passengers in the pickup order have already been picked up
                        p_order = self.pickup_order.index(p.id) if p.id in self.pickup_order else None
                        if p_order is None:  # not in order, can always be picked up
                            preceding_picked_up = True
                        else:  # is ordered, check preceding passengers' pickup statuses
                            preceding = self.pickup_order[:p_order]
                            preceding_picked_up = all(
                                sec_prop[self.__get_pickup_prop_idx(p_id)] == 1
                                for p_id in preceding
                            )

                        # if the passenger is in the taxi section and pickup ordering allows, it can be picked up
                        if passenger_sec == sec and preceding_picked_up:
                            passenger_picked_up_prop = single_props[self.__get_pickup_prop_idx(i)]
                            pickup_sec_prop = sec_prop + passenger_picked_up_prop

                            # add pickup edge with reward according to goal status
                            is_goal_status = np.all(pickup_sec_prop[self.num_sectors:])
                            delta.setdefault(tuple(sec_prop), {})[tuple(pickup_sec_prop)] = (self.goal_state_reward
                                                                                             if is_goal_status
                                                                                             else 0)

                    # add dropoff status change only if picked up and not dropped off
                    elif not self.pickup_only and sec_prop[self.__get_dropoff_prop_idx(i)] == 0:
                        dst_sec = self.loc_to_sec[p.destination]

                        # check that preceding passengers in the dropoff order have already been dropped off
                        p_order = self.dropoff_order.index(p.id) if p.id in self.dropoff_order else None
                        if p_order is None:  # not in order, can always be picked up
                            preceding_dropped_off = True
                        else:  # is ordered, check preceding passengers' pickup statuses
                            preceding = self.dropoff_order[:p_order]
                            preceding_dropped_off = all(
                                sec_prop[self.__get_dropoff_prop_idx(p_id)] == 1
                                for p_id in preceding
                            )

                        # if the destination is in the taxi section, it can be dropped off
                        if dst_sec == sec and preceding_dropped_off:
                            passenger_dropped_off_prop = single_props[self.__get_dropoff_prop_idx(i)]
                            dropoff_sec_prop = sec_prop + passenger_dropped_off_prop

                            # add dropoff edge with reward according to goal status
                            is_goal_status = np.all(dropoff_sec_prop[self.num_sectors:])
                            delta.setdefault(tuple(sec_prop), {})[tuple(dropoff_sec_prop)] = (self.goal_state_reward
                                                                                              if is_goal_status
                                                                                              else 0)

        return delta

    def L(self, s):
        # using domain knowledge on state
        s = self.env.state()

        props = np.zeros(self.num_propositions,)

        # current taxi sector
        taxi_loc = s.taxis[0].location  # assuming single agent
        taxi_sector = self.loc_to_sec[taxi_loc]
        props[self.__get_sector_prop_idx(taxi_sector)] = 1

        # check if taxi picked up passenger
        for p in s.passengers:
            if p.in_taxi or p.arrived:
                props[self.__get_pickup_prop_idx(p.id)] = 1
            if not self.pickup_only and p.arrived:
                props[self.__get_dropoff_prop_idx(p.id)] = 1

        return tuple(props)

    def __get_sector_prop_idx(self, sector_num):
        return sector_num

    def __get_pickup_prop_idx(self, pass_num):
        return self.num_sectors + pass_num * (1 if self.pickup_only else 2)

    def __get_dropoff_prop_idx(self, pass_num):
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
