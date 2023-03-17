import numpy as np

from rmrl.reward_machines.reward_machine import *
from ..half_cheetah import MIN_VEL, MAX_VEL, MIN_X, MAX_X


class LapsRM(RewardMachine):

    def _init(self, goal_state_reward=1000):
        self.goal_state_reward = goal_state_reward

        # thresholds pruned after target threshold
        self.node_thresh = np.array(self.env.section_thresholds)[:self.env.section_thresholds.index(self.env.task) + 1]

        self.forward_nodes = list(map(tuple, np.tri(self.num_thresh, self.num_thresh + 1, k=-1)))
        self.backward_nodes = list(map(tuple, np.hstack([np.tri(self.num_thresh),
                                                         np.ones((self.num_thresh, 1))])))[::-1]

        self.nodes = self.forward_nodes + self.backward_nodes

        self._start_thresh = self.node_thresh[0]
        self._goal_thresh = self.node_thresh[-1]

        self.__thresholds_passed = 0

        self._prev_node = None

    @property
    def num_thresh(self):
        return len(self.node_thresh)

    def reset(self):
        self.__thresholds_passed = 0
        self._prev_node = None

    def _delta(self) -> TransitionMap:
        delta = {}

        # self loops for all nodes (remain in the same speed)
        for n in self.nodes:
            delta.setdefault(n, {})[n] = 0

        # neighbor positions
        for n1, n2 in zip(self.forward_nodes[:-1], self.forward_nodes[1:]):
            delta[n1][n2] = 0
        for n1, n2 in zip(self.backward_nodes[:-1], self.backward_nodes[1:]):
            delta[n1][n2] = 0

        # cross lap checkpoint
        delta[self.forward_nodes[-1]][self.backward_nodes[0]] = 0
        delta[self.backward_nodes[-1]][self.forward_nodes[0]] = self.goal_state_reward

        return delta

    def L(self, s):
        cur_pos = s[0]
        is_backward = s[-1]

        if is_backward:
            if cur_pos < self.node_thresh[self.__thresholds_passed]:
                self.__thresholds_passed -= 1
            node = self.backward_nodes[self.num_thresh - self.__thresholds_passed - 1]
        else:
            if cur_pos >= self.node_thresh[self.__thresholds_passed]:
                self.__thresholds_passed += 1
            node = self.forward_nodes[self.__thresholds_passed]

        # print(f'{cur_pos=}')
        # print(f'{is_backward=}')
        # print(f'{self._prev_node=}')
        # print(f'{node}')
        # self._prev_node = node

        return node


class LocationRM(RewardMachine):

    def _init(self, goal_state_reward=1):
        self.goal_state_reward = goal_state_reward

        goal_cutoff_idx = self.env.goal_list.index(self.env.task) + 1

        self.node_pos = np.array(self.env.goal_list)[:goal_cutoff_idx]
        self.nodes = list(map(tuple, np.eye(len(self.env.goal_list) + 2)))
        self.nodes = self.nodes[:goal_cutoff_idx + 1] + [self.nodes[-1]]

        self.section_nodes, self.term_node = self.nodes[:-1], self.nodes[-1]
        self.target_node = self.section_nodes[-1]

        self.__prev_state = None
        self.__thresholds_passed = 0
        self.__failed = False

    def reset(self):
        self.__prev_state = None
        self.__thresholds_passed = 0
        self.__failed = False

    def _delta(self) -> TransitionMap:
        delta = {}

        # self loops for all nodes (remain in the same speed)
        for n in self.section_nodes:
            delta.setdefault(n, {})[n] = self.__reward_for_node(n)

        # neighbor positions
        for n1, n2 in zip(self.section_nodes[:-1], self.section_nodes[1:]):
            delta[n1][n2] = self.__reward_for_node(n2)

        # left target section
        delta[self.target_node][self.term_node] = 0

        return delta

    def L(self, s):
        cur_pos = s[0]

        in_goal = self.__thresholds_passed == len(self.node_pos)
        threshold_crossed = self.__thresholds_passed < len(self.node_pos) and cur_pos >= self.node_pos[self.__thresholds_passed]
        threshold_receeded = self.__thresholds_passed > 0 and cur_pos < self.node_pos[self.__thresholds_passed - 1]

        if in_goal:
            if threshold_receeded or threshold_crossed:
                self.__failed = True
        elif threshold_crossed:
            self.__thresholds_passed += 1
            self.__failed = False

        if self.__failed:
            return self.nodes[-1]

        return self.nodes[self.__thresholds_passed]

    def __reward_for_node(self, n):
        return self.goal_state_reward if n == self.section_nodes[-1] else 0


# class LocationRM(RewardMachine):  # NON ONE-HOT SYMBOLS
#
#     def _init(self, goal_state_reward=1, pos_resolution=0.5):
#         self.goal_state_reward = goal_state_reward
#         self.pos_resolution = pos_resolution
#
#         self.sec_thresh = np.array(self.env.goal_list)
#         # self.nodes = list(map(tuple, np.eye(len(self.node_pos) + 2)))
#
#         self.all_section_nodes = list(map(tuple, np.tri(len(self.sec_thresh) + 1, len(self.sec_thresh) + 1, k=-1)))
#         self.section_nodes = self.all_section_nodes[:self.env.goal_list.index(self.env.task) + 2]
#         self.target_section_node = self.section_nodes[-1]
#         self.termination_node = tuple(np.eye(len(self.sec_thresh) + 1)[-1])
#
#         self.nodes = self.section_nodes + [self.termination_node]
#
#         # self.section_nodes, self.term_node = self.nodes[:-1], self.nodes[-1]
#         # self.target_section_node = self.section_nodes[-1]
#         # self.termination_node = self.nodes[-1]
#
#         self.__thresholds_passed = 0
#         self.__failed = False
#
#     @property
#     def num_sections(self):
#         return len(self.all_section_nodes)
#
#     @property
#     def num_sec_nodes(self):
#         return len(self.section_nodes)
#
#     def reset(self):
#         self.__thresholds_passed = 0
#         self.__failed = False
#
#     def _delta(self) -> TransitionMap:
#         delta = {}
#
#         # self loops for all nodes (remain in the same section)
#         for n in self.section_nodes:
#             delta.setdefault(n, {})[n] = self.__reward_for_node(n)
#
#         # neighbor positions
#         for n1, n2 in zip(self.section_nodes[:- 1], self.section_nodes[1:]):
#             delta[n1][n2] = self.__reward_for_node(n2)
#
#         # left target section
#         delta[self.target_section_node][self.termination_node] = 0
#
#         return delta
#
#     def L(self, s):
#         cur_pos = s[0]
#
#         # cross all section node thresholds to have visited the goal
#         in_goal = self.__thresholds_passed == self.num_sec_nodes - 1
#
#         # crossed next threshold or recede previous threshold to change section
#         threshold_crossed = (self.__thresholds_passed < self.num_sections and
#                              cur_pos >= self.sec_thresh[self.__thresholds_passed])
#         threshold_receded = (self.__thresholds_passed > 0 and
#                              cur_pos < self.sec_thresh[self.__thresholds_passed - 1])
#         section_changed = threshold_crossed or threshold_receded
#
#         if not in_goal and threshold_crossed:
#             self.__thresholds_passed += 1
#         elif in_goal and section_changed:
#             self.__failed = True
#
#         if self.__failed:
#             return self.termination_node
#
#         if self.__thresholds_passed >= len(self.section_nodes):
#             print(self.__thresholds_passed)
#             print(s)
#         return self.section_nodes[self.__thresholds_passed]
#
#     def __reward_for_node(self, n):
#         return self.goal_state_reward if n == self.target_section_node else 0


class VelocityRM(RewardMachine):

    def _init(self, goal_state_reward=1, vel_resolution=0.5):
        self.goal_vel_reward = goal_state_reward
        self.vel_resolution = vel_resolution

        self.__num_states = int(np.ceil((MAX_VEL - MIN_VEL) / self.vel_resolution)) + 1
        self.node_vels = np.linspace(start=MIN_VEL, stop=MAX_VEL, num=self.__num_states)
        # self.nodes = [
        #     tuple((((x > 0) & (self.node_vels > 0) & (self.node_vels <= x)) |
        #           ((x < 0) & (self.node_vels >= x) & (self.node_vels < 0)) |
        #           (self.node_vels == x)).astype(float))
        #     for x in self.node_vels
        # ]
        self.nodes = list(map(tuple, np.eye(len(self.node_vels))))

        self.nearest_node_to_goal = self.__closest_vel_node(self.env.task)

        self.__prev_state = None

    def _delta(self) -> TransitionMap:
        delta = {}

        # self loops for all nodes (remain in the same speed)
        for n in self.nodes:
            delta.setdefault(n, {})[n] = self.__reward_for_node(n)

        # neighbor velocities
        for n1, n2 in zip(self.nodes[:-1], self.nodes[1:]):
            delta[n1][n2] = 0  #self.__reward_for_node(n2)
            delta[n2][n1] = 0  #self.__reward_for_node(n1)

        return delta

    def L(self, s):
        next_state = self.__closest_vel_node(self.env.cur_vel, neighbor_of=None)  # self.__prev_state)
        self.__prev_state = next_state
        return next_state

    def __closest_vel_node(self, vel, neighbor_of=None):
        if neighbor_of is None:
            return self.nodes[np.argmin(np.abs(self.node_vels - vel)).squeeze()]

        neighbor_idx = self.nodes.index(neighbor_of)
        neighborhood_vels = self.node_vels[max(0, neighbor_idx - 1):min(len(self.node_vels), neighbor_idx + 2)]
        closest_neighbor_vel = neighborhood_vels[np.argmin(np.abs(neighborhood_vels - vel)).squeeze()]
        closest_neighbor = self.nodes[np.argwhere(self.node_vels == closest_neighbor_vel).squeeze()]

        # print(f'{vel=}')
        # print(f'{neighbor_of=}')
        # print(f'neighbor_speed={self.node_vels[np.argwhere(neighbor_of).squeeze()]}')
        # print(f'{neighbor_idx=}')
        # print(f'{neighborhood_vels=}')
        # print(f'{closest_neighbor_vel=}')
        # print(f'{closest_neighbor=}')

        return closest_neighbor

    def __reward_for_node(self, n):
        return self.goal_vel_reward if n == self.nearest_node_to_goal else 0

#     def __init__(self, goal_vel, goal_tolerance=1e-2, goal_reward=1):
#         G = self.create_state_machine(goal_vel, goal_tolerance, goal_reward, len(self.P))
#
#         super().__init__(state_machine=G, initial_state_id=4)
#
#         self.__goal_vel = goal_vel
#         self.__goal_tolerance = goal_tolerance
#
#         self.__tol_low = abs(goal_vel) - self.__goal_tolerance
#         self.__tol_hi = abs(goal_vel) + self.__goal_tolerance
#
#         self.__goal_reward = goal_reward
#
#     @property
#     def P(self):
#         return ['<=-100%', '-75%', '-50%', '-25%', '0', '25%', '50%', '75%', '<tolerance', '+-tolerance', '>tolerance']
#
#     def delta(self, u, props):
#         if not props:
#             return u, self.G[u][u][REWARD_ATTR]
#
#         props_bitmap = self.prop_list_to_bitmap(props)
#         nearest_node = np.where(props_bitmap)[0][0]
#         approx_vel = self.G.nodes[nearest_node][VEL_ATTR]
#
#         neighbors = self.G[u]
#         nearest_neighbor = min(neighbors, key=lambda nei: abs(approx_vel - self.G.nodes[nei][VEL_ATTR]))
#
#         return nearest_neighbor, neighbors[nearest_neighbor][REWARD_ATTR]
#
#     def L(self, s, a, s_prime):
#         vel = s_prime[0] - s[0]
#
#         nn, _ = min(self.G.nodes(data=True), key=lambda n_tup: abs(vel - n_tup[1][VEL_ATTR]))
#         # if nd[VEL_ATTR] == self.__goal_vel:
#         #     if not self.__tol_low <= abs(vel) <= self.__tol_hi:
#         #         nn -= 1
#
#         return [self.P[nn]]
#
#     @staticmethod
#     def create_state_machine(goal_vel, goal_tol, goal_reward, num_states):
#         sm = nx.DiGraph()
#
#         # set node labels
#         sm.add_nodes_from(range(num_states))
#
#         node_vels = np.linspace(start=-goal_vel, stop=goal_vel, num=num_states - 2)
#         neg = -1 if goal_vel < 0 else 1
#         node_vels = np.concatenate([node_vels[:-1],                    # -100% - 75%
#                                     [goal_vel - 2 * neg * goal_tol],   # < 100% - tolerance
#                                     node_vels[-1:],                    # 100% +- tolerance
#                                     [goal_vel + 2 * neg * goal_tol]])  # > 100% + tolerance
#
#         # add node features
#         for (_, n_data), vel in zip(sm.nodes(data=True), node_vels):
#             n_data[VEL_ATTR] = vel
#
#         # forward edges
#         sm.add_edges_from([(i, j) for i, j in zip(range(num_states - 1), range(1, num_states))])
#
#         # backward edges
#         sm.add_edges_from([(i, j) for i, j in zip(range(1, num_states), range(num_states - 1))])
#
#         # self edges
#         sm.add_edges_from([(i, i) for i in range(num_states)])
#
#         # rewards
#         for u in sm.nodes:
#             if sm.nodes[u][VEL_ATTR] == goal_vel:
#                 sm[u][u][REWARD_ATTR] = goal_reward
#
#         # for u in sm.nodes:
#         #     for v in sm[u]:
#         #         if sm.nodes[v][VEL_ATTR] == goal_vel:
#         #             sm[u][v][REWARD_ATTR] = goal_reward
#         #         elif sm.nodes[u][VEL_ATTR] == goal_vel and u != v:
#         #             sm[u][v][REWARD_ATTR] = -goal_reward
#
#         return sm
#
#     def draw(self, layout=None, plot_figsize=None, **nxdraw_kwargs):
#         vel_labels = {i: f"{self.G.nodes[i][VEL_ATTR]:.2f}" for i in self.G.nodes}
#         fig_width = self.num_states + 5
#         colormap = ['red'] * self.G.number_of_nodes()
#
#         colormap[self.__goal_node] = '#66FF66'
#         colormap[self.__u0] = '#6666FF'
#
#         custom_nx_draw_kwargs = dict(with_labels=True,
#                                      labels=vel_labels,
#                                      node_size=1000,
#                                      node_color=colormap,
#                                      font_weight='bold',
#                                      connectionstyle='arc3, rad = 0.1')
#         custom_nx_draw_kwargs.update(nxdraw_kwargs)
#
#         super().draw(layout=layout or linear_layout,
#                      plot_figsize=plot_figsize or (fig_width, fig_width / 5),
#                      **custom_nx_draw_kwargs)
#
#     def __is_goal_node(self, u):
#         return self.G.nodes[u][VEL_ATTR] == self.__goal_vel
#
#     @property
#     def __goal_node(self):
#         return [n for n in self.G.nodes if self.__is_goal_node(n)][0]
#
#
# def linear_layout(G):
#     return {u: [i * 2, 0] for i, u in enumerate(G.nodes())}
