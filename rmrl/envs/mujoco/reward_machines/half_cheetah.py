import numpy as np

from rmrl.reward_machines.reward_machine import *


VEL_ATTR = 'vel'


class VelocityRM(RewardMachine):
    def __init__(self, goal_vel, goal_tolerance=1e-2, goal_reward=1):
        G = self.create_state_machine(goal_vel, goal_tolerance, goal_reward, len(self.P))

        super().__init__(state_machine=G, initial_state_id=4)

        self.__goal_vel = goal_vel
        self.__goal_tolerance = goal_tolerance

        self.__tol_low = abs(goal_vel) - self.__goal_tolerance
        self.__tol_hi = abs(goal_vel) + self.__goal_tolerance

        self.__goal_reward = goal_reward

    @property
    def P(self):
        return ['<=-100%', '-75%', '-50%', '-25%', '0', '25%', '50%', '75%', '<tolerance', '+-tolerance', '>tolerance']

    def delta(self, u, props):
        if not props:
            return u, self.G[u][u][REWARD_ATTR]

        props_bitmap = self.prop_list_to_bitmap(props)
        nearest_node = np.where(props_bitmap)[0][0]
        approx_vel = self.G.nodes[nearest_node][VEL_ATTR]

        neighbors = self.G[u]
        nearest_neighbor = min(neighbors, key=lambda nei: abs(approx_vel - self.G.nodes[nei][VEL_ATTR]))

        return nearest_neighbor, neighbors[nearest_neighbor][REWARD_ATTR]

    def L(self, s, a, s_prime):
        vel = s_prime[0] - s[0]

        nn, _ = min(self.G.nodes(data=True), key=lambda n_tup: abs(vel - n_tup[1][VEL_ATTR]))
        # if nd[VEL_ATTR] == self.__goal_vel:
        #     if not self.__tol_low <= abs(vel) <= self.__tol_hi:
        #         nn -= 1

        return [self.P[nn]]

    @staticmethod
    def create_state_machine(goal_vel, goal_tol, goal_reward, num_states):
        sm = nx.DiGraph()

        # set node labels
        sm.add_nodes_from(range(num_states))

        node_vels = np.linspace(start=-goal_vel, stop=goal_vel, num=num_states - 2)
        neg = -1 if goal_vel < 0 else 1
        node_vels = np.concatenate([node_vels[:-1],                    # -100% - 75%
                                    [goal_vel - 2 * neg * goal_tol],   # < 100% - tolerance
                                    node_vels[-1:],                    # 100% +- tolerance
                                    [goal_vel + 2 * neg * goal_tol]])  # > 100% + tolerance

        # add node features
        for (_, n_data), vel in zip(sm.nodes(data=True), node_vels):
            n_data[VEL_ATTR] = vel

        # forward edges
        sm.add_edges_from([(i, j) for i, j in zip(range(num_states - 1), range(1, num_states))])

        # backward edges
        sm.add_edges_from([(i, j) for i, j in zip(range(1, num_states), range(num_states - 1))])

        # self edges
        sm.add_edges_from([(i, i) for i in range(num_states)])

        # rewards
        for u in sm.nodes:
            if sm.nodes[u][VEL_ATTR] == goal_vel:
                sm[u][u][REWARD_ATTR] = goal_reward

        # for u in sm.nodes:
        #     for v in sm[u]:
        #         if sm.nodes[v][VEL_ATTR] == goal_vel:
        #             sm[u][v][REWARD_ATTR] = goal_reward
        #         elif sm.nodes[u][VEL_ATTR] == goal_vel and u != v:
        #             sm[u][v][REWARD_ATTR] = -goal_reward

        return sm

    def draw(self, layout=None, plot_figsize=None, **nxdraw_kwargs):
        vel_labels = {i: f"{self.G.nodes[i][VEL_ATTR]:.2f}" for i in self.G.nodes}
        fig_width = self.num_states + 5
        colormap = ['red'] * self.G.number_of_nodes()

        colormap[self.__goal_node] = '#66FF66'
        colormap[self.u0] = '#6666FF'

        custom_nx_draw_kwargs = dict(with_labels=True,
                                     labels=vel_labels,
                                     node_size=1000,
                                     node_color=colormap,
                                     font_weight='bold',
                                     connectionstyle='arc3, rad = 0.1')
        custom_nx_draw_kwargs.update(nxdraw_kwargs)

        super().draw(layout=layout or linear_layout,
                     plot_figsize=plot_figsize or (fig_width, fig_width / 5),
                     **custom_nx_draw_kwargs)

    def __is_goal_node(self, u):
        return self.G.nodes[u][VEL_ATTR] == self.__goal_vel

    @property
    def __goal_node(self):
        return [n for n in self.G.nodes if self.__is_goal_node(n)][0]

    def get_node_feature_attr(self):
        return [VEL_ATTR]

    def get_edge_feature_attr(self):
        return []


def linear_layout(G):
    return {u: [i * 2, 0] for i, u in enumerate(G.nodes())}
