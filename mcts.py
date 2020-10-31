import copy
import numpy as np
import math
import random
from board import Stone, Board, Player
import conf
from log_util import mcts_logger as logger


class Node(object):
    def __init__(self, parent, p, player):
        self.parent = parent
        self.N = 0
        self.W = 0
        self.P = p
        self.player = player
        self.children = {}

    @property
    def Q(self):
        return (self.W / self.N if self.N else 0) * self.player

    def select(self, need_log=False):
        if need_log:
            logger.info("Children Ns: %s" % [getattr(self.children.get(i), "N", 0) for i in range(conf.num_outputs)])
            logger.info("Children Ws: %s" % [getattr(self.children.get(i), "W", 0) for i in range(conf.num_outputs)])
            logger.info("Children Qs: %s" % [getattr(self.children.get(i), "Q", 0) for i in range(conf.num_outputs)])

        nb = sum([child.N for action, child in self.children.items()])
        max_qu = -1
        for idx, child in self.children.items():
            U = conf.CPUCT * child.P  * math.sqrt(nb) / (1 + child.N)
            if child.Q + U > max_qu:
                max_qu = U + child.Q
                select_action = idx
        return self.children[select_action], select_action

    def expand(self, ps, player):
        for idx, p in enumerate(ps):
            if p > 0:
                self.children[idx] = Node(parent=self, p=p, player=player)

    def backup(self, v):
        self.N += 1
        self.W += v
        if self.parent is not None:  # not root
            self.parent.backup(v)

    def is_leaf(self):
        if len(self.children) > 0:
            return False
        else:
            return True

    def __str__(self):
        info = "N: %s, W: %s, Q: %s, P: %s" % (self.N, self.W, self.Q, self.P)
        return info


class MCTS(object):
    def __init__(self):
        self.root = Node(parent=None, p=None, player=Player.O)

    def dirichlet_noise(self, ps):
        """ Add Dirichlet noise in the root node """

        dim = (ps.shape[0],)
        new_ps = (1 - conf.dirichlet_esp) * ps + \
            conf.dirichlet_esp * np.random.dirichlet(np.full(dim, conf.dirichlet_alpha))
        return new_ps

    def check_and_log(self, log_str, condition):
        if condition:
            logger.info(log_str)

        return

    def search(self, board, net, simulate_num, T, add_dirichlet_noise):
        logger.info("Search start!")
        for i in range(simulate_num):
            log_condition = (i <= 1)

            self.check_and_log("simulate %s" % i, log_condition)
            cp_board = copy.deepcopy(board)
            self.check_and_log("root %s" % cp_board, log_condition)

            # move to leaf
            node = self.root
            is_done, winner = None, None
            while not node.is_leaf():
                node, action = node.select(need_log= log_condition)
                self.check_and_log("select action: %s, select node info: %s" % (action, node), log_condition)

                pos = (action // conf.board_size, action% conf.board_size)
                is_done, winner = cp_board.step(Stone(pos, cp_board.turn))
                self.check_and_log("board %s" % cp_board, log_condition)
            if is_done:
                if winner is None:
                    v = 0
                else:
                    v = winner
                self.check_and_log("Get done leaf, v: %s" % v, log_condition)
                
            else:
                ps, v = net.predict(np.array([cp_board.get_feature()]))

                self.check_and_log("Get leaf, v: %s" % v, log_condition)

                if add_dirichlet_noise and i == 0:
                    ps = self.dirichlet_noise(ps)
                for idx in cp_board.illegal_idx:
                    ps[idx] = 0
                ps /= sum(ps)
                node.expand(ps=ps, player=cp_board.turn)
            node.backup(v)
        ret = [0] * conf.board_size ** 2
        for idx, child in self.root.children.items():
            ret[idx] = child.N
        return np.array(ret) ** (1.0 / T) / sum(np.array(ret) ** (1.0 / T))

    def change_root(self, action):
        if action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None
        else:
            self.root = Node(parent=None, p=None, player=Player.O if self.root.player == Player.X else Player.X)

