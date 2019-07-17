"""
@author: erathorn
@date: July 2019
@version: 1.0
"""

import collections
import itertools
import os
import random

import numpy as np
import scipy.special as spsp

import Node as Node
import src.C_Extensions.sample as c_sam
import src.SamplerSettings.Constraint_Tree as ConTree
import src.SamplerSettings.Prior as SamPrior


class Tree(object):
    __slots__ = ("name", "anc_to_off_dict", "all_nodes", "times", "__tree_l", "_terminals", "check_for_consistency",
                 "_name_node_map", "_root", "sampling_functions_local_topology", "time_store", "lpair_mapping", "lnp",
                 "sampling_functions_local_branch", "calc_po", "__po"
                 )

    def __init__(self, name):

        self.anc_to_off_dict = None
        self.name = name
        self.all_nodes = None

        self._name_node_map = {}
        self.times = collections.defaultdict(lambda: collections.defaultdict())

        self.time_store = None
        self.lpair_mapping = None
        self.terminal_nodes = []
        self.check_for_consistency = True
        self.root = None
        self.calc_po = True
        self.__po = None
        self.sampling_functions_local_topology = []
        self.sampling_functions_local_branch = []
        self.lnp = None
        # first part of the tree prior is calculated
        self.calculate_lnp()

        # set the sampling functions
        self.sampling_functions_local_topology = [
            (self.sample_tree_topology, self.revert_topology_move, "topology", {"move": 0}),
            (self.sample_tree_topology, self.revert_topology_move, "topology", {"move": 1})
        ]
        self.sampling_functions_local_branch = [
            (self.sample_tree_length, self.revert_length_move, "tree time", {"move": 0}),
            (self.sample_tree_length, self.revert_length_move, "tree time", {"move": 1})]

    @property
    def post_order(self):
        if self.calc_po is True:
            self.__po = self.traverse_tree()
        self.calc_po = False
        return self.__po

    @property
    def n_terminal(self):
        # type: () -> int
        return len(self.terminal_nodes)

    @property
    def terminal_nodes(self):
        # type: () -> list
        return self._terminals

    @terminal_nodes.setter
    def terminal_nodes(self, value):
        for _n in value:  # type: Node
            _n.terminal = True
            _n.accumulated_likelihood_rel = 0.0
        self._terminals = value

    @property
    def newick(self):
        # type: () -> str
        """
        returns a newick representation of the tree
        :return: newick representation of the tree
        :rtype: str
        """
        n_fom = str(self.__newick())[1:-1]
        return n_fom.replace("[", "(").replace("]", ")").replace("'", "").replace(", +,", "").replace(" ", "") + ";"

    @property
    def all_edges(self):
        return np.array([node.incident_length for node in self.post_order])

    @property
    def tree_height(self):
        # type: () -> float
        return max([self.__path(n, self.root) for n in self.terminal_nodes])

    @property
    def tree_length(self):
        # type: () -> float
        return sum([node.incident_length for node in self.all_nodes])

    @property
    def root(self):
        # type: () -> Node.Node
        return self._root

    @root.setter
    def root(self, value):
        # type: (Node.Node) -> None
        self._root = value
        if value is not None:
            value.incident_length = 0.0
            value.mother = None

    def __getitem__(self, item):
        return self.times[item[0]][item[1]]

    def __str__(self):
        # type: () -> str
        return self.newick

    def __len__(self):
        n_fom = str(self.__newick())[1:-1]
        n_form = n_fom.replace("[", "").replace("]", "").replace("'", "").replace("+", "").replace(" , ", "").replace(
            " ", "").split(",")
        return len(n_form)

    @staticmethod
    def __sym_path(offspring, ancestor):
        # type: (Node.Node, Node.Node) -> list
        """
        creates the symbolic path, i.e. actual nodes from offspring to ancestor.
        This function assumes that ancestor is an actual ancestor of offspring.
        Otherwise there will be an infinite loop.

        :param offspring: Node in the tree
        :param ancestor: ancestor of offspring
        :return: list of nodes specifying the path from ancestor to offspring
        """

        f_path = []
        while True:
            f_path.append(offspring)
            offspring = offspring.mother
            if offspring == ancestor:
                break

            # add lca
        f_path.append(offspring)
        return f_path

    def __path(self, offspring, ancestor):
        # type: (Node.Node, Node.Node) -> float
        """
        Calculate the length of the path from offspring to ancestor

        This function uses previously calculated prior knowledge about sub paths.

        :param offspring: Target of the path
        :type offspring: Node.Node
        :param ancestor: origin of the path
        :type ancestor: Node.Node
        :return: length of the path from offspring to ancestor
        :rtype: float
        """
        if self.anc_to_off_dict[ancestor.binary][offspring.binary]:
            return self.anc_to_off_dict[ancestor.binary][offspring.binary]

        # ancestor stays fixed
        ancestor_bin = ancestor.binary

        org_off_bin = offspring.binary

        self.anc_to_off_dict[ancestor_bin][org_off_bin] += offspring.incident_length

        offspring = offspring.mother

        while offspring.binary != ancestor_bin:
            if self.anc_to_off_dict[ancestor_bin][offspring.binary]:
                # this sub path is already calculated and stored in the dictionary, so this knowledge is used
                self.anc_to_off_dict[ancestor_bin][org_off_bin] += self.anc_to_off_dict[ancestor_bin][offspring.binary]
                return self.anc_to_off_dict[ancestor_bin][org_off_bin]
            else:
                # the sub path is not known yet, but created here
                self.anc_to_off_dict[ancestor_bin][offspring.binary] += offspring.incident_length
                self.anc_to_off_dict[ancestor_bin][org_off_bin] += offspring.incident_length
                offspring = offspring.mother

        return self.anc_to_off_dict[ancestor_bin][org_off_bin]

    def __newick(self, node=None):
        # type: (Node.Node) -> list
        """
        DO NOT CALL THIS FUNCTION DIRECTLY
        performs the internal newick recursion
        :param node: a node in the tree
        :type node: Node.Node
        :return: newick representation in list format
        :rtype: list
        """
        some_list = []
        if not node:
            node = self.root
        if not node.terminal:
            if node.mother is not None:

                inc_len = node.incident_length

            else:
                inc_len = 1.0
            some_list += [
                [self.__newick(node.left),
                 self.__newick(node.right)],
                "+", node.name + ":" + str(inc_len)]
        else:

            return node.name + ":" + str(node.incident_length)

        return some_list

    def __tree_length_sep(self):
        # type: () -> tuple
        """
        Calculate the length of the internal branches and the branches leading to the leaves separately

        :return: leave branch length, internal branch length
        :rtype: tuple
        """
        v1, v2 = 0.0, 0.0
        for node in self.all_nodes:
            if node.terminal is True:
                v1 += np.log(node.incident_length)
            else:
                if node.binary != "1":
                    v2 += np.log(node.incident_length)

        return v1, v2

    def calculate_lnp(self):
        # type: () -> None
        """
        Calculate the first part of the tree prior
        """
        v1 = SamPrior.alpha_t * np.log(SamPrior.beta)
        v2 = spsp.gammaln(SamPrior.alpha_t)
        v3 = spsp.gammaln(SamPrior.a * self.n_terminal + SamPrior.a * SamPrior.c * (self.n_terminal - 3))
        v4 = self.n_terminal * spsp.gammaln(SamPrior.beta) - (self.n_terminal - 3) * spsp.gammaln(
            SamPrior.beta * SamPrior.a)
        self.lnp = v1 - v2 + v3 - v4

    def pre_set_times(self):
        # type: () -> None
        """
        Create the times dictionary, so it is properly initialized
        """
        self.times = collections.defaultdict()
        for l1 in self.terminal_nodes:
            self.times[l1.name] = collections.defaultdict(float)
            for l2 in self.terminal_nodes:
                self.times[l1.name][l2.name] = 0.0

    def set_all_edges(self, new_ls):
        # type: (np.ndarray|list) -> list
        """
        set all edge lengths
        
        :param new_ls: new edge length vector
        :type new_ls: np.ndarray|list
        :return: old branch lengths
        :return: list
        """
        old = []
        for node, nl in zip(self.post_order, new_ls):
            old.append(node.incident_length)
            node.incident_length = nl
        self.recalculate_time()
        return old

    def traverse_tree(self, node=None, traversal=None, non_terminal=True):
        # type: (Node.Node, list, bool) -> list
        """
        This function performs a post-order tree-traversal

        :param node: Node in the tree
        :type node: Node.Node|None
        :param traversal: temporary list
        :type traversal: list|None
        :param non_terminal: include terminal symbols in traversal
        :type non_terminal: bool
        :return: post order tree traversal
        :rtype: list
        """

        traversal = [] if traversal is None else traversal

        node = self.root if node is None else node

        if not node.terminal:
            self.traverse_tree(node.left, traversal, non_terminal)
            self.traverse_tree(node.right, traversal, non_terminal)

        traversal.append(node)

        return traversal

    @staticmethod
    def get_children(node):
        # type: (Node.Node) -> tuple|bool
        """
        get the immediate children of a node.
        Returns False if node is terminal

        :param node: Node in the tree
        :return: the two immediate children of @param: node or False if node is terminal
        """
        if node.terminal:
            return False
        return node.left, node.right

    def set_binary(self, node=None):
        # type: (Node.Node) -> None
        """
        This function sets the binary representation of the nodes.
        :param node: A node in the tree, if the node is not set the root is selected
        """
        if node is None:
            node = self.root
            node.binary = "1"

        check = not node.terminal  # type: bool
        if check:
            left = node.left
            right = node.right

            binary_code = node.binary
            left.binary = "".join((binary_code, "1"))
            right.binary = "".join((binary_code, "0"))
            self.set_binary(left)
            self.set_binary(right)

    def recalculate_time(self):
        # type: () -> None
        """
        This function recalculates the branch lengths, i.e. time, for all of the terminal nodes
        of the tree.
        """
        self.anc_to_off_dict = collections.defaultdict(lambda: collections.defaultdict(float))

        for l1, l2 in itertools.combinations(self.terminal_nodes, 2):
            p = self.path(l1, l2)
            self.times[l1.name][l2.name] = p
            self.times[l2.name][l1.name] = p
            if self.lpair_mapping is not None:
                self.time_store[self.lpair_mapping[(l1.name, l2.name)]] = p

    def sample_tree_length(self, move):
        # type: (int) -> (bool, float, tuple)
        """
        Sample branch length in the tree
        :return:
        :rtype: tuple
        """
        o_length_prior = self.calculate_length_prior()

        if move == 0:
            """
            A new branch length is sampled
            """

            # get random node
            lang = self.random_node(not_root=True, not_term=False)

            # get previous branch length
            prev_param = lang.incident_length

            # sample new branch length
            proposed_value, m_h_ratio, old_val = c_sam.sample_uniform_single(d_max=SamPrior.dmax_time, r=prev_param,
                                                                             lower=0.0, upper=np.Infinity)

            # assign new branch length
            lang.incident_length = proposed_value

            old = (lang.name, prev_param, move)
        else:
            # execute slide move
            lang = self.random_node(not_root=True, not_term=True)

            proportion = random.uniform(0, 1)

            n_name, (olm, olc), _ = self.slide(lang, proportion)

            m_h_ratio = 0.0

            old = (lang.name, (n_name, olm, olc), move)
        self.recalculate_time()

        n_length_prior = self.calculate_length_prior()

        return False, m_h_ratio, n_length_prior - o_length_prior, old

    def random_node(self, not_root=True, not_term=False):
        # type: (bool,bool) -> Node.Node
        """
        select a random node from all the nodes of the tree

        :return: Node.Node
        """

        random.shuffle(self.all_nodes)

        if not_root and not not_term:
            t = self.all_nodes[0]
            while t.binary == "1":
                random.shuffle(self.all_nodes)
                t = self.all_nodes[0]
        elif not not_root and not_term:
            t = self.all_nodes[0]
            while t.terminal:
                random.shuffle(self.all_nodes)
                t = self.all_nodes[0]
        else:
            t = self.all_nodes[0]
            while t.terminal or t.binary == "1":
                random.shuffle(self.all_nodes)
                t = self.all_nodes[0]

        return t

    def calculate_length_prior(self):
        # type: () -> float
        """
        This function calculates the prior of the current tree length
        The CompoundDirichlet distribution described in Zhang, Rannala and Yang 2012. (DOI:10.1093/sysbio/sys030)

        :return: prior of the current tree length
        """
        leave, internal = self.__tree_length_sep()
        ln1 = (SamPrior.a - 1.0) * leave + (SamPrior.a * SamPrior.c - 1) * internal
        ln2 = (SamPrior.alpha_t - SamPrior.a * self.n_terminal - SamPrior.a * SamPrior.c * (
                self.n_terminal - 3)) * np.log(self.tree_length) - SamPrior.beta * self.tree_length

        return ln1 + ln2 + self.lnp

    def slide(self, node, proportion):
        # type: (Node.Node, float) -> (str, float, float)
        """
        Perform a slide move. This move moves an internal node up or down the edge created by its
        incoming and one outgoing edge. The total tree height is not affected.

        :param node: node which moves
        :param proportion: size of the move
        :return: the name of the node, the old incident length, and the old incident length of the selected child
        """

        n1 = node.left
        n2 = node.right

        side = random.choice([True, False])
        if side:
            # left side slide
            olm = node.incident_length
            olc = n1.incident_length
            comb = olm + olc
            l1 = proportion * comb
            ml = comb - l1
            node.incident_length = ml
            n1.incident_length = l1
            n_name = n1.name
        else:
            # right side slide
            olm = node.incident_length
            olc = n2.incident_length
            comb = olm + olc
            l1 = proportion * comb
            ml = comb - l1
            node.incident_length = ml
            n2.incident_length = l1
            n_name = n2.name

        return n_name, (olm, olc), (l1, ml)

    def revert_slide(self, node, side, olm, olc):
        # type: (Node.Node, str, float, float) -> None
        """
        revert a slide move

        :param node: Node in the tree where the slide move needs to be reverted
        :type node: Node.Node
        :param side: string indicating where the slide happened
        :type side: str
        :param olm: old length of node
        :type olm: float
        :param olc: old lenght of slided node
        :type olc: float
        """

        node.incident_length = olm
        if node.left.name == side:
            node.left.incident_length = olc
        else:
            node.right.incident_length = olc

    def lca_multiple(self, list_of_nodes):
        # type: (list) -> Node.Node
        """
        This function finds the latest common ancestor of a multitude of nodes.

        :param list_of_nodes: List of nodes for which to find the latest common ancestor
        :return: The latest common ancestor of the nodes specified in the list
        """
        n1 = list_of_nodes.pop()
        n2 = list_of_nodes.pop()
        curr_lca = self.lca(n1, n2)
        while len(list_of_nodes) > 0:
            curr_node = list_of_nodes.pop()
            curr_lca = self.lca(curr_lca, curr_node)
        return curr_lca

    def check_consistency(self, constraints_dict):
        # type: (dict) -> bool
        """
        This function checks if the current tree is consistent wit the constraints specified in
        constraints_dict. Constraints are specified in terms of latest common ancestor, i.e. a set of nodes
        specified in constraints_dict needs to have a common ancestor which is not the root. Also the least common
        ancestor should not have any other terminal children then the given set of nodes.

        :param constraints_dict: dictionary specifying topology constraints
        :return: boolean indicating whether the constraints are observed or not
        """

        root_bin = "1"  # type: str
        for constraint, information in constraints_dict.iteritems():

            # check if least common ancestor is not root
            bin_collection = [self._name_node_map[node].binary for node in information]

            lca_bin_str = os.path.commonprefix(bin_collection)  # type: str
            if lca_bin_str == root_bin:
                return False

            # check if least common ancestor has only the children it should have
            lca = self._name_node_map[node].find_anc(lca_bin_str)  # type: Node.Node
            all_term = set([n for n in lca.get_terminal()])
            if all_term != information:
                return False

        return True

    def sample_tree_topology(self, move):
        # type: (int) -> (None, float, tuple)
        """
        This function chooses a tree topology move.
        The two topology moves are:
        - rooted nearest neighbour interchange (rNNI)
        - rooted subtree prune and regraft (rSPR)
        Currently, all tree topologies have equal prior probability.

        :return: tuple consisting of None (needed to match signatures),
                 metropolis hastings prior ratio and previous state
        """

        pre_lengths = {no.name: no.incident_length for no in self.all_nodes}
        pre_top = {no.name: (no.left.name, no.right.name) if no.terminal is False else (None, None) for no in
                   self.all_nodes}
        old_root = self.root.name
        self.rNNI() if move == 0 else self.rSPR()

        old_top = (pre_top, pre_lengths, old_root)

        # boolean of check_consistency needs to be flipped, because check_consistency returns True
        # if constraints are matched, so no further sampling necessary
        self.set_binary()

        # all tree topologies are equally likely if they adhere to the constraints
        # if the topology does not adhere to the constraints the m_h_ratio is negative infinity
        m_h_ratio = 0.0
        if self.check_for_consistency:
            m_h_ratio = 0.0 if self.check_consistency(constraints_dict=ConTree.constraints) else -np.inf

        self.recalculate_time()
        self.calc_po = True

        return False, m_h_ratio, 0.0, old_top

    def random_tree(self, list_of_terminals):
        # type: (list) -> None
        """
        Generate a random binary tree from a list of nodes

        :param list_of_terminals: list of terminal nodes
        """
        assert len(list_of_terminals) >= 2
        self.terminal_nodes = tuple(i for i in list_of_terminals)
        temp_list = list_of_terminals
        ct = 1
        all_nodes = []
        while len(temp_list) != 1:
            random.shuffle(temp_list)
            n1 = temp_list.pop()
            n2 = temp_list.pop()
            all_nodes.append(n1)
            all_nodes.append(n2)
            latest_node = Node.Node(str(ct))  # type: Node.Node

            # get random branch lengths
            l1 = random.uniform(0, 1.5)
            l2 = random.uniform(0, 1.5)
            latest_node.left = n1
            latest_node.right = n2
            n1.incident_length = l1
            n2.incident_length = l2

            ct += 1
            temp_list.append(latest_node)
        all_nodes.append(latest_node)
        self.all_nodes = all_nodes

        self._name_node_map = {n.name: n for n in all_nodes}
        self.root = latest_node
        self.set_binary()

    @staticmethod
    def check_des(node1, node2):
        # type: (Node.Node, Node.Node) -> bool
        """
        Checks if one node1 is descendant form node2

        :param node1: node in the tree
        :type node1: Node
        :param node2: node in the tree
        :type node2: Node
        :return: boolean indicating if one of the nodes is a descendant of the other
        :rtype: bool
        """
        n1_bin = node1.binary  # type: str
        n2_bin = node2.binary  # type: str
        return n1_bin.startswith(n2_bin)

    @staticmethod
    def lca(node1, node2):
        # type: (Node.Node, Node.Node) -> Node.Node
        """
        find latest common ancestor of node1 and node2.

        :param node1: first node
        :param node2: second node
        :return: latest common ancestor of node1 and node2
        """

        lca_binary = os.path.commonprefix((node1.binary, node2.binary))
        if node1.binary == lca_binary:
            return node1
        node = node1
        while True:
            node = node.mother
            if node.binary == lca_binary:
                return node

    def path(self, node1, node2):
        # type: (Node.Node, Node.Node) -> float
        """
        Calculate the path between any two nodes in the tree

        :param node1: Node in the tree
        :type: node1: Node.Node
        :param node2: Node in the tree
        :type node2: Node.Node
        :return: path length separating node1 and node2
        :rtype: float
        """

        node1 = self._name_node_map[node1] if isinstance(node1, str) or isinstance(node1, unicode) else node1
        node2 = self._name_node_map[node2] if isinstance(node2, str) or isinstance(node2, unicode) else node2

        # if both nodes are the same the solution is trivial
        if node1.binary == node2.binary:
            return 0.0

        # find lowest common ancestor
        lca = self.lca(node1, node2)

        path_len = self.__path(node1, lca)
        path_len2 = self.__path(node2, lca)

        return path_len + path_len2

    def rNNI(self):
        # type: () -> (Node.Node, int)
        """
        preforms random rooted nearest neighbour interchange

        :return: list of terminal children, unaffected by the move
        :rtype: (tuple)
        """

        origin = self.random_node(not_term=True, not_root=True)

        target = random.randint(0, 1)

        self.__rNNI(origin, target)

        return origin.name, target

    def __rNNI(self, origin, child_chosen):
        # type: (Node.Node, int) -> None
        """
        Rooted nearest neighbour interchange

        :param origin: Node to be moved
        :type origin: Node.Node
        :param child_chosen: child of origin to be connected to mother
        :type child_chosen: int
        :return: None
        :rtype: None
        """

        # get relevant nodes
        left_child = origin.left
        right_child = origin.right

        mother = origin.mother
        sister = origin.sister

        # swap nodes
        if child_chosen == 0:
            right1 = left_child
            right2 = right_child

        else:
            right2 = left_child
            right1 = right_child

        # add new nodes
        mother.left = origin
        mother.right = right1

        origin.left = sister
        origin.right = right2

    def rSPR(self):
        # type: () -> (Node.Node, dict, dict)
        """
        random subtree prune and regraft move

        :return: list of terminal children, unaffected by the move
        :rtype: list, list
        """
        # old = self.create_prev_state()

        while True:
            # origin, but it can not be root
            origin = self.random_node(not_root=True)

            # target, can also be root
            target = self.random_node(not_root=False)

            if target.binary.startswith(origin.binary):
                # target can not be something that is moved
                pass
            elif origin.mother == target:
                # node is already at the target position, new node choice necessary
                pass
            elif target.binary == "1":
                break
            elif origin.mother == target.mother:
                # origin and target are sisters
                pass
            else:
                break

        old = self.__rSPR(origin, target)

        return old

    def __rSPR(self, origin, target):
        # type: (Node.Node, Node.Node) -> (Node.Node, Node.Node)
        """
        performs a subtree prune and regraft move from origin to target

        :param origin: node to be moved
        :type origin: Node.Node
        :param target: node where new node is attached above
        :type target: Node.Node
        :return: None
        :rtype: (Node.Node, Node.Node)
        """

        # necessary structural info
        origin_sister = origin.sister  # type: Node.Node
        origin_mother = origin.mother  # type: Node.Node

        if target.binary == "1":
            """
            if the target is the root, a new root node is created. This will be done by making the old mother
            of the origin node the root. 
            """
            # get necessary structural information
            gran = origin_mother.mother  # type: Node.Node
            aunt = origin_mother.sister  # type: Node.Node
            origin_mother_len = origin_mother.incident_length  # type: float

            # make origin_mother to new root
            origin_mother.left = target
            target.incident_length = origin_mother_len
            origin_mother.right = origin

            # rewire old location of origin_mother
            gran.left = aunt
            gran.right = origin_sister

            # set root to new location
            self.root = origin_mother

        else:
            # get necessary structural information
            target_sis = target.sister
            target_mother = target.mother

            # target_sister_len = target_sis.incident_length

            origin_mother_binary = origin_mother.binary

            if target_sis.binary == origin_mother_binary:
                # target is aunt
                # problematic because origin mother is moved
                target_sis = origin_sister

            if origin_mother_binary == "1":
                # mother of origin node is root
                the_len = origin_sister.incident_length
                self.root = origin_sister
                origin_mother.incident_length = the_len

            else:
                gran = origin_mother.mother
                aunt = origin_mother.sister

                gran.left = aunt
                gran.right = origin_sister

            # attach new node to target
            target_mother.left = target_sis
            target_mother.right = origin_mother
            # target_sis.incident_length = target_sister_len

            # rewire new node
            origin_mother.left = target
            origin_mother.right = origin

        return origin.name, origin_sister.name

    def revert_topology_move(self, old, prev_len, old_root):
        # type: (dict, dict, str) -> None
        """
        This function reverts a topology move

        :param old: old tree state
        :type old: dict
        :param prev_len: old branch lenghts
        :type prev_len: dict
        :param old_root: old root node
        :type old_root: str
        """
        # reset old state
        for k, v in old.iteritems():
            if v[0] is not None:
                self._name_node_map[k].left = self._name_node_map[v[0]]
            if v[1] is not None:
                self._name_node_map[k].right = self._name_node_map[v[1]]
        self.root = self._name_node_map[old_root]

        for k, v in prev_len.iteritems():
            self._name_node_map[k].incident_length = v

        # recalculate additional info

        self.set_binary()
        self.calc_po = True
        self.recalculate_time()

    def revert_length_move(self, node, prev_param, move):
        # type: (str, tuple|float, int) -> None
        """
        revert a tree move
        """
        node = self._name_node_map[node]
        if move == 0:

            node.incident_length = prev_param
        else:
            side, olm, olc = prev_param
            self.revert_slide(node, side, olm, olc)

        self.recalculate_time()

    @classmethod
    def from_newick(cls, newick):
        # type: (str|tuple) -> Tree
        """
        Create a class instance from a newick representation

        :param newick: newick representation of the initial tree
        :type newick: str|tuple
        :return: An instance of the Tree class
        :rtype: Tree
        """
        if isinstance(newick, str):
            n, all_nodes = parse_node_str(iter(newick))
        else:
            n, all_nodes = parse_node(newick)
        tr = cls(name="my_tree")
        tr.all_nodes = list(all_nodes)

        tr._name_node_map = {n.name: n for n in all_nodes}
        tr.root = n
        term = [i for i in all_nodes if i.right is None and i.left is None]
        tr.terminal_nodes = tuple(term)

        tr.pre_set_times()
        tr.calculate_lnp()
        return tr

    def write_cluster(self, outf, lang_list):
        """

        Still here for legacy reasons
        """
        self.root.write_cluster(outf, lang_list)


def parse_node_str(in_str, res=None, all_nodes=None):
    # type: (iter, list , list) -> tuple
    """
    parse a string representation of a newick tree

    :param in_str: iterator of input string
    :type in_str: iter
    :param res: intermediate resulting tree list
    :type res: list
    :param all_nodes: all nodes of the tree
    :type all_nodes: list
    :return: resulting tree list and list of all nodes
    :rtype: tuple
    """
    if res is None:
        res = []
    if all_nodes is None:
        all_nodes = []

    ms = ""
    while True:
        cc = next(in_str)
        if cc == ";":
            name = ms.split(":")[0]
            n1 = Node.Node(name=name)
            n1.incident_length = 1.0
            n1.left = res[0]
            n1.right = res[1]
            all_nodes.append(n1)
            return n1, all_nodes
        if cc == "(":
            res, all_nodes = parse_node_str(in_str, res, all_nodes)
        elif cc == ")":
            splited = ms.split(",")
            if "" in splited:
                if splited[0] == splited[1]:
                    return res, all_nodes
                if splited[0] == "":
                    name, inc_length = splited[1].split(":")
                    n1 = Node.Node(name=name)
                    n1.incident_length = float(inc_length)
                    res.append(n1)
                    all_nodes.append(n1)
                    return res, all_nodes
                if splited[1] == "":
                    name, inc_length = splited[0].split(":")
                    n1 = Node.Node(name=name)
                    n1.incident_length = float(inc_length)
                    res.append(n1)
                    all_nodes.append(n1)
                    return res, all_nodes

            for part in splited:
                name, inc_length = part.split(":")

                n1 = Node.Node(name=name)
                n1.incident_length = float(inc_length)
                all_nodes.append(n1)
                res.append(n1)
            if len(splited) == 1:
                res.pop()

                n1.left = res.pop()
                n1.right = res.pop()
                res.append(n1)

            return res, all_nodes
        else:
            ms += cc


def parse_node(newick, sub_list=None):
    # type: (tuple|str, list) -> tuple
    """
    parse a tuple representation of a newick tree

    :param newick: tuple|str
    :param sub_list: list of nodes
    :return: tuple
    """
    if sub_list is None:
        sub_list = set([])
    if not isinstance(newick, str):
        sub1, sub2 = newick
        l_node = parse_node(sub1, sub_list)[0]
        r_node = parse_node(sub2, sub_list)[0]

        name = "n_" + str(len(sub_list) + 1)
        n1 = Node.Node(name)
        sub_list.add(n1)

        n1.left = l_node
        l_node.incident_length = random.uniform(0, 1)
        n1.right = r_node
        r_node.incident_length = random.uniform(0, 1)

    else:

        n1 = Node.Node(name=newick)
        sub_list.add(n1)
    return n1, sub_list
