"""
@author: erathorn
@date: July 2019
@version: 1.0
"""


class Node(object):
    """
    This class implements a tree node.
    A Node is identified by its name.
    """

    __slots__ = ("name", "binary", "data", "cluster", "terminal", "mother", "__left", "__right", "incident_length",
                 "likelihood_rel", "accumulated_likelihood_rel", "cluster_store", "arity", "state_probs")

    def __init__(self, name):
        """
        constructor of Node
        branch length is set to 1.0 by default

        :param name: name of the node
        :type name: str
        """
        self.name = name
        self.binary = ""
        self.terminal = True
        self.mother = None
        self.__left = None
        self.__right = None
        self.incident_length = 0.0
        self.likelihood_rel = 0.0

        self.accumulated_likelihood_rel = 0.0

        self.arity = 0

    @property
    def right(self):
        # type: () -> Node
        return self.__right

    @right.setter
    def right(self, value):
        # type: (Node) -> None

        value.mother = self
        self.__right = value
        self.terminal = False

    @property
    def left(self):
        # type: () -> Node
        return self.__left

    @left.setter
    def left(self, value):
        # type: (Node) -> None

        value.mother = self
        self.__left = value
        self.terminal = False

    @property
    def sister(self):
        # type: () -> Node
        m = self.mother
        return m.right if m.left == self else m.left

    def __str__(self):
        return self.name

    def __repr__(self):
        return "Node: ".join((str(self.name)))

    def __eq__(self, other):
        # type: (Node) -> bool
        """
        overwrite the build in eq function to compare Nodes by their binary representation

        :param other: other node to compare with
        :type other: Node.Node
        """
        return self.binary == other.binary

    def get_terminal(self, term=None):
        # type: (Node) -> list
        """
        get all the terminal descendants of the current node.

        :param term: list which collects all the terminal nodes
        :return: list of terminal nodes descending from self
        """

        term = [] if term is None else term

        if self.terminal:
            term += [self.name]
        else:
            self.left.get_terminal(term=term)
            self.right.get_terminal(term=term)

        return term

    def find_anc(self, anc_bin):
        # type: (str) -> Node
        """
        Get the node which is specified through the binary representation anc_bin.
        This node is an ancestor of the current node. This function does not check, if anc_bin is actually
        a representation of an ancestor of self

        :param anc_bin: binary representation of an ancestor of self
        :return: the node corresponding to anc_bin
        """

        if self.binary == anc_bin:
            return self
        node = self
        while True:
            node = node.mother
            if node.binary == anc_bin:
                return node

    def write_cluster(self, outfile, lang_list):
        """

        Still here fore legacy reasons

        :param outfile:
        :param lang_list:
        :return:
        """
        for concept_ind, concept in enumerate(self.cluster_store):
            with open(outfile + "_" + str(concept_ind), "a") as out:
                for cluster in range(int(concept[36][0][0])):
                    for l in range(int(concept[cluster][0][36])):
                        out.write(lang_list[int(concept[cluster][0][l])] + ",")
                    out.write("\t")
                out.write("\n")
