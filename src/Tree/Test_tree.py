import unittest
import Tree
import Node
import ete2
import random

x = Tree.Tree("test_tree")

a = Node.Node("a")
# print a
b = Node.Node("b")
c = Node.Node("c")
d = Node.Node("d")
e = Node.Node("e")
f = Node.Node("f")
g = Node.Node("g")
h = Node.Node("h")
i = Node.Node("i")
j = Node.Node("j")
k = Node.Node("k")
l = Node.Node("l")
m = Node.Node("m")
n = Node.Node("n")
o = Node.Node("o")
p = Node.Node("p")

class MyTestCase(unittest.TestCase):




    def test_rnni(self):

        x.random_tree([a, b, c, d, e, f, g, h, i, j, k, l])


        s = x.newick
        etetree1 = ete2.Tree(s, format=1)
        tl = x.tree_length
        pren = {no.name: no.incident_length for no in x.all_nodes}
        for origin in x.all_nodes:
            if origin.binary != "1" and origin.terminal is False:
                for child in [0,1]:
                    pre_top2 = {no.name: (no.left.name, no.right.name) if no.terminal is False else (None,None) for no in x.all_nodes}
                    old_root = x.root.name
                    x.__rNNI(origin, child)
                    x.set_binary()

                    for target in x.all_nodes:

                        if target.binary.startswith(origin.binary):
                            pass

                        elif origin.binary == "1":
                            pass
                        elif origin.mother == target:
                            pass
                        elif target.mother is not None and target.mother == origin.mother:
                            pass

                        else:
                            s3 = x.newick
                            pren2 = {no.name: no.incident_length for no in x.all_nodes}
                            pre_top = {no.name: (no.left.name, no.right.name) if no.terminal is False else (None,None) for no in x.all_nodes}
                            old_root_2 = x.root.name
                            pre = x.__rSPR(origin, target)
                            etetree3 = ete2.Tree(s3, format=1)
                            x.set_binary()

                            x.revert_topology_move(pre_top, pren, old_root_2)

                            aftn2 = {no.name: no.incident_length for no in x.all_nodes}
                            s2 = x.newick
                            etetree2 = ete2.Tree(s2, format=1)

                            rf = etetree2.robinson_foulds(etetree3)[0]

                            self.assertEqual(rf, 0, "failed for " + origin.name + " and " + target.name)
                            self.assertEqual(pren2, aftn2)

                    x.revert_topology_move(pre_top2, pren, old_root)
                    tl2 = x.tree_length
                    aftn = {no.name: no.incident_length for no in x.all_nodes}
                    s2 = x.newick
                    etetree2 = ete2.Tree(s2, format=1)

                    rf = etetree1.robinson_foulds(etetree2)[0]
                    # self.assertEqual(s, s2)
                    self.assertEqual(rf, 0)
                    self.assertEqual(tl, tl2)
                    self.assertEqual(pren, aftn)



    def test_rspr(self):
        random.seed(1234)
        x.random_tree([a, b, c, d, e, f, g, h, i, j, k, l])
        s = x.newick
        etetree1 = ete2.Tree(s, format=1)
        tl = x.tree_length
        pren = {no.name: no.incident_length for no in x.all_nodes}
        for origin in x.all_nodes:
            for target in x.all_nodes:

                if len(origin.binary) > len(target.binary):
                    origin,target = target,origin
                if origin.binary == "1" or target.binary == "1":
                    pass
                elif origin.mother == target or origin.mother == target.mother:
                    pass
                elif target.mother == origin:
                    pass
                elif target.binary.startswith(origin.binary):
                    pass
                else:
                    old_root = x.root.name
                    pre_top2 = {no.name: (no.left.name, no.right.name) if no.terminal is False else (None,None) for no in x.all_nodes}
                    pre=x.__rSPR(origin, target)

                    x.set_binary()

                    if target.binary != "1" and target.terminal == False:
                        for child in [0, 1]:
                            s3 = x.newick
                            pren2 = {no.name: no.incident_length for no in x.all_nodes}
                            old_root_2 = x.root.name
                            pre_top = {no.name: (no.left.name, no.right.name) if no.terminal is False else (None,None) for no in x.all_nodes}
                            x.__rNNI(target, child)
                            etetree3 = ete2.Tree(s3, format=1)
                            x.revert_topology_move(pre_top, pren2, old_root_2)
                            #tl2 = x.tree_length
                            aftn2 = {no.name: no.incident_length for no in x.all_nodes}
                            s2 = x.newick
                            etetree2 = ete2.Tree(s2, format=1)

                            rf = etetree2.robinson_foulds(etetree3)[0]

                            self.assertEqual(rf, 0, "failed for "+origin.name + " and " + target.name)
                            self.assertEqual(pren2, aftn2)

                    x.revert_topology_move(pre_top2, pren, old_root)
                    tl2 = x.tree_length
                    aftn = {no.name: no.incident_length for no in x.all_nodes}
                    s2 = x.newick
                    etetree2 = ete2.Tree(s2, format=1)

                    rf = etetree1.robinson_foulds(etetree2)[0]

                    self.assertEqual(rf, 0)
                    self.assertEqual(tl, tl2)
                    self.assertEqual(pren, aftn, "failed for " + str(origin.name) + " and " + str(target.name))
if __name__ == '__main__':
    random.seed(1234)

    unittest.main()

