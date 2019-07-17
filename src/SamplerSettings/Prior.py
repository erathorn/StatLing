"""
@author: erathorn
@date: July 2019
@version: 1.0

"""

"""
The next for parameters specify the prior on the branch lengths of the tree.
The CompoundDirichlet distribution described in Zhang, Rannala and Yang 2012. (DOI:10.1093/sysbio/sys030)
"""
# concentration parameter of the dirichlet distribution
a = 1.0

# ratio of prior means of internal and external branch lengths
c = 1.0

# (alpha_t/beta) defines the mean and (alpha_t/beta**2) the variance of the joint distribution of the branch lengths
beta = 0.100
alpha_t = 1.0


"""
The next parameters specify the proportion of the respective moves in the random walk part
"""
# prior on tree sampling topology
tree_sample_topology = 10

# prior on tree sampling time
tree_sample_branches = 5

# prior on emission sampling (frequency & evo)
emission_sample_class = 1

# prior on emission sampling (frequency & evo)
emission_sample_freq = 1

# prior on transition sampling
transition_sample = 1


"""
The next parameters specify the change window in the random walk sampling part
"""
# change window for frequency values
dmax_freq = 0.005

# change window for evolutionary rates
dmax_evo = 0.005

# change window for time
dmax_time = 0.1

# change window for transition a
dmax_trans_a = 0.05

# change window for transition r
dmax_trans_r = 0.1


"""
The next parameters specify the sampling width for the slice sampler
"""
# width for tree slice sampling
dmax_time_width = 0.01

# slice scale for frequencies
freq_scale = 0.0005

# slice scale for evo
evo_scale = 0.0005
