# StatLing

This is an implementation of a statistical cognate detection and tree inference approach. Instead of an absolute
judgement of cognacy a posterior probability can be retrieved. The parameter estimation and the estimation of posterior
cognate judgements is done using a Markov Chain Monte Carlo framework. The parameters of the model
are estimated through a combined scheme of *random walk* and *slice sampling* moves.

## Dependencies
- Python 2.7
- numpy
- pandas
- dendropy
- scipy
- cython
- igraph (python-igraph)
- mpi4py

## Installation

In order to use the module some C code has to be compiled to make it accessible to the program. 


Navigate to `src\C_Extensions` and then compile the C-code:
```
gcc -c fw_log.c
ar -rcs libfw_log.a fw_log.o
``` 

To install the remaining C functionality run
```
python setup.py build_ext --inplace
```

This installs the C-functionality locally for this module.

If you want to enable optimizations you can do so in the setup file.

## Usage

There are some places where you can set parameters for the MCMC estimation.

1. A *settings file* which should be supplied to the main file using the `-s` flag.
2. The folder `src\SamplerSettings\` contains a file `Prior.py` which contains information about the prior
distribution and the sampling widths. The file `Constraint_Tree.py` contains information about tree constraints.
Please refer to the two files for information about their exact usage.
3. The folder `src\Utils` contains the file `ltrees.py` which contains a tree in newick format. The data type of this
tree is a `tuple`. This tree is used as a seed.

### Flags in the settings file

The *settings file* consists of several flags which set some options for the MCMC engine.

* \[Seed\] - Specify the seed for the random number generator
* \[Iterations\] - Specify the number of iterations the MCMC runs
* \[Thinning\] - Thinning parameter for the MCMC
* \[Data\] - Path to the data file
* \[Output\] - Path were the output is stored
* \[ldn\] - exclude word pairs who's normalized Levenshtein distance is below this threshold
* \[MC3\] - should the MC3 parameter estimation scheme be used. (requires MPI)
* \[concept_col\] - name of the column in the data file where the information about the concepts is stored
* \[lang_col\] - name of the column where the language is stored 
* \[transcription_col\] - name of the column where the word is stored
* \[Sound Model\] - string specifying the Sound Model in use, currently only supports "asjp"
* \[consistency_checker\] - Should the consistency of the tree with predefined constraints be checked
* \[Transition_Model\] - Specifies the transition model for the alignment model. Currently only supports "KM03"
* \[bottom-up\] - Still here for legacy reasons. Must be set to false. Will probably be deleted.
* \[randomize\] - Should the starting point be randomized
* \[randomize steps\] - Number of randomization steps
* \[folder\] - Specify a folder where information about an explicit starting point is stored.

This is an example of how the settings file should look like.
```text
[Seed]
42

[Iterations]
100

...
```

### The Data File

The data file should be in `.tsv` format, i.e. tab separated.
For example: 

iso_code | gloss | transcription
---------|-------|--------------
ger      | eye   | ai
fra      | eye   | Ey
...      | ...   | ...

In the settings file the values for \[concept_col\] would be "gloss", \[lang_col\] would be set to 
"iso_code" and \[transcription_col\] to "transcription".
 
