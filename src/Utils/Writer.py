"""
@author: erathorn
@date: July 2019
@version: 1.0
"""


class DataWriter(object):
    """
    This class handles Parameter Writing during an MCMC run
    """
    __slots__ = ("evo_model", "output_dir", "concepts", "save_state")

    def __init__(self, evo_model, output_dir, sample):
        """
        Create Data Writer instance
        :param evo_model:
        :type evo_model: EvolutionaryModel
        :param output_dir:
        :type output_dir: str
        :param sample:
        :type sample: bool
        """
        self.evo_model = evo_model
        self.output_dir = output_dir
        self.concepts = self.evo_model.data.concepts

        # expose the relevant functions to the outside
        if sample:
            self.save_state = self._save_state_with_cognates
        else:
            self.save_state = self._save_state_without_cognates

    def _save_state_with_cognates(self, counter, likelihood, header):
        # type: (int, float, bool) -> None
        """
        Save the current state with sampled cognates

        still here for legacy reasons

        :param counter: current iteration
        :type counter: int
        :param likelihood: current likelihood
        :type likelihood: float
        :param header: should the header be written
        :type header: bool
        """
        self.track_params(counter, likelihood)
        self.store_sound(header)
        self.store_trans(int(header))
        self.store_cognates(header)

    def _save_state_without_cognates(self, counter, likelihood, header):
        # type: (int, float, bool) -> None
        """
        Save the current state with sampled cognates

        :param counter: current iteration
        :type counter: int
        :param likelihood: current likelihood
        :type likelihood: float
        :param header: should the header be written
        :type header: bool
        """
        self.track_params(counter, likelihood)
        self.store_sound(header)
        self.store_trans(int(header))
        self.store_lambda(header)

    def track_params(self, run, lik):
        # type: (int, float) -> None
        """

        :param run: current iteration of the MCMC
        :type run: int
        :param lik: current likelihood of the model
        :type lik: float
        """
        self.store_likelihood(run, lik)

        self.store_tree(run)

    def create_params(self):
        # type: () -> None
        """
        Create the files for parameter storage
        """
        self.create_likelihood()

    def store_likelihood(self, run, lik):
        # type: (int, float) -> None
        """

        :param lik: current likelihood of the model
        :type lik: float
        :param run: current iteration of the MCMC
        :type run: int
        """
        with open(self.output_dir + "MCMC_test.params.log", "a") as outf:
            outf.write(str(run) + "\t" + str(lik))
            outf.write("\n")

    def store_tree(self, run):
        # type: (int) -> None
        """
        Store the tree
        
        :param run: current iteration of the MCMC
        :type run: int
        """
        with open(self.output_dir + "MCMC_test.trees.log", "a") as outf:
            outf.write(str(run) + "\t")
            outf.write(self.evo_model.data.tree.newick)
            outf.write("\n")

        with open(self.output_dir + "MCMC_test.trees_vals.log", "a") as outf:
            outf.write(str(run) + "\t")
            outf.write(str(self.evo_model.data.tree.tree_height) + "\t" + str(self.evo_model.data.tree.tree_length))
            outf.write("\n")

    def store_lambda(self, header=True):
        # type: (bool) -> None
        """
        Store the lambda parameter
        
        :param header: should the header line be written
        :type header: bool 
        """
        if header is True:
            with open(self.output_dir + "MCMC_test.lambda.log", "a") as outf:
                outf.write("lambda")
                outf.write("\n")
        with open(self.output_dir + "MCMC_test.lambda.log", "a") as outf:
            outf.write(str(self.evo_model.lam))
            outf.write("\n")

    def store_sound(self, header=True):
        # type: (bool) -> None
        """
        Store the sound model

        :param header: should the header line be written
        :type header: bool
        """
        self.evo_model.em_mod.sound_model.to_file(self.output_dir + "sound_mod.log", header=header)

    def store_trans(self, header=1):
        # type: (int) -> None
        """
        Store the transition parameters

        :param header: Should the header be written, if 1 header is written
        :type header: int
        """
        self.evo_model.tr_mod.to_file(header=header, file_name=self.output_dir + "tr_mod.log")

    def store_cognates(self, header=True):
        # type: (bool) -> None
        """
        Store the cognates

        Still here fore legacy reasons

        :param header: should the header line be written
        :type header: bool
        """
        self.evo_model.data.to_file(self.output_dir + "cognate_sample.log", header=header)

    def store_sampled(self, header=True):
        if header:
            self.evo_model.data.write_cognate_map(self.output_dir + "cognate_map.log")
        self.evo_model.tree.write_cluster(self.output_dir + "cognate_sample.log", self.evo_model.data.languages)

    def create_likelihood(self):
        """
        create the likelihood file
        """
        with open(self.output_dir + "MCMC_test.params.log", "a") as outf:
            outf.write("it" + "\t" + "lik")
            outf.write("\n")
        with open(self.output_dir + "MCMC_test.trees_vals.log", "a") as outf:
            outf.write("it" + "\t" + "tree_height" + "\t" + "tree_length")
            outf.write("\n")
