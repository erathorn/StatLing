# coding: utf-8
import collections
import itertools
import pickle

import dendropy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import src.Analysis.Clustering as Clust
import src.EmissionModel as srcEM
import src.TransitionModel as srcTR
import src.Utils.Data_class as srcUTData
from src.C_Extensions.algorithms_cython import calculaute_odds_scores, calculaute_odds_scores_tkf


class Evaluator(object):

    def __init__(self, folder, dialect, thinning, burnin, trans_mod="km03", **kwargs):

        self.folder = folder
        self.dialect = dialect
        self.thinning = thinning
        self.trans_mod = trans_mod
        self.lambda_filename = self.folder + "MCMC_test.lambda.log"
        self.tr_filename = self.folder + "tr_mod.log"
        self.sound_mod_filename = self.folder + "sound_mod.log"
        self.sound_classes_filename = self.folder + "sound_mod.log_classes"
        self.tree_filename = self.folder + "MCMC_test.trees.log"
        self.likelihood_filename = self.folder + "MCMC_test.params.log"
        self.check_kwargs(**kwargs)

        up_to = min([self.file_len(self.tree_filename),
                     self.file_len(self.tr_filename),
                     self.file_len(self.sound_mod_filename),
                     self.file_len(self.likelihood_filename)])

        self.skip_rows = self.skip_idx_func(burnin, up_to, thinning)

        self.sound_mod_pd_raw = self.read_file(self.sound_mod_filename, nrows=up_to, dtype=np.float64)
        self.sound_classes_pd = self.read_file(self.sound_classes_filename)
        self.tr_pd_raw = self.read_file(self.tr_filename, nrows=up_to, dtype=np.float64)
        self.lambda_pd_raw = self.read_file(self.lambda_filename, nrows=up_to, dtype=np.float64)
        self.likelihood_raw = self.read_file(self.likelihood_filename, nrows=up_to, dtype=np.float64)
        self.tree_dist_dict, self.t_l, self.tree_heights_raw = self.read_tree(self.tree_filename, self.skip_rows)
        #
        #self.lambda_dict = .to_dict()["lambda"]
        self.lambda_dict = {}
        ld = self.thin(self.lambda_pd_raw)
        k =0
        for ind, row in ld.iterrows():
            self.lambda_dict[k] = row["lambda"]
            k +=1
        self.emission_dict = self.create_sound_model_dict()
        self.transition_dict = self.create_trans_dict_multi()


    def add_other_model(self, other_model):
        # type: (Evaluator) -> None
        """
        This function adds another model from the same experiment.

        :param other_model: Evaluator
        """

        self.tree_dist_dict = self.dict_joiner(self.tree_dist_dict, other_model.tree_dist_dict)
        self.emission_dict = self.dict_joiner(self.emission_dict, other_model.emission_dict)
        self.transition_dict = self.dict_joiner(self.transition_dict, other_model.transition_dict)
        self.lambda_dict = self.dict_joiner(self.lambda_dict, other_model.lambda_dict)
        self.t_l += other_model.t_l

    @staticmethod
    def dict_joiner(first_dict, second_dict):
        # type: (dict, dict) -> dict

        keyval = max(first_dict.keys())
        keyval += 1
        second_dict_keys = sorted(second_dict.keys())
        for k in second_dict_keys:
            first_dict[keyval] = second_dict[k]
            keyval += 1
        return first_dict

    def evaluation_pipeline(self, data_file, pickled=None, to_keep=None):
        # type: (str|srcUT.Data_class.DataClass, str, list) -> None
        """
        This is a pipeline for evaluating an MCMC run. The results are printed to the screen and not returned.

        :param data_file: either a path describing where to find the data file or an instance of DataClass
        :type data_file: str|srcUT.Data_class.DataClass
        :param pickled: path to a pickled version of the gold standard data
        :type pickled: str
        :param to_keep: list indicating the languages which are evaluated if only a subset of the data is tested
        :type to_keep: list
        :return: Nothing is returned. The results are printed
        :rtype: None
        """
        if isinstance(data_file, str):
            if self.dialect == "asjp":
                header = ["iso_code", "gloss_global", "ASJP", "cog_class_ielex"]
            elif self.dialect == "ipa":
                header = ["iso_code", "gloss_global", "IPA", "cog_class_ielex"]
            else:
                raise Exception("unknown dialect")
            ldn = 1.0
            cc_sample = False
            to_keep = [] if to_keep is None else to_keep
            data = srcUTData.DataClass.create_data(data_file=data_file, to_keep=to_keep, header=header,
                                                   sound_model=self.dialect, ldn=ldn, cc_sam=cc_sample)
        elif isinstance(data_file, srcUTData.DataClass):
            data = data_file
        else:
            raise Exception("Data File is not in proper format. Consider just passing a string with the location of "
                            "the data file")

        if pickled is not None:
            assert isinstance(pickled, str)
            with open(pickled, 'rb') as handle:
                gs_cluster = pickle.load(handle)
        else:
            gs_cluster = data.cognate_clusters()
        cut = [percentile_cut / 100.0 for percentile_cut in range(90, 100, 2)]
        scores, o2 = self.calculate_scores(data, cut, size=1000, gs=gs_cluster)
        '''
        choose method here
        '''
        info = False
        '''
        true = infomap
        false = label propagation
        '''

        if info is True:
            print "infomap"
        else:
            print "label prob"
        for percentile_cut in cut:
            print("percentile = " + str(percentile_cut))
            for ic in range(75, 100, 5):
                infomap_cut = ic / 100.0
                statement_dict = self.extract_cognacy_statements(data, scores[percentile_cut], infomap_cut, info=info)

                print("b_cubed " + str(infomap_cut))
                p, r, f = Clust.calc_f_score(gs_cluster, pred_clusters=statement_dict)

                print(np.mean(p), np.mean(r), np.mean(f))
                print("-----")
            #with open("statement_dict" + str(percentile_cut) + ".pickle", "wb") as out:
            #    pickle.dump(statement_dict, out)
            #p2 = [v[percentile_cut][0] for k,v in o2.iteritems()]
            #r2 = [v[percentile_cut][1] for k, v in o2.iteritems()]
            #f2 = [v[percentile_cut][2] for k, v in o2.iteritems()]
            #print(np.median(p2), np.median(r2), np.median(f2))
            #print(np.mean(p2), np.mean(r2), np.mean(f2))
            #print(np.std(p2), np.std(r2), np.std(f2))

            print("=====")

    @staticmethod
    def extract_cognacy_statements(data, estimates, infomap_cut, info):
        # type: (srcUT.Data_class.DataClass, np.ndarray, float, bool) -> dict
        """
        This function extracts cognate clusters from the pairwise cognacy probabilities stored in estimates. It uses the
        infomap algorithm.

        :param info:
        :param data: instance of the Data class which stores the gold standard information
        :type data: srcUT.Data_class.DataClass
        :param estimates: numpy array storing the posterior probability information over cognates
        :type estimates: np.ndarray
        :param infomap_cut: probabilities which are less then this threshold are ignored
        :type infomap_cut: float
        :return: dictionary over cognate classes
        :rtype: dict
        """
        statement_dict = {}
        if info is True:
            method = 'infomap'
        else:
            method = 'labelprop'
        for ind, concept in enumerate(data.concepts):
            statement_mat = estimates[:, :, ind]
            # present_langs = [k for k in data.languages if k in [j for i in gs_cluster[concept].items() for j in i[1]]]
            # rm_inds = [ind for ind, itm in enumerate(data.languages) if itm not in present_langs]
            # statement_mat_new = np.delete(np.delete(statement_mat, rm_inds,0), rm_inds, 1)
            cog_sets = collections.defaultdict(set)
            cluster = Clust.igraph_clustering(statement_mat, infomap_cut, method="labelprop")
            for lang, label in cluster.iteritems():
                cog_sets[label].add(data.languages[lang])
            statement_dict[concept] = cog_sets
        return statement_dict

    @staticmethod
    def construct_3d_statement_array(data, percentile_cut, scores):
        # type: (srcUT.Data_class.DataClass, float,  dict) -> np.ndarray
        """
        This function takes the pairwise alignment scores in a dictionary and returns a three-dimensional numpy array of
        the pairwise cognate statements as probabilities

        :param data: instance of the Data class which stores the data set
        :type data: srcUT.Data_class.DataClass
        :param percentile_cut: percentile of the the off-diagonal scores be used to calculate the threshold of cognacy
        :type percentile_cut: float
        :param scores: dictionary storing the alignment scores
        :type scores: dict
        :return: three dimensional numpy array of pairwise cognate statements per concept
        :rtype: np.ndarray
        """

        estim = np.zeros((data.n_concepts, data.n_languages, data.n_languages), dtype=np.float64)
        for draw_num, scores_dict in scores.iteritems():
            for (lang1, lang2), (scores_on, scores_off) in scores_dict.iteritems():
                ind = int(len(scores_off) * percentile_cut)

                threshold = np.sort(scores_off, kind="mergesort")[ind]

                l1_ind = data.languages.index(lang1)
                l2_ind = data.languages.index(lang2)
                estim[:, l1_ind, l2_ind] += np.greater(scores_on, threshold)
        return estim / len(scores)

    def create_sound_model_dict(self):
        # type: () -> dict
        """
        Return a dictionary of emission models

        :return: dictionary containing the class building information for each sampled iteration
        :rtype: dict
        """
        sound_mod_pd_thinned = self.thin(self.sound_mod_pd_raw)

        header = sound_mod_pd_thinned.columns.tolist()

        # header for evo class columns
        evo_class_value_header = [i for i in header if i.split("_")[0] == "clv"]

        # frequencies column names
        freq_header = [i for i in header if i.split("_")[0] == "freq"]

        # indices of the class value for the class
        class_indices = self.sound_classes_pd.values[0].tolist()

        # get the names of the sounds
        names = [i.split("_")[1] for i in freq_header]
        names = [i.decode("utf-8") for i in names]
        dct = {}
        k = 0
        for ind, row in sound_mod_pd_thinned.iterrows():
            dct[k] = {"names": names,
                      "freqs": np.array([row[i] for i in freq_header]),
                      "evo_map": class_indices,
                      "evo_vals": [row[i] for i in evo_class_value_header],
                      "model": self.dialect}
            k += 1

        return dct

    def create_trans_dict_multi(self):
        # type: () -> dict
        """
        Get the relevant transition model parameters in a dictionary

        :return: dictionary of model parameters
        :rtype: dict
        """
        tr_pd_thinned = self.thin(self.tr_pd_raw)
        head = tr_pd_thinned.columns.tolist()
        dct = {}
        k = 0
        for ind, row in tr_pd_thinned.iterrows():
            dct[k] = {i.split("_")[1]: row[i] for i in head}
            k += 1
        return dct

    def calculate_scores(self, data, percentile_list, gs, size=800):
        # type: (srcUT.Data_class.DataClass,list, int) -> dict
        """
        This function calculates the alignment scores for all the data which needs to be evaluated

        :param data: language data
        :return: dictionary of alignment scores for different model instances
        :rtype: dict
        """

        langs, _ = self.tree_dist_dict[self.emission_dict.keys()[0]]
        out_dict = {p: np.zeros((len(langs), len(langs), data.n_concepts), dtype=np.double) for p in percentile_list}
        o2 = {k: {} for k in self.emission_dict.keys()}
        for k, v in tqdm(self.emission_dict.iteritems(), total=len(self.emission_dict.keys())):
            emmod = srcEM.Feature_Single.FeatureSoundsSingle.from_dict(v)
            em = srcEM.EmissionModel_Sound.EmissionModelSound(alphabet=emmod.names,
                                                              sound_mod=emmod)
            if self.trans_mod == "km03":
                tr = srcTR.Trans_cy_wrap.Trans_KM03(time=1.0, **self.transition_dict[k])
            elif self.trans_mod == "tkf92":
                tr = srcTR.Trans_cy_wrap.Trans_TKF92(time=1.0, **self.transition_dict[k])
            elif self.trans_mod == "tkf91":
                tr = srcTR.Trans_cy_wrap.Trans_TKF91(time=1.0, **self.transition_dict[k])
            else:
                raise Exception("Transition Model unknown")

            em.sound_model.diagonalize_q_matrix()
            langs, time_mat = self.tree_dist_dict[k]
            lam = self.lambda_dict[k]
            ts = self.t_l[k]*2
            for (ind1, l1), (ind2, l2) in itertools.combinations(enumerate(langs), 2):
                l1_ind = data.languages.index(l1)
                l2_ind = data.languages.index(l2)
                wl1, wl2, _, _ = data.get_diagonal((l1, l2))
                diag_words = (wl1, wl2)
                off_diag_words = data.get_off_diagonal((l1, l2), size)
                t2 = time_mat[ind1][ind2]

                scores_on, scores_off = self.calculate_alignment_scores(em, tr, ts, diag_words,
                                                                        off_diag_words, self.trans_mod)
                scores_on -= t2*lam
                scores_off += np.log(1-np.exp(-t2*lam))
                for percentile in percentile_list:
                    ind = int(len(scores_off) * percentile)

                    threshold = np.sort(scores_off, kind="mergesort")[ind]
                    statements = np.greater(scores_on, threshold)
                    out_dict[percentile][l1_ind, l2_ind] += statements
                    out_dict[percentile][l2_ind, l1_ind] += statements

            #for percentile in percentile_list:
            #    statement_dict = self.extract_cognacy_statements(data, out_dict[percentile],
            #                                                     0.0, gs)
            #    p, r, f = Clust.calc_f_score(gs, pred_clusters=statement_dict)
            #    o2[k][percentile] = (np.mean(p), np.mean(r), np.mean(f))
        out_dict = {k: v / len(self.emission_dict.keys()) for k, v in out_dict.iteritems()}
        return out_dict, o2

    @staticmethod
    def calculate_alignment_scores(em, tr, t, diag_words, off_diag_words, tr_mod):
        # type: (srcEM.EmissionModel_Sound, srcTR.Trans_km03_cy, np.ndarray, tuple, tuple, str) -> (np.ndarray, np.ndarray)
        """
        Calculate the alignment scores for a particular language pair

        :param em: emission model
        :type em: srcEM.EmissionModel
        :param tr: transition model
        :param t: evolutionary time
        :type t: np.ndarray
        :param diag_words: supposedly cognate words
        :type: diag_words: tuple
        :param off_diag_words: supposedly non-cognate words to simulate baseline
        :type off_diag_words: tuple
        :param tr_mod: identifier for transition model
        :type: str
        :return: on and off diagonal scores
        :rtype: tuple
        """

        # combt = sum(t)
        #t = 1.0
        tr.set_model(t)
        em.create_model(time=t)
        # d = pd.DataFrame(em.emission_mat_log-np.log(np.outer(em.sound_model.frequencies,em.sound_model.frequencies)), columns=em.alphabet)
        # d = pd.DataFrame(em.emission_mat_log, columns=em.alphabet)
        # sns.clustermap(d, center=0, cmap="vlag", linewidths=.75, figsize=(13, 13))
        # plt.show()
        if tr_mod == "km03":
            res_on = calculaute_odds_scores(diag_words, em.emission_mat_log, em.gap_log, em.gap_log, tr.trans_vals,
                                            len(em.gap_log))
            res_off = calculaute_odds_scores(off_diag_words, em.emission_mat_log, em.gap_log, em.gap_log, tr.trans_vals,
                                             len(em.gap_log))

        elif tr_mod == "tkf92":
            res_on = calculaute_odds_scores_tkf(diag_words, em.emission_mat_log, em.gap_log, em.gap_log, tr.trans_vals,
                                                tr.r, tr.l, tr.mu, t)
            res_off = calculaute_odds_scores_tkf(off_diag_words, em.emission_mat_log, em.gap_log, em.gap_log,
                                                 tr.trans_vals, tr.r, tr.l, tr.mu, t)
        elif tr_mod == "tkf91":
            res_on = calculaute_odds_scores_tkf(diag_words, em.emission_mat_log, em.gap_log, em.gap_log, tr.trans_vals,
                                                tr.l, tr.mu, t)
            res_off = calculaute_odds_scores_tkf(off_diag_words, em.emission_mat_log, em.gap_log, em.gap_log,
                                                 tr.trans_vals, tr.l, tr.mu, t)
        else:
            raise Exception("Transition Model not known")

        inds = res_off == -np.inf
        res_off = res_off[~inds]

        return res_on, res_off

    def medain_emission_plot(self, t=1):

        names_ = self.emission_dict[self.emission_dict.keys()[0]]["names"]
        t = np.mean(self.t_l)
        l = len(names_)
        em_store = np.zeros((l, l))
        f_store = np.zeros(l)
        for k, v in self.emission_dict.iteritems():
            emmod = srcEM.Feature_Single.FeatureSoundsSingle.from_dict(v)
            em = srcEM.EmissionModel_Sound.EmissionModelSound(alphabet=emmod.names,
                                                              sound_mod=emmod)
            em.sound_model.diagonalize_q_matrix()
            em.create_model(time=t)
            em_store += em.emission_mat_log
            f_store += em.sound_model.frequencies
        em_store /= len(self.emission_dict.keys())
        f_store /= len(self.emission_dict.keys())

        d = pd.DataFrame(em_store - np.log(np.outer(f_store, f_store)), columns=names_, index=names_)
        d = np.max(d.values) - d
        # d = pd.DataFrame(em_store, columns=em.alphabet)
        # d.corr()
        # d = np.exp(d1)
        # d /= (1 + d)

        # d = spspdi.squareform(d, checks=False)
        # d = 1 - d
        # linkage = hierarchy.linkage(d, "complete")
        # dendr = hierarchy.dendrogram(linkage)
        sns.clustermap(d,
                       # 1, row_linkage=linkage, col_linkage=linkage,
                       center=np.mean(d.values), cmap="vlag", linewidths=.75,
                       method="ward",
                       figsize=(13, 13)
                       )
        plt.show()
    @staticmethod
    def v_merge(in_folder, out_folder, iterations=True, thin=1):
        merge1 = Evaluator.merge(in_folder, out_folder, iterations=iterations, thin=thin)
        merge2 = Evaluator.merge(in_folder[:-1]+"_contd/", out_folder, iterations=iterations, thin=thin)
        merge2["it"] += (max(merge1["it"])+1000)
        res = pd.concat([merge1, merge2])
        res.to_csv(out_folder, sep="\t", index=False)
    @staticmethod
    def merge(in_folder, out_folder, iterations=True, thin=1):
        # type: (str, str, bool) -> None
        """
        Writes a merged version of the transition, likelihood and sound model file into a new file.
        For analysis with the coda R package set iterations to False

        :param in_folder: folder where the files are stored
        :type in_folder: str
        :param out_folder: filename of the merged version of the file
        :type out_folder: str
        :param iterations: indicate if iteration number should be printed as first column or not
        :type iterations: bool
        :return: Nothing is returned. The resulting file is written to the specified location
        :rtype: None
        """

        tr_filename = in_folder + "tr_mod.log"
        sound_mod_filename = in_folder + "sound_mod.log"

        tree_filename = in_folder + "MCMC_test.trees_vals.log"
        likelihood_filename = in_folder + "MCMC_test.params.log"
        lambda_filename = in_folder + "MCMC_test.lambda.log"


        up_to = min([Evaluator.file_len(tree_filename),
                     Evaluator.file_len(tr_filename),
                     Evaluator.file_len(sound_mod_filename),
                     Evaluator.file_len(likelihood_filename),
                     Evaluator.file_len(lambda_filename)])
        # _, _, tree_heights_raw = Evaluator.read_tree(tree_filename, skip_rows=None)
        tree_stats_raw = Evaluator.read_file(tree_filename, nrows=up_to, dtype=np.float64).iloc[::thin]
        sound_mod_pd_raw = Evaluator.read_file(sound_mod_filename, nrows=up_to, dtype=np.float64).iloc[::thin]

        tr_pd_raw = Evaluator.read_file(tr_filename, nrows=up_to, dtype=np.float64).iloc[::thin]
        likelihood_raw = Evaluator.read_file(likelihood_filename, nrows=up_to).iloc[::thin]
        lambda_raw = Evaluator.read_file(lambda_filename, nrows=up_to, dtype=np.float64).iloc[::thin]
        likelihood_raw["it"] = likelihood_raw["it"].apply(lambda x: x*50)
        if iterations:
            comb = likelihood_raw.join(tr_pd_raw).join(sound_mod_pd_raw).join(
                tree_stats_raw[["tree_height", "tree_length"]]).join(lambda_raw)
        else:
            comb = tr_pd_raw.join(likelihood_raw["lik"]).join(sound_mod_pd_raw).join(
                tree_stats_raw[["tree_height", "tree_length"]]).join(lambda_raw)

        comb.to_csv(out_folder, sep="\t", index=False)
        return comb
    def check_kwargs(self, **kwargs):
        """

        :param kwargs: kwargs
        """
        if "trans" in kwargs:
            self.tr_filename = self.folder + kwargs["trans"]
        if "sound_mod" in kwargs:
            self.sound_mod_filename = self.folder + kwargs["sound_mod"]
        if "sound_classes" in kwargs:
            self.sound_classes_filename = self.folder + kwargs["sound_classes"]
        if "tree" in kwargs:
            self.tree_filename = self.folder + kwargs["tree"]
        if "likelihood" in kwargs:
            self.likelihood_filename = self.folder + kwargs["likelihood"]

    @staticmethod
    def skip_idx_func(start, end, step):
        # type: (int, int, int) -> np.ndarray
        """
        Calculates the rows which need to be skipped while reading the target file

        :param start: start row
        :param end: end row
        :param step: step size
        :return: array of rows to skip
        """

        skip_idx = list(range(0, start))
        idx = start + 1
        while idx < end:
            if idx % step != 0:
                skip_idx.append(idx)
            idx += 1
        return np.array(skip_idx)

    def write_tree_sample(self, tree_target_file):
        """
        This functions writes the tree sample to a file. Each line in the resulting file holds a tree in
        newick format.

        :param tree_target_file: filename for the tree sample file
        :type tree_target_file: str
        """
        with open(tree_target_file, "w") as outf:
            for tree_string in self.t_l:
                tree2 = tree_string.split("\t")[1].strip()
                outf.write(tree2 + "\n")

    @staticmethod
    def read_tree(my_file, skip_rows):
        # type: (str, np.ndarray|None) -> tuple
        """
        This function reads in the tree file and returns a dictionary of arrays which stores information of distances
        within a tree

        :param my_file: filename of the tree file
        :type my_file: str
        :param skip_rows: array of rows in the tree file which are omitted while reading
        :type skip_rows: np.ndarray|None
        :return: tuple of several relevant information
        :rtype: tuple
        """
        store = {}
        if skip_rows is None:
            skip_idx = None
            to_write = False
        else:
            to_write = True
            skip_idx = set(skip_rows)

        t_l = []
        tree_idx = 0
        with open(my_file, "r") as tree_input:

            for k, tree_string_raw in enumerate(tree_input):

                if to_write:
                    if k not in skip_idx:

                        tr = dendropy.Tree.get_from_string(tree_string_raw.split("\t")[1], schema="newick")
                        t = dendropy.PhylogeneticDistanceMatrix.from_tree(tr)
                        l = tr.length()/len(tr.edges())
                        t_l.append(l)
                        # t = Phylo.read(tree_string, format="newick", rooted=True)
                        # t_l.append(tree_string_raw)
                        term = t.taxon_namespace
                        arr = np.zeros((len(term), len(term)))
                        for (ind, itm), (jnd, jtm) in itertools.combinations(enumerate(term), 2):
                            d = t.distance(itm, jtm)
                            arr[ind][jnd] = d

                            arr[jnd][ind] = d

                        term2 = [i.label for i in term]
                        store[tree_idx] = (term2, arr)
                        tree_idx += 1

        return store, t_l, None

    def thin(self, data_frame):
        # type: (pd.DataFrame) -> pd.DataFrame
        """
        Thin the data frame for analysis

        :param data_frame: Data frame holding the raw information
        :type data_frame: pd.DataFrame
        :return: pd.DataFrame
        """
        return data_frame.drop(self.skip_rows, inplace=False)

    @staticmethod
    def read_file(input_file, **kwargs):
        # type: (str, any) -> pd.DataFrame
        """
        reads a parameter file

        :param input_file: file with information to be read
        :type input_file: str
        :return: pandas data frame of the information in the file
        :rtype: pd.DataFrame
        """

        return pd.read_csv(input_file, sep="\t", engine="c", na_filter=False, memory_map=True, **kwargs)

    @staticmethod
    def file_len(filename):
        # type: (str) -> int
        """
        calculates the amount of rows in a file

        :param filename: filename
        :return: number of rows
        """
        i = 0
        with open(filename, "r") as f:
            for i, l in enumerate(f):
                pass
        return i + 1

    def read_cognate_map(self, filename):
        mapping = {}
        with open("mapping.pickle", "rb") as infile:
            nlex_to_ielex = pickle.load(infile)
        with open(filename, "r") as infile:
            for line in infile:
                cont = line.strip().split()
                conc = cont[1].decode("utf-8")
                mapping[nlex_to_ielex[conc]] = int(cont[0])
        return self.read_cognate_classes(mapping)

    def read_cognate_classes(self, mapping):
        # type: (dict) -> tuple

        ret_dict = collections.defaultdict()

        for concept, ind in mapping.iteritems():
            ret_dict[concept] = {}
            with open(self.folder + "cognate_sample.log_" + str(ind), "r") as infile:
                ct_it = 0
                for it, line in enumerate(infile):
                    if it in self.skip_rows:
                        pass
                    else:
                        inner = {}
                        classes = line.strip().split("\t")
                        for c_c, els in enumerate(classes):
                            langs = els.split(",")[:-1]
                            inner[c_c] = langs
                        ret_dict[concept][ct_it] = inner
                        ct_it += 1
        second_ret_dict = {}
        for k, v in ret_dict.iteritems():
            for k1, v1 in v.iteritems():
                if k1 in second_ret_dict.keys():
                    second_ret_dict[k1][k] = v1
                else:
                    second_ret_dict[k1] = {k: v1}

        with open("gs_dict.pickle", 'rb') as handle:
            gs_cluster = pickle.load(handle)
        p_l = []
        r_l = []
        f_l = []
        for it, cluster in second_ret_dict.iteritems():
            if len(cluster.keys()) != len(gs_cluster.keys()):
                pass
            else:
                p, r, f = Clust.calc_f_score(gs_cluster, cluster)
                p_l.append(p)
                r_l.append(r)
                f_l.append(f)
        return p_l, r_l, f_l
