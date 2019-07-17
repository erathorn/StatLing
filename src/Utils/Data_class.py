"""
@author: erathorn
@date: July 2019
@version: 1.0

"""
import collections
import itertools
import math
import random

import numpy as np
import pandas as pd

import ltrees
import src.EmissionModel.Alphabets as Alphas
import src.SamplerSettings.Prior as Prior
import src.Tree.Tree as sc_Tree

# This variable determines the size of the vector holding the translation of a word
vec_size = 35  # type: int
initial_classes = 8  # type: int
alpha = 50  # 0.5

# These are the default column names in the data file
concept_col = "gloss_global"
lang_col = "iso_code"
transcription_col = "phon_northeuralex"
cognate_class_col = "cog_class_ielex"


class DataClass(object):
    """
    This Class is responsible for the data in the MCMC computation
    """
    header = [lang_col, concept_col, transcription_col, cognate_class_col] if cognate_class_col is not None else [
        lang_col, concept_col, transcription_col]

    __slots__ = ("concepts", "n_concepts", "language_data", "languages", "language_pairs",
                 "lpair_ind_dict", "lang_ind_dict", "alphabet", "alphabet_size", "related_scores",
                 "_diag_dict", "expected_lengths", "right_stack", "left_stack", "all_words", "ind_lpair_dict", "unrel",
                 "cognacy_classes", "tree", "sampling_functions", "off_diag_dict", "el")

    def __init__(self, language_data, concepts, sound_model, data_diag=True,
                 ldn_cut=0.5, cc_class=False, check_consistency=True, pre_tree=None):
        """
        Creates an instance of DataClass

        :param language_data: A dictionary of languages. The key is an identifier and the value a list of translations
        :type language_data: dict
        :param concepts: A list of concepts for which there are translations
        :type concepts: List[unicode]
        :param sound_model: specify the sound model, ipa or asjp
        :type sound_model: str
        :param data_diag: should the data be stored in a diagonal format
        :type data_diag: bool
        :param ldn_cut: cutoff of ldn scores. Each word pair with an ldn score lower than this is considered.
        :type ldn_cut: float
        :param cc_class: indicate if cognate classes should be sampled (deprecated)
        :type cc_class: bool
        :param check_consistency: indicate if the consistency of the tree topology with a certain constrains is checked
        :type check_consistency: bool
        :param pre_tree: Tree from specified starting point
        :type pre_tree: str
        """

        self.concepts = concepts
        self.n_concepts = len(concepts)
        self.language_data = language_data
        self.languages = sorted(self.language_data.keys())

        if data_diag is True:
            self.language_pairs = [tuple(i) for i in itertools.product(self.languages, self.languages)]
        else:
            self.language_pairs = [tuple(i) for i in itertools.combinations(self.languages, 2)]

        # create language pair index map
        self.lpair_ind_dict = collections.defaultdict()
        for ind, itm in enumerate(self.language_pairs):
            self.lpair_ind_dict[itm] = ind
            self.lpair_ind_dict[(itm[1], itm[0])] = ind
        self.ind_lpair_dict = {v: k for k, v in self.lpair_ind_dict.iteritems()}
        self.lang_ind_dict = {itm: ind for ind, itm in enumerate(self.languages)}

        if sound_model == "ipa":
            self.alphabet = sorted([i.decode("utf-8") for i in Alphas.alph_ipa])
        elif sound_model == "asjp":
            self.alphabet = sorted([i.encode("utf-8") for i in Alphas.alph_asjp])
        else:
            raise Exception("Sound Model " + str(sound_model) + " not known")

        self.alphabet_size = len(self.alphabet)

        self.related_scores = np.zeros((len(self.language_pairs), self.n_concepts, 2), dtype=np.double, order="c")
        self.unrel = None
        self._diag_dict = {}

        self.expected_lengths = {}
        self.right_stack = np.empty((len(self.language_pairs), self.n_concepts, vec_size), dtype=np.int8)
        self.left_stack = np.empty((len(self.language_pairs), self.n_concepts, vec_size), dtype=np.int8)
        self.all_words = np.empty((self.n_languages, self.n_concepts, vec_size), dtype=np.int8)

        self.translate_words()

        self.cognacy_classes = None
        self.tree = None

        self.create_tree(extend_arity=cc_class, check_consistency=check_consistency, pre_tree=pre_tree)

        if cc_class is False:
            self.pre_calc_diagonal_ldn(ldn_cut=ldn_cut)

            self.sampling_functions = []
        else:
            self.cognacy_classes = np.random.randint(initial_classes, size=(self.n_languages, self.n_concepts))
            self.pre_calc_diagonal_ldn(ldn_cut=ldn_cut)
            self.sampling_functions = [(self.sample_class_assignments, self.revert, "cognacy", {"concept_ind": i}) for i
                                       in range(self.n_concepts)]
            raise DeprecationWarning("Sampling of cognate classes is deprecated. Might be deleted in future versions")
        if data_diag is True:
            self.pre_calc_single()
        else:
            self.pre_calc_stack()
            self.pre_calc_single()
        self.calculate_el()
        self.off_diag_dict = {}

    @property
    def n_languages(self):
        return len(self.language_data.keys())

    def __getitem__(self, item):
        return self.language_data[item]

    def calculate_el(self):
        """
        calculate the expected length as the mean of the word lengths
        """
        for lpair in self.language_pairs:
            x = []
            wl1, wl2, _, _ = self.get_diagonal(lpair)
            for i, j in zip(wl1, wl2):
                if i[-1] != -1 and j[-1] != -1:
                    x.append((i[-1] + j[-1]) / 2.0)

            self.expected_lengths[lpair] = np.mean(x)
        ls = []
        for language, data in self.language_data.items():
            for word in data.wordlist:
                if word is not None:
                    ls.append(len(word))
        self.el = np.mean(ls)

    def translate_words(self):
        """
        translate the words into number representation
        """
        for l in self.languages:
            self[l].translate(self.alphabet)

    def get_diagonal(self, lpair):
        """
        Get the word pairs along the diagonal of the matrix, i.e. it returns the word pair describing the same concept

        :param lpair: language pair in question
        :type lpair: tuple
        :return: tuple of word list, word list, calculation info, calculation info
        :rtype: tuple
        """
        return self._diag_dict[lpair]

    def pre_calc_single(self):
        """
        prepare words for the parallel C-code

        probably outdated
        """
        cont = []
        for language in self.languages:
            cont.append(self[language].translated)

        np.stack(cont, axis=0, out=self.all_words)

    def pre_calc_stack(self):
        """
        prepare the word pairs for the parallel C-code
        """

        cont_l = []
        cont_r = []
        for lpair in self.language_pairs:
            wl1, wl2, judgements_l1, judgements_l2 = self.get_diagonal(lpair)
            wl1[:, -2] = judgements_l1
            wl2[:, -2] = judgements_l2
            cont_l.append(wl1.copy())
            cont_r.append(wl2.copy())
        np.stack(cont_l, axis=0, out=self.left_stack)
        np.stack(cont_r, axis=0, out=self.right_stack)

    def pre_calc_diagonal_cc(self):
        """
        prepare the word pairs for the parallel C-code (used for cognate class sampling)
        """
        for lpair in self.language_pairs:
            l1, l2 = lpair
            l1_ind = self.languages.index(l1)
            l2_ind = self.languages.index(l2)
            cc_list_l1 = self.cognacy_classes[l1_ind]
            cc_list_l2 = self.cognacy_classes[l2_ind]
            self._diag_dict[(l1, l2)] = (self[l1].translated, self[l2].translated, cc_list_l1, cc_list_l2)
            self._diag_dict[(l2, l1)] = (self[l2].translated, self[l1].translated, cc_list_l2, cc_list_l1)

    def pre_calc_diagonal_ldn(self, ldn_cut):
        # type: (float) -> None
        """
        Calculate the word pair information once, so time is saved.

        :param ldn_cut: cutoff of ldn scores. Each word pair with an ldn score lower than this is considered.
        :type ldn_cut: float
        """
        for lpair in self.language_pairs:
            l1, l2 = lpair
            wl1 = self[l1].wordlist
            wl2 = self[l2].wordlist
            ldn_list1 = []
            ldn_list2 = []
            for ind, (w1, w2) in enumerate(zip(wl1, wl2)):
                if w1 is None or w2 is None:
                    ldn_list1.append(0)
                    ldn_list2.append(1)
                else:
                    if self.ldn(w1, w2) <= ldn_cut:
                        # include this word into the calculation
                        ldn_list1.append(1)
                        ldn_list2.append(1)
                    else:
                        # exclude this word from the calculation
                        ldn_list1.append(0)
                        ldn_list2.append(1)

            self._diag_dict[(l1, l2)] = (
                self[l1].translated, self[l2].translated, np.array(ldn_list1), np.array(ldn_list2))
            self._diag_dict[(l2, l1)] = (
                self[l2].translated, self[l1].translated, np.array(ldn_list1), np.array(ldn_list2))

    def cognate_clusters(self):
        # type: () -> dict
        """
        Returns a dictionary of gold standard cognate information

        :return: dictionary storing gold standard cognate information
        :rtype: dict
        """
        ret_dict = {}
        for ind, conc in enumerate(self.concepts):
            cluster = collections.defaultdict(set)
            for lang in self.languages:
                cog_class = self.language_data[lang].get_cognacy(ind)[1]
                if cog_class is not None:
                    if "/" in cog_class:
                        cog_class = cog_class.split("/")[0]
                    cluster[cog_class].add(lang)

            ret_dict[conc] = cluster
        return ret_dict

    def get_off_diagonal(self, lpair, size=None):
        # type: (tuple, None|int) -> tuple
        """
        get off diagonal entries of the word matrix

        :param lpair: language pair specifying the word matrix
        :type lpair: tuple
        :param size: number of word pairs to extract
        :type size: None|int
        :return: tuple of word pairs
        :rtype: tuple
        """

        if lpair in self.off_diag_dict.keys():
            return self.off_diag_dict[lpair]

        indices = np.triu_indices(self.n_concepts, 1)

        l1, l2 = lpair
        if size is None:
            self.off_diag_dict[lpair] = self[l1][indices[0]], self[l2][indices[1]]
            return self[l1][indices[0]], self[l2][indices[1]]
        else:
            ind_list = range(len(indices[0]))

            l_ind = indices[0][ind_list[:size]]
            r_ind = indices[1][ind_list[:size]]
            self.off_diag_dict[lpair] = self[l1][l_ind], self[l2][r_ind]
            return self[l1][l_ind], self[l2][r_ind]

    def create_tree(self, extend_arity, check_consistency, pre_tree=None):
        # type: (bool, bool, str) -> None
        """
        Create a tree (currently based on a reference tree)

        :param extend_arity: should arity be extended due to cognate class sampling
        :type extend_arity: bool
        :param check_consistency: indicate if the consistency of the tree topology with a certain constrains is checked
        :type check_consistency: bool
        :param pre_tree: tree from speciefied starting state
        :type: str
        """
        if pre_tree is not None:
            my_ref_tree = pre_tree
        else:
            # use starting tree present in ltrees.IE_tree
            # might change later
            my_ref_tree = ltrees.IE_tree

        self.tree = sc_Tree.Tree.from_newick(my_ref_tree)

        tree_it = self.tree.post_order

        for node in tree_it:
            if node.terminal:
                lang = node.name
                node.data = self.language_data[lang]
                inds = np.where(node.data.translated[:, 0] == -1)
                if extend_arity:
                    inds = inds if len(inds[0]) == 0 else [(i * Prior.n_cats) + j for i in inds[0] for j in
                                                           range(Prior.n_cats)]
                l_score_store = len(node.data.translated) * Prior.n_cats if extend_arity else len(node.data.translated)
                node.arity = np.full(l_score_store, 1.0)
                node.arity[inds] = 0.0
                node.likelihood_rel = 0.0

        self.tree.lpair_mapping = self.lpair_ind_dict
        self.tree.time_store = np.empty(len(self.language_pairs), dtype=np.double)

        self.tree.set_binary()
        self.tree.calc_po = True
        self.tree.recalculate_time()
        self.tree.check_for_consistency = check_consistency

    def sample_class_assignments(self, concept_ind):
        # type: (int) -> tuple
        """
        Sample cognate class assignments

        Still here fore legacy reasons

        :param concept_ind: index of  the concept
        :type concept_ind: int
        :return: information for metropolis hastings sampling and previous state
        :rtype: tuple  
        """
        prev = self.cognacy_classes.copy()

        # concept_ind = np.random.randint(self.n_concepts, size=1)
        column = self.cognacy_classes[:, concept_ind]
        language_ind_1 = 0
        language_ind_2 = 0
        while language_ind_1 == language_ind_2:
            language_ind_1, language_ind_2 = np.random.randint(self.n_languages, size=2)

        if column[language_ind_1] == column[language_ind_2]:
            # split is desired
            # a split always creates a new class

            new_class_identifier = next(itertools.ifilterfalse(set(column).__contains__, itertools.count(1)))
            current_class = np.where(column == column[language_ind_1])[0]
            column[language_ind_2] = new_class_identifier
            i_split = 0
            j_split = 0
            for entry in current_class:
                if entry == language_ind_1 or entry == language_ind_2:
                    pass
                else:
                    if random.choice([True, False]):
                        i_split += 1
                        column[entry] = new_class_identifier
                    else:
                        j_split += 1

            prior = alpha * (math.factorial(i_split) * math.factorial(j_split)) / math.factorial(i_split + j_split + 1)
            # prior = spst.norm.logpdf(len(set(column[:, 0])), 4)
            m_h_ratio = 1.0 - np.log(0.5) * (i_split + j_split)
        else:
            # merge is desired
            current_class = np.where(column == column[language_ind_1])[0]
            current_class_2 = np.where(column == column[language_ind_2])[0]
            column[current_class] = column[language_ind_2]

            cl_1_size = current_class.size
            cl_2_size = current_class_2.size
            prior = (1.0 / alpha) * math.factorial(cl_1_size + cl_2_size - 1) / (
                    math.factorial(cl_2_size - 1) * math.factorial(cl_1_size - 1))
            # prior = spst.norm.logpdf(len(set(column[:,0])), 4)
            m_h_ratio = np.log(0.5) * (cl_1_size + cl_2_size - 2)

        # make labels continuous
        mapping = {itm: ind for ind, itm in enumerate(set(column))}
        for i in range(self.n_languages):
            column[i] = mapping[column[i]]
        self.cognacy_classes[:, concept_ind] = column
        self.binarise_cognate_judgements()
        return True, m_h_ratio, np.log(prior), prev

    def reset_arity(self, lang_indices, cc_indices):
        # type: (list, list) -> None
        """
        Reset the arity of the nodes. This may change due to cognate class sampling

        still here fore legacy reasons

        :param lang_indices: list of indices indicating the languages
        :type lang_indices: list
        :param cc_indices: list of indices indicating the cognate classes
        :type cc_indices: list
        """
        for ind, ind2 in itertools.product(lang_indices, cc_indices):
            arity = self.tree._name_node_map[self.languages[ind]].arity
            arity[ind2 * Prior.n_cats:ind2 * Prior.n_cats + Prior.n_cats] = 0
            c = self.cognacy_classes[ind][ind2]
            arity[(ind2 * Prior.n_cats) + c] = 1

    def revert(self, prev):
        # type: (np.ndarray) -> None
        """
        revert a cognate class sampling move

        Still here for legacy reasons

        :param prev: previous state
        :type prev: np.ndarray
        """
        self.cognacy_classes = prev
        self.binarise_cognate_judgements()

    def write_cognate_map(self, file_name):
        # type: (str) -> None
        """
        Write the cognate list to a file to recover information about cognate class samples

        Still here for legacy reasons

        :param file_name: target file
        :type file_name: str
        """

        with open(file_name, "w") as outfile:
            for ind, con in enumerate(self.concepts):
                outfile.write(str(ind) + "\t" + str(con) + "\n")

    def to_file(self, file_name):
        # type: (str) -> None
        """
        Write the cognate class information to a file

        :param file_name: filename where the cognate class information should be written to
        :type file_name: str
        """

        for concept_ind in range(self.n_concepts):
            with open(file_name + "_" + str(concept_ind), "a") as out:
                all_classes = set(self.cognacy_classes[:, concept_ind])
                for cognate_class in all_classes:
                    l_inds = np.where(self.cognacy_classes[:, concept_ind] == cognate_class)[0]
                    for l in l_inds:
                        out.write(self.languages[l] + ",")
                    out.write("\t")
                out.write("\n")



    @classmethod
    def create_data(cls, data_file, header, sound_model, ldn=0.5, data_diag=True,
                    check_consistency=True, pre_tree=None):
        # type: (str, list, str, float, bool, bool, str) -> DataClass
        """
        This function creates a DataClass instance from the cldf format.

        :param data_file: location of the data file
        :type data_file: str
        :param header: column names of the interesting columns
        :type header: list
        :param sound_model: asjp or ipa sound model
        :type sound_model: str
        :param ldn: cutoff of ldn scores. Each word pair with an ldn score lower than this is considered.
        :type ldn: float
        :param data_diag: should the data be stored in diagonal format
        :type data_diag: bool
        :param check_consistency: indicate if the consistency of the tree topology with a certain constrains is checked
        :type check_consistency: bool
        :param pre_tree: tree from specified starting state
        :type pre_tree: str
        :return: A DataClass instance
        :rtype: DataClass
        """
        # read data file into pandas dataframe
        nlex = pd.read_csv(data_file, sep="\t", encoding="utf-8", dtype=object)
        exc = [i.decode("utf-8") for i in Alphas.diachritics_ipa] + [" ".decode("utf-8")]
        language_col, concept_col, _, cog_class_col = header
        header = header if cog_class_col is not None else header[:-1]

        concepts = np.unique(nlex[concept_col].values).tolist()
        languages = np.unique(nlex[language_col].values).tolist()


        # convert data frame into dictionary for class construction
        my_langs = {}
        len_conc = len(concepts)
        for ind, row in nlex[header].iterrows():
            lang = row[0]
            if lang in languages:
                if lang not in my_langs.keys():
                    # create LanguageData instance if necessary
                    lang_data = LanguageData(lang, [None] * len_conc)
                    lang_data.cognacy = [None] * len_conc
                    my_langs[lang] = lang_data
                conc_ind = concepts.index(row[1])
                try:
                    word = tuple([j for j in row[2] if j not in exc])

                except TypeError:
                    raise Exception(
                        "There is something wrong with the data. Check line {0} of your data file. {1} caused the error".format(
                            str(ind), str(row)))
                my_langs[lang].wordlist[conc_ind] = word
                if cog_class_col is not None:
                    my_langs[lang].cognacy[conc_ind] = row[3]

        return cls(language_data=my_langs, concepts=concepts, sound_model=sound_model, ldn_cut=ldn, data_diag=data_diag,
                   check_consistency=check_consistency, cc_class=False, pre_tree=pre_tree)

    @staticmethod
    def ldn(a, b):
        # type: (str, str) -> float
        """
        Levensthein distance normalized

        :param a: word
        :type a: str
        :param b: word
        :type b: str
        :return: distance score
        :rtype: float
        """
        m = []
        la = len(a) + 1
        lb = len(b) + 1
        for i in range(0, la):
            m.append([])
            for j in range(0, lb):
                m[i].append(0)
            m[i][0] = i
        for i in range(0, lb):
            m[0][i] = i
        for i in range(1, la):
            for j in range(1, lb):
                s = m[i - 1][j - 1] if a[i - 1] == b[j - 1] else m[i - 1][j - 1] + 1
                m[i][j] = min(m[i][j - 1] + 1, m[i - 1][j] + 1, s)
        la -= 1
        lb -= 1
        return float(m[la][lb]) / float(max(la, lb))

    @staticmethod
    def families(data, fams):
        # type: (str, list|set) -> list
        """
        Extract a subset of the languages in the data based on families specified in fams

        :param data: location of the data file
        :param fams: families which need to be considered
        :return: list of languages which are considered
        """
        with open(data, "r") as infile:
            cont = infile.readlines()
        ret_list = []
        while cont:
            cur = cont.pop(0).strip().replace(",", ".").split(".")
            if cur[2] in fams:
                ret_list.append(cur[0])
        return ret_list

    def binarise_bottom_up(self):
        """
        Still here for legacy reasons. May be deleted later

        :return:
        """
        n_classes = self.tree.root.cluster_store[:, 36, 0, 0]
        m_l = np.sum(n_classes)
        ind_l = [int(sum(n_classes[:i])) for i in range(self.n_concepts)]
        l_set = set([])
        for concept_ind, concept in enumerate(self.tree.root.cluster_store):
            for cluster in range(int(concept[36][0][0])):
                for l in range(int(concept[cluster][0][36])):
                    lind = int(concept[cluster][0][l])
                    node = self.tree._name_node_map[self.languages[lind]]
                    if lind not in l_set:
                        node.state_probs = np.zeros((int(m_l), 2), dtype=np.float64)
                        node.state_probs[:, 1] = 1
                        l_set.add(lind)
                    ind = ind_l[concept_ind] + cluster

                    node.state_probs[ind] = [1, 0]

    def binarise_cognate_judgements(self):
        """
        Still here for legacy reasons. May be deleted later

        :return:
        """

        # number of classes per concept slot
        n_classes = [len(set(self.cognacy_classes[:, i])) for i in range(self.n_concepts)]

        m_l = np.sum(n_classes)

        for node in self.tree.terminal_nodes:
            node.state_probs = np.zeros((m_l, 2), dtype=np.float64)
            node.state_probs[:, 1] = 1
            l_cl = self.cognacy_classes[self.lang_ind_dict[node.name]]
            trailing = 0
            for ind, c in enumerate(l_cl):

                self[node.name][ind][-2] = c
                node.state_probs[trailing + c][0] = 1
                node.state_probs[trailing + c][1] = 0

                trailing += n_classes[ind]
        self.pre_calc_single()


class LanguageData(object):
    __slots__ = ("name", "wordlist", "cognacy", "translated")

    def __init__(self, name, wordlist):
        # type: (str, list) -> None
        """
        This class stores the basic data for one language

        :param name: an identifier for the language
        :param wordlist: the list of words for this langauge
        """

        self.name = name
        self.wordlist = wordlist
        self.cognacy = {}
        self.translated = {}

    def __getitem__(self, item):
        return self.translated[item]

    def get_cognacy(self, item):
        # type: (int) -> tuple
        """
        Get a word and its cognate class by index

        :param item: index of an item in the word list
        :type item: int
        :return: the actual word and its cognate class
        :rtype: tuple
        """
        return self.wordlist[item], self.cognacy[item]

    def translate(self, alphabet):
        # type: (list[str]) -> None
        """
        Translate the words into a list of alphabet indices

        :param alphabet: the alphabet in a list format
        :type alphabet: list
        """
        translated = np.empty((len(self.wordlist), vec_size), dtype=np.int8)
        translated.fill(-1)
        for idx, word in enumerate(self.wordlist):
            if word:
                for jdx, let in enumerate(word):
                    translated[idx][jdx] = alphabet.index(let)
                translated[idx][-1] = len(word)

        self.translated = translated
