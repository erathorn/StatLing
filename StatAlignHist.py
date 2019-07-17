import argparse
import datetime
import os
import random
import sys

import numpy as np
from mpi4py import MPI

import src.Analysis.Analysis as srcAn
import src.C_Extensions.sample as c_ext_sam
import src.MCMC as MCMC
import src.MCMC.MCMC_MC3 as MCMC_3
import src.Utils.Data_class as srcUDC
import src.Utils.Read_Settings as srcRead


reload(sys)
sys.setdefaultencoding('utf8')


def write_settings_file(p, time):
    # type: (str, datetime.datetime) -> None
    """
    This function writes down the information for the current run such as to identify it

    :param p: file name
    :type p: str
    :param time: call time
    :type time: datetime
    :rtype: None
    """
    with open(p, "w") as outfile:
        outfile.write("Settings for MCMC estimation\n")
        outfile.write("Calltime:\n")
        outfile.write(time.strftime("%Y-%m-%d %H:%M") + "\n")
        outfile.write("-------Settings-------\n")
        for entry in vars(args):
            outfile.write("{0}\t{1}\n".format(str(entry), str(getattr(args, entry))))


def data_creation(data_file, sound_model, ldn, header, cc_sample, diag, check_consistency, pre_tree):
    # type: (str, str, float, list, bool, bool, bool, str) -> srcUDC.DataClass
    """
    Processes the data stored in data_file for the MCMC computation
    cc_sample has precedence over ldn. is only taken into account if cc_sample is false

    :param cc_sample: if cognate classes should be sampled or not
    :type cc_sample: bool
    :param data_file: the location of the data file
    :type data_file: str
    :param sound_model: asjp or ipa encoding
    :type sound_model: str
    :param ldn: Levenshtein distance for word pairs, all wordpairs below the threshold are not considered
    :type ldn: float
    :param header: specification of the relevant columns in the data file
    :type header: list
    :param diag:
    :type diag: bool
    :param check_consistency: check if consistencies with tree constraints should be checked
    :type check_consistency: bool
    :param pre_tree: specified starting point tree
    :type pre_tree: str
    :return: Class object holding the data for the computation
    :rtype: srcUDC.DataClass
    """

    return srcUDC.DataClass.create_data(data_file=data_file, header=header,
                                        sound_model=sound_model, ldn=ldn, cc_sam=cc_sample, data_diag=diag,
                                        check_consistency=check_consistency, pre_tree=pre_tree)


def set_random_seed(seed_val):
    # type: (int) -> None
    """
    This function sets the random seed for the computation

    :param seed_val: random seed to use
    :type seed_val: int
    :rtype: None
    """

    if seed_val is not None:
        try:
            np.random.seed(seed_val)
            random.seed(seed_val)
            c_ext_sam.set_srand_seed(seed_val)
        except ValueError:
            s = 1234
            np.random.seed(s)
            random.seed(s)
            c_ext_sam.set_srand_seed(s)
            print("could not transform given seed into integer. Using default seed: 1234")
    else:
        s = random.randint(0, 9999)
        c_ext_sam.set_srand_seed(s)


def MCMC_setup():
    """
    Perform setup for the MCMC from here on
    """
    # set seed
    set_random_seed(parameter_dict["Seed"])
    # create output folder if not existent
    if not os.path.isdir(parameter_dict["Output"]):
        os.mkdir(parameter_dict["Output"])
    # check if data file exists
    if not os.path.exists(parameter_dict["Data"]):
        raise Exception("Data file not found " + parameter_dict["Data"])
    # write information about MCMC run
    write_settings_file(parameter_dict["Output"] + "settings.log", time=now)

    # if the header in the data file is not the default one, use these names for the respective columns
    header = [parameter_dict["lang_col"],
              parameter_dict["concept_col"],
              parameter_dict["transcription_col"],
              parameter_dict["cognate_class_col"]]
    # process the data
    if parameter_dict["folder"] is not None:
        pre_tree, sm_d, tr_d = read_state_from_file(parameter_dict["folder"])
    else:
        pre_tree = None
        sm_d = None
        tr_d = None

    # process the data
    data = data_creation(data_file=parameter_dict["Data"],
                         header=header,
                         sound_model=parameter_dict["Sound Model"],
                         cc_sample=parameter_dict["sample_cognates"],
                         ldn=parameter_dict["ldn"],
                         diag=parameter_dict["bottom-up"],
                         check_consistency=parameter_dict["consistency_checker"],
                         pre_tree=pre_tree)
    MCMC_mod = MCMC.MCMC.create_mcmc(data=data, parameters=parameter_dict, tr_params=tr_d, em_params=sm_d)
    # set up the MCMC

    return MCMC_mod


def swap_store_gen(swap_store):
    max_index = len(swap_store)
    index = 0
    while index < max_index:
        yield swap_store[index]
        index += 1


def MCMC_MC3_setup(temperature, mpi_size, now):
    """
    Perform setup for the MCMC from here on
    """
    # set seed
    set_random_seed(parameter_dict["Seed"] + int(temperature))
    samples = parameter_dict["Iterations"] / 5
    if temperature == 1:
        # create output folder if not existent
        if not os.path.isdir(parameter_dict["Output"]):
            os.mkdir(parameter_dict["Output"])
        # check if data file exists
        if not os.path.exists(parameter_dict["Data"]):
            raise Exception("Data file not found " + parameter_dict["Data"])
        # write information about MCMC run
        write_settings_file(parameter_dict["Output"] + "settings.log", time=now)

        # create swap list
        donor_recip = np.random.randint(mpi_size, size=(samples, 2))
        swap_store = np.zeros((samples, 3), dtype=np.int)
        swap_store[::, 0:2] = donor_recip
        swap_store[::, 2] = np.sort(np.random.choice(parameter_dict["Iterations"], samples, replace=False))
    else:
        swap_store = np.empty((samples, 3), dtype=np.int)
    comm.Barrier()
    swap_store = comm.bcast(swap_store, root=0)

    # if the header in the data file is not the default one, use these names for the respective columns
    header = [parameter_dict["lang_col"],
              parameter_dict["concept_col"],
              parameter_dict["transcription_col"],
              parameter_dict["cognate_class_col"]]
    if parameter_dict["folder"] is not None:
        pre_tree, sm_d, tr_d = read_state_from_file(parameter_dict["folder"])
        parameter_dict["randomize"] = False
    else:
        pre_tree = None
        sm_d = None
        tr_d = None
    # process the data
    data = data_creation(data_file=parameter_dict["Data"],
                         header=header,
                         sound_model=parameter_dict["Sound Model"],
                         cc_sample=parameter_dict["sample_cognates"],
                         ldn=parameter_dict["ldn"],
                         diag=parameter_dict["bottom-up"],
                         check_consistency=parameter_dict["consistency_checker"],
                         pre_tree=pre_tree)
    # set up the MCMC
    MCMC_mod = MCMC_3.MCMC_MC3.create_mcmc(data=data, parameters=parameter_dict, temperature=temperature,
                                           swap_store=swap_store, tr_params=tr_d, em_params=sm_d)
    return MCMC_mod


def read_state_from_file(folder):
    tr_filename = folder + "tr_mod.log"
    sound_mod_filename = folder + "sound_mod.log"
    sound_classes_filename = folder + "sound_mod.log_classes"
    tree_filename = folder + "MCMC_test.trees.log"

    with open(tree_filename, "r") as infile:
        ct = infile.readlines()[-1]

    tree = ct.strip().split()[-1]
    tr_mod = srcAn.Evaluator.read_file(tr_filename)
    tr_d = {k.split("_")[1]: v for k, v in zip(tr_mod.iloc[-1:].columns, tr_mod.iloc[-1:].values[0])}
    sound_mod = srcAn.Evaluator.read_file(sound_mod_filename).iloc[-1:]
    sc = srcAn.Evaluator.read_file(sound_classes_filename)
    sm_d = sound_mod_dict(sound_mod, sc, "asjp")

    return tree, sm_d, tr_d


def sound_mod_dict(sound_mod, sound_classes, dialect):
    header = sound_mod.columns.tolist()

    # header for evo class columns
    evo_class_value_header = [i for i in header if i.split("_")[0] == "clv"]

    # frequencies column names
    freq_header = [i for i in header if i.split("_")[0] == "freq"]

    # indices of the class value for the class
    class_indices = sound_classes.values[0].tolist()

    # get the names of the sounds
    names = [i.split("_")[1] for i in freq_header]
    names = [i.decode("utf-8") for i in names]

    dct = {"names": names,
           "freqs": np.array([sound_mod[i].values[0] for i in freq_header]),
           "evo_map": class_indices,
           "evo_vals": np.array([sound_mod[i].values[0] for i in evo_class_value_header]),
           "model": dialect}

    return dct


def calc_temp(my_rank, heat_scale=0.1):
    # type: (int, float) -> float
    """
    This function calculates the temperature of the heated chains.

    :param my_rank: rank of the MCMC in the mpi setting
    :type my_rank: float|int
    :param heat_scale: scale factor for the heat
    :type heat_scale: float
    :return: the temperature of the chain
    :rtype: float
    """

    return 1.0 / (1.0 + heat_scale * my_rank)


if __name__ == '__main__':

    # get time of start
    now = datetime.datetime.now()

    # get location of settings file from commandline parameter
    parser = argparse.ArgumentParser(description="Model Setup for Historical Linguistics Statistical Alignment")
    parser.add_argument("-s", dest="Settings", type=str, help="path to settings file", required=True)
    args = parser.parse_args()

    # get parameters from settings file
    parameter_dict = srcRead.read_settings_file(args.Settings)

    if parameter_dict["MC3"]:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        mpi_size = comm.Get_size()
        temperature = calc_temp(my_rank=rank)

        print("rank " + str(rank) + " starts model setup now.")
        MCMC_mod = MCMC_MC3_setup(temperature, mpi_size, now)
        print("rank " + str(rank) + " model setup done.")
        comm.Barrier()

        if parameter_dict["randomize"]:
            # randomize starting point if desired
            MCMC_mod.randomize_starting_point(parameter_dict["randomize steps"], window_size=parameter_dict["Window size"])
        MCMC_mod.MC3_estimate(parameter_dict["Iterations"], parameter_dict["Thinning"], wsize=parameter_dict["Window size"])
        comm.Barrier()
    else:

        MCMC_mod = MCMC_setup()
        wsize = parameter_dict["Window size"]
        # start the MCMC
        if parameter_dict["randomize"]:
            MCMC_mod.randomize_starting_point(parameter_dict["randomize steps"], window_size=wsize)
        MCMC_mod.estimate(parameter_dict["Iterations"], parameter_dict["Thinning"], wsize=wsize)

    print "we are done"
