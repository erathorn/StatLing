"""
@author: erathorn
@date: July 2019
@version: 1.0
"""

import argparse

"""
keeps track of the allowed Tags for the settings file
"""
tagset_float = ("[ldn]",)
tagset_int = ("[Seed]", "[Iterations]", "[Thinning]", "[Window size]", "[randomize steps]")
tagset_str = ("[Algorithm]", "[Data]", "[Output]", "[concept_col]", "[lang_col]",
              "[transcription_col]", "[cognate_class_col]", "[Sound Model]", "[cognates]", "[Transition_Model]",
              "[folder]")

tagset_bool = ("[mpi]", "[overwrite_data_names]", "[sample_cognates]", "[MC3]", "[consistency_checker]",
               "[bottom-up]", "[randomize]")
allowed = {"[Model]": ("Sound", "Feature"), "[algorithm]": ("vit", "fw", "vit_double"), "[cognates]": ("pair", "mult"),
           "[Sound Model]": ("ipa", "asjp"), "[Transition_Model]": ("KM03", "TKF92", "TKF91", "PA")}


def read_settings_file(filename):
    # type: (str) -> dict
    """
    This function reads the settings file and turns it into a dictionary with the necessary data types

    :param filename: filename of the file with the settings for the MCMC run
    :return: Dictionary containing the information from the settings file
    """

    with open(filename, "r") as infile:
        # file is small so it can be read in one sweep
        cont = infile.readlines()

    # initialize dictionary, the standard column names are default
    parameters = {"overwrite_data_names": False, "sample_cognates": False, "ldn": 0.5, "consistency_checker": True,
                  "randomize": False, "folder": None}

    while cont:
        line = cont.pop(0)
        if line.startswith("["):
            # current line is the tag
            tag = line.strip()
            # next line is the value for the given tag
            val = cont.pop(0).strip()

            # data type conversion
            if tag in tagset_bool:
                val = str2bool(val)
            elif tag in tagset_int:
                val = int(val)
            elif tag in tagset_float:
                val = float(val)
            elif tag in tagset_str:
                val = str(val)
                if val.lower() == "none":
                    val = None
            else:
                # the tag is not recognized
                raise Exception("This is not a valid parameter: " + tag)

            if tag in allowed.keys() and val not in allowed[tag]:
                # can the value be interpreted for the tag
                raise Exception(
                    "This is not a valid parameter identifier combination: {0}, {1}".format(tag, str(val)))
            parameters[tag.strip("[").strip("]")] = val
    return parameters


def str2bool(v):
    # type: (str) -> bool
    """
    turn a string into a boolean

    :param v: string which will be translated into a boolean
    :return: boolean representation of v
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
