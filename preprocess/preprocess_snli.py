"""
Preprocess the SNLI dataset and word embeddings to be used by the LEAN model.
"""
# Aurelien Coet, 2018.

import os
import pickle
import argparse
import fnmatch
import json
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))
from lean.data import Preprocessor
import random
import numpy as np
import torch
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
def preprocess_SNLI_data(inputdir,
                         embeddings_file,
                         w2h_file,
                         lear_file,
                         targetdir,
                         lowercase=False,
                         ignore_punctuation=False,
                         num_words=None,
                         labeldict={},
                         bos=None,
                         eos=None,
                         aug_rate=0.5,
                         aug_drop_p=0.3):
    if not os.path.exists(targetdir):
        os.makedirs(targetdir)
    train_file = ""
    dev_file = ""
    test_file = ""
    for file in os.listdir(inputdir):
        if fnmatch.fnmatch(file, "*_train.txt"):
            train_file = file
        elif fnmatch.fnmatch(file, "*_dev.txt"):
            dev_file = file
        elif fnmatch.fnmatch(file, "*_test.txt"):
            test_file = file
    # -------------------- Train data preprocessing -------------------- #
    preprocessor = Preprocessor(lowercase=lowercase,
                                ignore_punctuation=ignore_punctuation,
                                num_words=num_words,
                                labeldict=labeldict,
                                bos=bos,
                                eos=eos,
                                aug_rate=aug_rate,
                                aug_drop_p=aug_drop_p)
    print(20*"=", " Preprocessing train set ", 20*"=")
    print("* Reading data...")
    data = preprocessor.read_data(os.path.join(inputdir, train_file), train=True)

    print("* Computing worddict and saving it...")
    preprocessor.build_worddict(data)
    with open(os.path.join(targetdir, "worddict.pkl"), "wb") as pkl_file:
        pickle.dump(preprocessor.worddict, pkl_file)

    print("* Preparing the data...")
    prepared_data = preprocessor.prepare_data(data)

    print("* Saving result...")
    with open(os.path.join(targetdir, "train_data.pkl"), "wb") as pkl_file:
        pickle.dump(prepared_data, pkl_file)

    # -------------------- Validation data preprocessing -------------------- #
    print(20*"=", " Preprocessing dev set ", 20*"=")
    print("* Reading data...")
    data = preprocessor.read_data(os.path.join(inputdir, dev_file))

    print("* Preparing the data...")
    prepared_data = preprocessor.prepare_data(data)

    print("* Saving result...")
    with open(os.path.join(targetdir, "dev_data.pkl"), "wb") as pkl_file:
        pickle.dump(prepared_data, pkl_file)

    # -------------------- Test data preprocessing -------------------- #
    print(20*"=", " Preprocessing test set ", 20*"=")
    print("* Reading data...")
    data = preprocessor.read_data(os.path.join(inputdir, test_file))

    print("* Preparing the data...")
    prepared_data = preprocessor.prepare_data(data)

    print("* Saving result...")
    with open(os.path.join(targetdir, "test_data.pkl"), "wb") as pkl_file:
        pickle.dump(prepared_data, pkl_file)

    # -------------------- Embeddings preprocessing -------------------- #
    print(20*"=", " Preprocessing embeddings ", 20*"=")
    print("* Building embedding matrix and saving it...")
    embed_matrix = preprocessor.build_embedding_matrix(embeddings_file)
    with open(os.path.join(targetdir, "embeddings.pkl"), "wb") as pkl_file:
        pickle.dump(embed_matrix, pkl_file)
    
    print("* Building Word2Hyp embedding matrix and saving it...")
    embed_matrix = preprocessor.build_embedding_matrix(w2h_file)
    with open(os.path.join(targetdir, "w2h_embeddings.pkl"), "wb") as pkl_file:
        pickle.dump(embed_matrix, pkl_file)

    print("* Building LEAR embedding matrix and saving it...")
    embed_matrix = preprocessor.build_embedding_matrix(lear_file)
    with open(os.path.join(targetdir, "lear_embeddings.pkl"), "wb") as pkl_file:
        pickle.dump(embed_matrix, pkl_file)


if __name__ == "__main__":
    set_seed(1)
    default_config = "../../config/preprocessing/snli_preprocessing.json"
    parser = argparse.ArgumentParser(description="Preprocess the SNLI dataset")
    parser.add_argument(
        "--config",
        default=default_config,
        help="Path to a configuration file for preprocessing SNLI"
    )
    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.realpath(__file__))
    if args.config == default_config:
        config_path = os.path.join(script_dir, args.config)
    else:
        config_path = args.config
    with open(os.path.normpath(config_path), "r") as cfg_file:
        config = json.load(cfg_file)
    preprocess_SNLI_data(
        os.path.normpath(os.path.join(script_dir, config["data_dir"])),
        os.path.normpath(os.path.join(script_dir, config["embeddings_file"])),
        os.path.normpath(os.path.join(script_dir, config["w2h_file"])),
        os.path.normpath(os.path.join(script_dir, config["lear_file"])),
        os.path.normpath(os.path.join(script_dir, config["target_dir"])),
        lowercase=config["lowercase"],
        ignore_punctuation=config["ignore_punctuation"],
        num_words=config["num_words"],
        labeldict=config["labeldict"],
        bos=config["bos"],
        eos=config["eos"],
        aug_rate=config["aug_rate"],
        aug_drop_p=config["aug_drop_p"]
    )
