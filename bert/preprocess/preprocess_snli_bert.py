"""
Preprocess the SNLI dataset and word embeddings to be used by the LEAN model.
"""
# Aurelien Coet, 2018.
import os
import pickle
import argparse
import fnmatch
import json
import os
import sys
from snli_preprocessing_bert_parameters import create_parser
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))
from data_bert import Preprocessor
from encoder import ENCODER

def preprocess_SNLI_data(inputdir,
                         targetdir,
                         encoder,
                         labeldict,
                         aug_rate = 0,
                         lowercase=False,
                         ignore_punctuation=False,):
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
    preprocessor = Preprocessor(labeldict = labeldict,
                                encoder = encoder,
                                lowercase=lowercase,
                                ignore_punctuation=ignore_punctuation)
    data = preprocessor.read_data(os.path.join(inputdir, train_file),aug_rate=aug_rate)

    print("* Preparing the data...")
    prepared_data= preprocessor.prepare_data(data)

    print("* Saving result...")
    with open(os.path.join(targetdir, "train_data.pkl"), "wb") as pkl_file:
        pickle.dump(prepared_data, pkl_file)

    # -------------------- Validation data preprocessing -------------------- #
    print(20 * "=", " Preprocessing dev set ", 20 * "=")
    print("* Reading data...")
    data = preprocessor.read_data(os.path.join(inputdir, dev_file),test=True)

    print("* Preparing the data...")
    prepared_data = preprocessor.prepare_data(data)

    print("* Saving result...")
    with open(os.path.join(targetdir, "dev_data.pkl"), "wb") as pkl_file:
        pickle.dump(prepared_data, pkl_file)

    # -------------------- Test data preprocessing -------------------- #
    print(20 * "=", " Preprocessing test set ", 20 * "=")
    print("* Reading data...")
    data = preprocessor.read_data(os.path.join(inputdir, test_file), test=True)

    print("* Preparing the data...")
    prepared_data = preprocessor.prepare_data(data)
    print("* Saving result...")
    with open(os.path.join(targetdir, "test_data.pkl"), "wb") as pkl_file:
        pickle.dump(prepared_data, pkl_file)

if __name__ == "__main__":
    args = create_parser()
    encoder = ENCODER(model=args.model, model_size=args.model_size, fine_tune=args.fine_tune, cased=True)
    inputdir = "/home/ywzhang/xsy/data/snli_1.0"
    targetdir = os.path.join("/home/ywzhang/xsy/data/snli_1.0",args.model,args.model_size)
    preprocess_SNLI_data(
        inputdir,
        targetdir,
        encoder = encoder,
        aug_rate = args.aug_rate,
        labeldict=args.labeldict,
        lowercase=args.lowercase,
        ignore_punctuation=args.ignore_punctuation)
