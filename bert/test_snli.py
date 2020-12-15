import numpy as np
import torch.nn as nn
import torch
import os
import zipfile
import pickle
import fnmatch
import json
import string
from encoder import ENCODER
from collections import Counter
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils import correct_predictions
import random
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


from torch.utils.data import DataLoader
from preprocess.data_bert import NLIDataset
from model import DRLSTM

import time
from tqdm import tqdm

import matplotlib.pyplot as plt

"""
Test the model on some preprocessed dataset.
"""
# Aurelien Coet, 2018.

def test(model, dataloader):
    model.eval()
    device = model.device

    time_start = time.time()
    batch_time = 0.0
    accuracy = 0.0
    first = True

    batch_i = 0
    n_high_overlap = 0
    n_reg_overlap = 0
    n_low_overlap = 0
    n_long_sentence = 0
    n_reg_sentence = 0
    n_short_sentence = 0
    n_negation = 0
    n_quantifier = 0
    n_belief = 0
    n_total_high_overlap = 0
    n_total_reg_overlap = 0
    n_total_low_overlap = 0
    n_total_long_sentence = 0
    n_total_reg_sentence = 0
    n_total_short_sentence = 0
    n_total_negation = 0
    n_total_quantifier = 0
    n_total_belief = 0
    n_total_entailment = 0
    n_total_neutral = 0
    n_total_contradiction = 0
    n_correct_entailment = 0
    n_correct_neutral = 0
    n_correct_contradiction = 0

    stat_file = 'test_statistics.pkl'
    with open(stat_file, "rb") as pkl:
            test_statistics = pickle.load(pkl)
    set_idx_high_overlap = set(test_statistics["high_overlap"])
    set_idx_reg_overlap = set(test_statistics["reg_overlap"])
    set_idx_low_overlap = set(test_statistics["low_overlap"])
    set_idx_long_sentence = set(test_statistics["long_sentence"])
    set_idx_reg_sentence = set(test_statistics["reg_sentence"])
    set_idx_short_sentence = set(test_statistics["short_sentence"])
    set_idx_negation = set(test_statistics["negation"])
    set_idx_quantifier = set(test_statistics["quantifier"])
    set_idx_belief = set(test_statistics["belief"])

    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for batch in dataloader:
            batch_start = time.time()
            # Move input and output data to the GPU if one is used.
            premises = batch["premise"].to(device)
            premises_lengths = batch["premise_length"].to(device)
            hypotheses = batch["hypothesis"].to(device)
            hypotheses_lengths = batch["hypothesis_length"].to(device)
            labels = batch["label"].to(device)
            _, probs = model(premises,
                             premises_lengths,
                             hypotheses,
                             hypotheses_lengths)
            accuracy += correct_predictions(probs, labels)
            batch_time += time.time() - batch_start
            _, out_classes = probs.max(dim=1)
            if first:
                print ('Predictions for the first 5 sentences:')
                print ('Predictions:')
                print (out_classes[:5])
                print ('Labels:')
                print (labels[:5])
                print ('0 = entailment, 1 = neutral, 2 = contradiction')
                first = False
                # statistics
            for i in range(len(out_classes)):
                line_i = batch_i*32 + i

                if labels[i] == 0:
                    n_total_entailment += 1
                elif labels[i] == 1:
                    n_total_neutral += 1
                elif labels[i] == 2:
                    n_total_contradiction += 1

                if line_i in set_idx_high_overlap:
                    n_total_high_overlap += 1
                if line_i in set_idx_reg_overlap:
                    n_total_reg_overlap += 1
                if line_i in set_idx_low_overlap:
                    n_total_low_overlap += 1
                if line_i in set_idx_long_sentence:
                    n_total_long_sentence += 1
                if line_i in set_idx_reg_sentence:
                    n_total_reg_sentence += 1
                if line_i in set_idx_short_sentence:
                    n_total_short_sentence += 1
                if line_i in set_idx_negation:
                    n_total_negation += 1
                if line_i in set_idx_quantifier:
                    n_total_quantifier += 1
                if line_i in set_idx_belief:
                    n_total_belief += 1

                if out_classes[i] == labels[i]:
                    if labels[i] == 0:
                        n_correct_entailment += 1
                    elif labels[i] == 1:
                        n_correct_neutral += 1
                    elif labels[i] == 2:
                        n_correct_contradiction += 1


                    if line_i in set_idx_high_overlap:
                        n_high_overlap += 1
                    if line_i in set_idx_reg_overlap:
                        n_reg_overlap += 1
                    if line_i in set_idx_low_overlap:
                        n_low_overlap += 1
                    if line_i in set_idx_long_sentence:
                        n_long_sentence += 1
                    if line_i in set_idx_reg_sentence:
                        n_reg_sentence += 1
                    if line_i in set_idx_short_sentence:
                        n_short_sentence += 1
                    if line_i in set_idx_negation:
                        n_negation += 1
                    if line_i in set_idx_quantifier:
                        n_quantifier += 1
                    if line_i in set_idx_belief:
                        n_belief += 1

            batch_i += 1

    print ('Total Entailment:' + str(n_total_entailment))
    print ('Correct Entailment:' + str(n_correct_entailment))
    print ('Accuracy:' + str(float(n_correct_entailment) / n_total_entailment))
    print ('Total Neutral:' + str(n_total_neutral))
    print ('Correct Neutral:' + str(n_correct_neutral))
    print ('Accuracy:' + str(float(n_correct_neutral) / n_total_neutral))
    
    print ('Total Contradiction:' + str(n_total_contradiction))
    print ('Correct Contradiction:' + str(n_correct_contradiction))
    print ('Accuracy:' + str(float(n_correct_contradiction) / n_total_contradiction))
    print ('Total high overlap sentence:' + str(n_total_high_overlap))
    print ('Correct high overlap sentence:' + str(n_high_overlap))
    print ('Accuracy:' + str(float(n_high_overlap) / n_total_high_overlap))
    print ('Total regular overlap sentence:' + str(n_total_reg_overlap))
    print ('Correct regular overlap sentence:' + str(n_reg_overlap))
    print ('Accuracy:' + str(float(n_reg_overlap) / n_total_reg_overlap))
    print ('Total low overlap sentence:' + str(n_total_low_overlap))
    print ('Correct low overlap sentence:' + str(n_low_overlap))
    print ('Accuracy:' + str(float(n_low_overlap) / n_total_low_overlap))
    print ('Total long sentence:' + str(n_total_long_sentence))
    print ('Correct long sentence:' + str(n_long_sentence))
    print ('Accuracy:' + str(float(n_long_sentence) / n_total_long_sentence))
    print ('Total regular sentence:' + str(n_total_reg_sentence))
    print ('Correct regular sentence:' + str(n_reg_sentence))
    print ('Accuracy:' + str(float(n_reg_sentence) / n_total_reg_sentence))
    print ('Total short sentence:' + str(n_total_short_sentence))
    print ('Correct short sentence:' + str(n_short_sentence))
    print ('Accuracy:' + str(float(n_short_sentence) / n_total_short_sentence))
    print ('Total sentence with negation:' + str(n_total_negation))
    print ('Correct sentence with negation:' + str(n_negation))
    print ('Accuracy:' + str(float(n_negation) / n_total_negation))
    print ('Total sentence with quantifier:' + str(n_total_quantifier))
    print ('Correct sentence with quantifier:' + str(n_quantifier))
    print ('Accuracy:' + str(float(n_quantifier) / n_total_quantifier))
    print ('Total sentence with belief:' + str(n_total_belief))
    print ('Correct sentence with belief:' + str(n_belief))
    print ('Accuracy:' + str(float(n_belief) / n_total_belief))

    batch_time /= len(dataloader)
    total_time = time.time() - time_start
    accuracy /= (len(dataloader.dataset))

    return batch_time, total_time, accuracy


def main_test(test_file, pretrained_file, batch_size=32):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:",device)
    print(20 * "=", " Preparing for testing ", 20 * "=")
    print(pretrained_file)
    checkpoint = torch.load(pretrained_file, map_location=torch.device(device))
    epoch = checkpoint["epoch"]
    model_state_dict = checkpoint["model"]
    best_score = checkpoint["best_score"]
    train_losses = checkpoint["train_losses"]
    valid_losses = checkpoint["valid_losses"]
    print("epoch:",epoch)
    print("best_score:",best_score)
    print("train_losses:",train_losses)
    print("valid_losses:",valid_losses)
    print("\t* Loading test data...")
    encoder = ENCODER("xlnet", model_size="base", fine_tune=False, cased=True)
    with open(test_file, "rb") as pkl:
        test_data = NLIDataset(pickle.load(pkl),encoder)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
    print("\t* Building model...")
    model = DRLSTM(embedding_layer = encoder,
                   hidden_size=450,
                   dropout=0.4,
                   num_classes=3,
                   device=device,
                   pooling_method_lst=["max","avg","max","avg"])
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.to(device)
    print(20 * "=",
          " Testing model on device: {} ".format(device),
          20 * "=")
    batch_time, total_time, accuracy = test(model, test_loader)

    print("-> Average batch processing time: {:.4f}s, total test time:\
 {:.4f}s, accuracy: {:.4f}%".format(batch_time, total_time, (accuracy*100)))


if __name__ == '__main__':
    # change
    # test_data = 'test_data.pkl'
    set_seed(1)
    test_data = '/home/ywzhang/xsy/data/snli_1.0/xlnet/base/test_data.pkl'
    checkpoint = 'checkpoint/xlnet/best.pth.tar'
    batch_size = 32
    print ('First 5 test sentences:')
    sentences = ['This church choir sings to the masses as they sing joyous songs from the book at a church.    The church has cracks in the ceiling.',
                'This church choir sings to the masses as they sing joyous songs from the book at a church. The church is filled with song.',
                'This church choir sings to the masses as they sing joyous songs from the book at a church. A choir singing at a baseball game.',
                'A woman with a green headscarf, blue shirt and a very big grin.    The woman is young.',
                'A woman with a green headscarf, blue shirt and a very big grin.    The woman is very happy.']
    for sent in sentences:
        print (sent)
    main_test(test_data,
          checkpoint,
          batch_size)