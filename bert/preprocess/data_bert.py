"""
Preprocessor and dataset definition for NLI.
"""
# Aurelien Coet, 2019.

import string
import torch
import numpy as np
import random
from collections import Counter
from nltk.corpus import stopwords
from tqdm import tqdm
from torch.utils.data import Dataset


class Preprocessor(object):
    """
    Preprocessor class for Natural Language Inference datasets.

    The class can be used to read NLI datasets, build worddicts for them
    and transform their premises, hypotheses and labels into lists of
    integer indices.
    """

    def __init__(self,
                 labeldict,
                 encoder,
                 lowercase=False,
                 ignore_punctuation=False):
        self.labeldict = labeldict
        self.encoder = encoder
        self.lowercase = lowercase
        self.ignore_punctuation = ignore_punctuation

    def read_data(self, filepath, test=False, aug_rate=0.5, aug_drop_p=0.5):
        with open(filepath, "r", encoding="utf8") as input_data:
            ids, premises, hypotheses, labels = [], [], [], []
            parentheses_table = str.maketrans({"(": None, ")": None})
            punct_table = str.maketrans({key: " " for key in string.punctuation})
            next(input_data)
            for line in input_data:
                line = line.strip().split("\t")
                if line[0] == "-":
                    continue
                pair_id = line[7]
                premise = line[1]
                hypothesis = line[2]
                premise = premise.translate(parentheses_table)
                hypothesis = hypothesis.translate(parentheses_table)
                if self.ignore_punctuation:
                    premise = premise.translate(punct_table)
                    hypothesis = hypothesis.translate(punct_table)
                if test == False:
                    premise, hypothesis = self.aug_function(premise, hypothesis, aug_rate, aug_drop_p)
                if type(premise) == str:
                    premise = [premise]
                    hypothesis = [hypothesis]
                for idx in range(len(premise)):
                    p = premise[idx]
                    h = hypothesis[idx]
                    premises_word_list = [w for w in p.rstrip().split()]
                    hypothese_word_list = [w for w in h.rstrip().split()]
                    p_text = self.encoder.tokenize(premises_word_list, get_subword_indices=False)
                    h_text = self.encoder.tokenize(hypothese_word_list, get_subword_indices=False)
                    labels.append(line[0])
                    ids.append(pair_id)
                    premises.append(p_text)
                    hypotheses.append(h_text)
            return {"ids": ids,
                    "premises": premises,
                    "hypotheses": hypotheses,
                    "labels": labels}

    def aug_function(self, premise,hypothesis, aug_rate, aug_drop_p = 0.5):
        is_aug = random.random()
        if is_aug < aug_rate:
            # aug
            rate = random.random()
            if rate > 0.5:
                auged_premise = self.shuffle(premise)
                auged_hypothesis = self.shuffle(hypothesis)
            else:
                auged_premise = self.dropout(premise,aug_drop_p)
                auged_hypothesis = self.dropout(hypothesis,aug_drop_p)
            return [premise, auged_premise], [hypothesis, auged_hypothesis]
        else:
            return premise, hypothesis

    def shuffle(self, text):
        text = np.random.permutation(text.strip().split())
        return ' '.join(text)

    def dropout(self, text, p=0.5):
        text = text.strip().split()
        len_ = len(text)
        indexs = np.random.choice(len_, int(len_ * p))
        for i in indexs:
            text[i] = ''
        return ' '.join(text)


    def prepare_data(self, data):
        prepared_data = {"ids": [],
                         "premises": [],
                         "hypotheses": [],
                         "labels": []}

        tqdm_iterator = tqdm(data["premises"], desc="** Preprocessing data: ")
        for i, premise in enumerate(tqdm_iterator):
            label = data["labels"][i]
            if label not in self.labeldict and label != "hidden":
                continue
            prepared_data["ids"].append(data["ids"][i])
            if label == "hidden":
                prepared_data["labels"].append(-1)
            else:
                prepared_data["labels"].append(self.labeldict[label])
            prepared_data["premises"].append(premise)
            prepared_data["hypotheses"].append(data["hypotheses"][i])
        return prepared_data

class NLIDataset(Dataset):
    def __init__(self,
                 data,
                 encoder,
                 max_premise_length=None,
                 max_hypothesis_length=None):
        self.padding_idx=encoder.tokenizer.pad_token_id
        self.premises_lengths = [len(seq) for seq in data["premises"]]
        self.max_premise_length = max_premise_length
        if self.max_premise_length is None:
            self.max_premise_length = max(self.premises_lengths)

        self.hypotheses_lengths = [len(seq) for seq in data["hypotheses"]]
        self.max_hypothesis_length = max_hypothesis_length
        if self.max_hypothesis_length is None:
            self.max_hypothesis_length = max(self.hypotheses_lengths)

        self.num_sequences = len(data["premises"])

        self.data = {"ids": [],
                     "premises": torch.ones((self.num_sequences,
                                             self.max_premise_length),
                                            dtype=torch.long) * self.padding_idx,
                     "hypotheses": torch.ones((self.num_sequences,
                                               self.max_hypothesis_length),
                                              dtype=torch.long) * self.padding_idx,
                     "labels": torch.tensor(data["labels"], dtype=torch.long)}

        for i, premise in enumerate(data["premises"]):
            self.data["ids"].append(data["ids"][i])
            end = min(len(premise), self.max_premise_length)
            self.data["premises"][i][:end] = torch.tensor(premise[:end])

            hypothesis = data["hypotheses"][i]
            end = min(len(hypothesis), self.max_hypothesis_length)
            self.data["hypotheses"][i][:end] = torch.tensor(hypothesis[:end])

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, index):
        return {"id": self.data["ids"][index],
                "premise": self.data["premises"][index],
                "premise_length": min(self.premises_lengths[index],
                                      self.max_premise_length),
                "hypothesis": self.data["hypotheses"][index],
                "hypothesis_length": min(self.hypotheses_lengths[index],
                                         self.max_hypothesis_length),
                "label": self.data["labels"][index]}
