import logging
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
def setup_logger(logger_name, filename, delete_old = False):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    stderr_handler = logging.StreamHandler(sys.stderr)
    file_handler   = logging.FileHandler(filename, mode='w') if delete_old else logging.FileHandler(filename)
    file_handler.setLevel(logging.DEBUG)
    stderr_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stderr_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)
    logger.addHandler(file_handler)
    return logger


def rank_logger_info(logger, rank, info):
    if rank in [-1, 0]:
        logger.info(info)
    else:
        pass

def report_result(epochs_count, train_losses, valid_losses, train_accuracy, valid_accuracy,save_path):
    plt.plot(epochs_count, train_losses, "-r")
    plt.plot(epochs_count, valid_losses, "-b")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(["Training loss", "Validation loss"])
    plt.title("Cross entropy loss")
    plt.savefig(os.path.join(save_path, "loss"))
    plt.figure()
    plt.plot(epochs_count, train_accuracy, "-r")
    plt.plot(epochs_count, valid_accuracy, "-b")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(["Training acc", "Validation acc"])
    plt.title("Accuracy")
    plt.savefig(os.path.join(save_path, "accuracy"))

    lst = np.array([epochs_count, train_losses, train_accuracy, valid_losses, valid_accuracy]).transpose(1, 0)
    name = ["epoches", "t_loss", "t_acc", "v_loss", "v_acc"]
    to_saved = pd.DataFrame(columns=name, data=lst)
    to_saved.to_csv(os.path.join(save_path, "fullresult.csv"))
# Continuing training from a checkpoint if one was given as argument.
# if checkpoint:
#     checkpoint = torch.load(checkpoint)
#     start_epoch = checkpoint["epoch"] + 1
#     best_score = checkpoint["best_score"]
#
#     print("\t* Training will continue on existing model from epoch {}..."
#           .format(start_epoch))
#
#     model.load_state_dict(checkpoint["model"])
#     optimizer.load_state_dict(checkpoint["optimizer"])
#     epochs_count = checkpoint["epochs_count"]
#     train_losses = checkpoint["train_losses"]
#     valid_losses = checkpoint["valid_losses"]
def correct_predictions(output_probabilities, targets):
    _, out_classes = output_probabilities.max(dim=1)
    correct = (out_classes == targets).sum()
    return correct.item()


def check_pooling_parser(parser):
    parser.pooling_filename = parser.pooling_method
    parser.pooling_method = parser.pooling_method.split("_")
    return parser

def check_filename(parser):
    if parser.local_rank in [-1, 0]:
        if not os.path.exists(parser.save_path):
            os.mkdir(parser.save_path)
    parser.save_path = os.path.join(parser.save_path,parser.model)
    if parser.local_rank in [-1, 0]:
        if not os.path.exists(parser.save_path):
            os.mkdir(parser.save_path)
    return parser


def check_data_path(parser):
    parser.train_data = os.path.join(parser.data_dir, parser.model, parser.model_size, parser.train_data)
    parser.test_data = os.path.join(parser.data_dir, parser.model, parser.model_size, parser.test_data)
    parser.valid_data = os.path.join(parser.data_dir, parser.model, parser.model_size, parser.valid_data)
    return parser
