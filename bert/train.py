import numpy as np
import torch.nn as nn
import torch
import os
import pickle
import json
import utils
from termcolor import colored
import time
from tqdm import tqdm
from utils import rank_logger_info
from utils import report_result
from utils import check_pooling_parser, check_filename, check_data_path, correct_predictions

from torch.utils.data import DataLoader
from preprocess.data_bert import NLIDataset
from model import DRLSTM
from multi_model import multi_model as Multi_DRLSTM
from parameters import create_parser
import random
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from encoder import ENCODER

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def train(args,
          epoch,
          model,
          dataloader,
          optimizer,
          criterion,
          max_gradient_norm,
          device):
    model.train()
    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0
    correct_preds = 0
    tqdm_batch_iterator = tqdm(dataloader)
    num_labels = 0
    for batch_index, batch in enumerate(tqdm_batch_iterator):
        if args.local_rank != -1:
            dataloader.sampler.set_epoch(epoch)
        batch_start = time.time()
        premises = batch["premise"].to(device)
        premises_lengths = batch["premise_length"].to(device)
        hypotheses = batch["hypothesis"].to(device)
        hypotheses_lengths = batch["hypothesis_length"].to(device)
        labels = batch["label"].to(device)
        num_labels += len(labels)
        optimizer.zero_grad()
        logits, probs = model(premises,
                              premises_lengths,
                              hypotheses,
                              hypotheses_lengths)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        optimizer.step()
        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()
        correct_preds += correct_predictions(probs, labels)
        description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}" \
            .format(batch_time_avg / (batch_index + 1),
                    running_loss / (batch_index + 1))
        tqdm_batch_iterator.set_description(description)
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct_preds / num_labels
    return epoch_time, epoch_loss, epoch_accuracy


def validate(model, dataloader, criterion, device):
    model.eval()
    epoch_start = time.time()
    running_loss = 0.0
    running_accuracy = 0.0
    with torch.no_grad():
        for batch in dataloader:
            premises = batch["premise"].to(device)
            premises_lengths = batch["premise_length"].to(device)
            hypotheses = batch["hypothesis"].to(device)
            hypotheses_lengths = batch["hypothesis_length"].to(device)
            labels = batch["label"].to(device)
            logits, probs = model(premises,
                                  premises_lengths,
                                  hypotheses,
                                  hypotheses_lengths)
            loss = criterion(logits, labels)
            running_loss += loss.item()
            running_accuracy += correct_predictions(probs, labels)
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_accuracy / (len(dataloader.dataset))
    return epoch_time, epoch_loss, epoch_accuracy


def main(args, logger):
    device = args.local_rank if args.local_rank != -1 else (
        torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
    encoder = ENCODER(model=args.model, model_size=args.model_size, fine_tune=args.fine_tune, cased=True)
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
    info = 20 * "=" + " Preparing for training " + 20 * "="
    rank_logger_info(logger, args.local_rank, info)
    info = "\t* Loading training data..."
    rank_logger_info(logger, args.local_rank, info)
    with open(args.train_data, "rb") as pkl:
        train_data = NLIDataset(pickle.load(pkl),encoder)
    train_sampler = DistributedSampler(train_data) if args.local_rank != - 1 else None
    train_loader = DataLoader(train_data,
                              shuffle=True if not train_sampler else False,
                              sampler=train_sampler,
                              batch_size=args.batch_size)
    info = "\t* Loading validation data..."
    rank_logger_info(logger, args.local_rank, info)
    with open(args.valid_data, "rb") as pkl:
        valid_data = NLIDataset(pickle.load(pkl),encoder)
    valid_loader = DataLoader(valid_data,
                              shuffle=False,
                              batch_size=args.batch_size)

    # -------------------- Model definition ------------------- #
    info = "\t* Building model..."
    rank_logger_info(logger, args.local_rank, info)
    if args.multimodel:
        model = Multi_DRLSTM(embedding_layer = encoder,
                             hidden_size=args.hidden_size,
                             dropout=args.dropout,
                             num_classes=args.num_classes,
                             device=device,
                             pooling_method_lst=args.pooling_method)
    else:
        model = DRLSTM(embedding_layer = encoder,
                       hidden_size=args.hidden_size,
                       dropout=args.dropout,
                       num_classes=args.num_classes,
                       device=device,
                       pooling_method_lst=args.pooling_method)
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[
            args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    # -------------------- Preparation for training  ------------------- #
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode="max",
                                                           factor=0.5,
                                                           patience=0)
    best_score = 0.0
    start_epoch = 1

    # Data for loss curves plot.
    epochs_count = []
    train_losses = []
    valid_losses = []
    train_accuracy = []
    valid_accuracy = []
    info = "\n" + 20 * "=" + "Training model on device: {}".format(device) + 20 * "="
    rank_logger_info(logger, args.local_rank, info)

    patience_counter = 0

    for epoch in range(start_epoch, args.epochs + 1):
        epochs_count.append(epoch)
        info = "* Training epoch {}:".format(epoch)
        rank_logger_info(logger, local_rank, info)
        epoch_time, epoch_loss, epoch_accuracy = train(args,
                                                       epoch,
                                                       model,
                                                       train_loader,
                                                       optimizer,
                                                       criterion,
                                                       args.max_gradient_norm,
                                                       device)
        train_losses.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)
        info = "Training epoch: {}, time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%\n".format(epoch, epoch_time,
                                                                                             epoch_loss,
                                                                                             (epoch_accuracy * 100))
        rank_logger_info(logger, args.local_rank, info)
        # change: logger
        info = "* Validation for epoch {}:".format(epoch)
        rank_logger_info(logger, local_rank, info)
        epoch_time, epoch_loss, epoch_accuracy = validate(model,
                                                          valid_loader,
                                                          criterion,
                                                          device)
        valid_losses.append(epoch_loss)
        valid_accuracy.append(epoch_accuracy)
        info = "Validing epoch: {}, time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%\n".format(epoch, epoch_time,
                                                                                             epoch_loss,
                                                                                             (epoch_accuracy * 100))
        rank_logger_info(logger, args.local_rank, info)
        # Update the optimizer's learning rate with the scheduler.
        scheduler.step(epoch_accuracy)
        # Early stopping on validation accuracy.
        # if args.local_rank in [-1, 0]:
        if epoch_accuracy <= best_score:
            patience_counter += 1
        else:
            best_score = epoch_accuracy
            patience_counter = 0
            if args.local_rank in [-1, 0]:
                torch.save({"epoch": epoch,
                            "model": model.state_dict(),
                            "best_score": best_score,
                            "epochs_count": epochs_count,
                            "train_losses": train_losses,
                            "valid_losses": valid_losses},
                           os.path.join(args.save_path, "best.pth.tar"))
        if patience_counter >= args.patience:
            info = "-> Early stopping: patience limit reached, stopping..."
            rank_logger_info(logger, args.local_rank, info)
            break
    if args.local_rank in [-1, 0]:
        report_result(epochs_count, train_losses, valid_losses, train_accuracy, valid_accuracy, args.save_path)


if __name__ == '__main__':
    parser = create_parser()
    parser = check_pooling_parser(parser)
    parser = check_filename(parser)
    parser = check_data_path(parser)
    set_seed(parser.seed)
    local_rank = parser.local_rank
    if local_rank != -1:
        dist_backend = 'nccl'
        dist.init_process_group(backend=dist_backend)
    device = local_rank if local_rank != -1 else (
        torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
    torch.cuda.set_device(local_rank)
    logger = utils.setup_logger(__name__, os.path.join(parser.checkpoint_path, 'train.log'))
    rank_logger_info(logger, parser.local_rank, colored(parser, "red"))
    main(parser, logger)