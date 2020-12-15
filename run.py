import numpy as np
import torch.nn as nn
import torch
import os
import pickle
import utils
from termcolor import colored
from utils import rank_logger_info
from utils import report_result
from utils import check_pooling_parser,check_filename
from torch.utils.data import DataLoader
from drlstm.data import NLIDataset
from drlstm.model import DRLSTM
from drlstm.multi_model import multi_model as Multi_DRLSTM
from parameters import create_parser
import random
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from train import train
from validate import validate
from test import main_test
"""
Utility functions for training and validating models.
"""
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def main(args, logger):
    device = args.local_rank if args.local_rank != -1 else (
        torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
    info = 20 * "=" + " Preparing for training " + 20 * "="
    rank_logger_info(logger, args.local_rank, info)
    info = "\t* Loading training data..."
    rank_logger_info(logger, args.local_rank, info)
    with open(args.train_data, "rb") as pkl:
        train_data = NLIDataset(pickle.load(pkl))
    train_sampler = DistributedSampler(train_data) if args.local_rank != - 1 else None
    train_loader = DataLoader(train_data,
                              shuffle=True if not train_sampler else False,
                              sampler = train_sampler,
                              batch_size=args.batch_size)
    info = "\t* Loading validation data..."
    rank_logger_info(logger, args.local_rank, info)
    with open(args.valid_data, "rb") as pkl:
        valid_data = NLIDataset(pickle.load(pkl))
    valid_loader = DataLoader(valid_data,
                              shuffle=False,
                              batch_size=args.batch_size)
    # -------------------- Model definition ------------------- #
    info = "\t* Building model..."
    rank_logger_info(logger, args.local_rank, info)
    with open(args.embeddings, "rb") as pkl:
        embeddings = torch.tensor(pickle.load(pkl), dtype=torch.float).to(device)

    if args.multimodel:
        model = Multi_DRLSTM(embeddings.shape[0],
            embeddings.shape[1],
            hidden_size=args.hidden_size,
            embeddings=embeddings,
            padding_idx=0,
            dropout=args.dropout,
            num_classes=args.num_classes,
            device=device,
            pooling_method_lst=args.pooling_method)
    else:
        model = DRLSTM(embeddings.shape[0],
                     embeddings.shape[1],
                     hidden_size = args.hidden_size,
                     embeddings=embeddings,
                     padding_idx=0,
                     dropout=args.dropout,
                     num_classes=args.num_classes,
                     device=device,
                     pooling_method_lst = args.pooling_method,
                     embedding_dropout = args.embedding_dropout)
    model.to(device)
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info = "total_num:{} trainable_num:{}".format(total_num, trainable_num)
    rank_logger_info(logger, local_rank, info)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,find_unused_parameters=True)
    criterion = nn.CrossEntropyLoss()

    if args.optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode="max",
                                                           factor=0.5,
                                                           patience=0)

    if not args.load_from:
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
        for epoch in range(start_epoch, args.epochs+1):
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
            info = "* Validation for epoch {}:".format(epoch)
            rank_logger_info(logger, local_rank, info)
            epoch_time, epoch_loss, epoch_accuracy = validate(model,
                                                              valid_loader,
                                                              criterion,
                                                              device)
            valid_losses.append(epoch_loss)
            valid_accuracy.append(epoch_accuracy)
            info = "Validing epoch: {}, time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%\n".format(epoch, epoch_time, epoch_loss, (epoch_accuracy*100))
            rank_logger_info(logger, args.local_rank, info)
            scheduler.step(epoch_accuracy)
            if epoch_accuracy <= best_score:
                patience_counter += 1
            else:
                best_score = epoch_accuracy
                best_model = model
                patience_counter = 0
                if args.local_rank in [-1, 0]:
                    torch.save({"epoch": epoch,
                                "model_state_dict": best_model.state_dict(),
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
            report_result(epochs_count, train_losses, valid_losses, train_accuracy, valid_accuracy,args.save_path)
    info = "-> Test : Loadding model from {}".format(os.path.join(args.save_path, "best.pth.tar"))
    rank_logger_info(logger, args.local_rank, info)
    main_test(args,logger)


if __name__ == '__main__':
    parser = create_parser()
    parser = check_pooling_parser(parser)
    # parser = check_filename(parser)
    parser.data_dir = os.path.join("preprocessed_data", parser.data_dir)
    parser.train_data = os.path.join(parser.data_dir, parser.train_data)
    parser.valid_data = os.path.join(parser.data_dir, parser.valid_data)
    parser.test_data = os.path.join(parser.data_dir, parser.test_data)
    parser.embeddings = os.path.join(parser.data_dir, parser.embeddings)

    parser.save_path = os.path.join("result", parser.save_path)
    if not os.path.exists(parser.save_path):
        os.mkdir(parser.save_path)
    set_seed(parser.seed)
    local_rank = parser.local_rank
    if local_rank != -1:
        dist_backend = 'nccl'
        dist.init_process_group(backend=dist_backend)
    device = local_rank if local_rank != -1 else (torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
    print(local_rank)
    torch.cuda.set_device(local_rank)
    logger = utils.setup_logger(__name__, os.path.join("log", parser.checkpoint_path))
    rank_logger_info(logger, parser.local_rank, colored(parser,"red"))
    main(parser, logger)
