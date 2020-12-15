import numpy as np
import torch.nn as nn
import torch
import os
import pickle
import utils
import time
from termcolor import colored
from utils import rank_logger_info
from utils import report_result
from utils import check_pooling_parser,check_filename
from torch.utils.data import DataLoader
from drlstm.data import NLIDataset
from drlstm.model import DRLSTM
from drlstm.multi_model import multi_model as Multi_DRLSTM
from drlstm.multi_model import Ensemble_model1
from drlstm.multi_model import Ensemble_model2
from drlstm.multi_model import Ensemble_model3
from drlstm.multi_model import Ensemble_model4
from utils import correct_predictions
from parameters import create_parser
import random
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from multi_model_train import multi_model_train
from multi_model_valid import multi_model_valid
"""
Utility functions for training and validating models.
"""
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def main(args, logger):
    stat_file = args.test_statistics
    device = args.local_rank if args.local_rank != -1 else (
        torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
    load_path = args.load_path
    test_file = args.test_data
    embedding_file = args.embeddings
    batch_size = args.batch_size

    info = "\t* Loading testing data..."
    rank_logger_info(logger, args.local_rank, info)
    with open(test_file, "rb") as pkl:
        test_data = NLIDataset(pickle.load(pkl))
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
    with open(embedding_file, "rb") as pkl:
        embeddings = torch.tensor(pickle.load(pkl), dtype=torch.float).to(device)
    with open(stat_file, "rb") as pkl:
            test_statistics = pickle.load(pkl)

    info = "\t* Loading pretrained models..."
    rank_logger_info(logger, args.local_rank, info)
    model_path_lst = os.listdir(load_path)
    checkpoint_lst = [torch.load(os.path.join(load_path, pretrained_file), map_location=torch.device(device)) for
                      pretrained_file in model_path_lst]
    model_state_dict_lst = [checkpoint["model_state_dict"] for checkpoint in checkpoint_lst]
    best_score_lst = [checkpoint["best_score"] for checkpoint in checkpoint_lst]
    epochs_count = [checkpoint["epochs_count"] for checkpoint in checkpoint_lst]
    model_n = len(model_state_dict_lst)
    info = "\t* Loading done : {}".format(model_n)

    rank_logger_info(logger, args.local_rank, info)
    model_lst = [DRLSTM(embeddings.shape[0],
                        embeddings.shape[1],
                        hidden_size=args.hidden_size,
                        embeddings=embeddings,
                        padding_idx=0,
                        dropout=args.dropout,
                        num_classes=args.num_classes,
                        device=device,
                        pooling_method_lst=args.pooling_method,
                        embedding_dropout=args.embedding_dropout) for i in range(model_n)]
    for idx, model in enumerate(model_lst):
        model.load_state_dict(model_state_dict_lst[idx])
        model.to(device)
        for params in model.parameters():
            params.requires_grad = False

    if args.ensemble_mode == 1 or args.ensemble_mode == 2:
        info = "\t* training..."
        rank_logger_info(logger, args.local_rank, info)
        if args.ensemble_mode == 1:
            ensemble_model = Ensemble_model1(model_n)
        else:
            ensemble_model = Ensemble_model2(model_n)
        ensemble_model.to(device)
        with open(args.valid_data, "rb") as pkl:
            valid_data = NLIDataset(pickle.load(pkl))
        valid_loader = DataLoader(valid_data,
                                  shuffle=False,
                                  batch_size=args.batch_size)
        criterion = nn.CrossEntropyLoss()
        if args.optim == "adam":
            optimizer = torch.optim.Adam(ensemble_model.parameters(), lr=args.lr)
        elif args.optim == "rmsprop":
            optimizer = torch.optim.RMSprop(ensemble_model.parameters(), lr=args.lr)
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
            epoch_time, epoch_loss, epoch_accuracy = multi_model_train(args,
                                                           epoch,
                                                           ensemble_model,
                                                           model_lst,
                                                           valid_loader,
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
            weight_lst = ensemble_model.weight_layer.weight.data.cpu().numpy().tolist()[0]
            rank_logger_info(logger, args.local_rank, weight_lst)
            info = "* Validation for epoch {}:".format(epoch)
            rank_logger_info(logger, local_rank, info)
            epoch_time, epoch_loss, epoch_accuracy = multi_model_valid(ensemble_model,
                                                              model_lst,
                                                              test_loader,
                                                              criterion,
                                                              device)
            valid_losses.append(epoch_loss)
            valid_accuracy.append(epoch_accuracy)
            info = "Validing epoch: {}, time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%\n".format(epoch, epoch_time,
                                                                                                 epoch_loss,
                                                                                                 (epoch_accuracy * 100))
            rank_logger_info(logger, args.local_rank, info)
            scheduler.step(epoch_accuracy)
            if epoch_accuracy <= best_score:
                patience_counter += 1
            else:
                best_score = epoch_accuracy
                best_model = ensemble_model
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
            report_result(epochs_count, train_losses, valid_losses, train_accuracy, valid_accuracy, args.save_path)
    else: # 模式3 4
        info = "\t* testing..."
        rank_logger_info(logger, args.local_rank, info)
        if args.ensemble_mode == 3:
            ensemble_model = Ensemble_model3(model_n)
        else:
            ensemble_model = Ensemble_model4(model_n)
        ensemble_model.to(device)
        ensemble_model.eval()
        time_start = time.time()
        batch_time = 0.0
        accuracy = 0.0
        with torch.no_grad():
            for batch in test_loader:
                batch_start = time.time()
                premises = batch["premise"].to(device)
                premises_lengths = batch["premise_length"].to(device)
                hypotheses = batch["hypothesis"].to(device)
                hypotheses_lengths = batch["hypothesis_length"].to(device)
                labels = batch["label"].to(device)
                logits_probs_lst = [model(premises,
                             premises_lengths,
                             hypotheses,
                             hypotheses_lengths) for model in model_lst]
                logits_lst = [i[0].unsqueeze(1) for i in logits_probs_lst]
                probs_lst = [i[1].unsqueeze(1) for i in logits_probs_lst]
                _, probs = ensemble_model(logits_lst, probs_lst)
                accuracy += correct_predictions(probs, labels)
                batch_time += time.time() - batch_start
                _, out_classes = probs.max(dim=1)
            batch_time /= len(test_loader)
            total_time = time.time() - time_start
            accuracy /= (len(test_loader.dataset))
            info = "-> Average batch processing time: {:.4f}s, total test time:\
             {:.4f}s, accuracy: {:.4f}%".format(batch_time, total_time, (accuracy * 100))
            rank_logger_info(logger, args.local_rank, info)
if __name__ == '__main__':
    parser = create_parser()
    parser = check_pooling_parser(parser)
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