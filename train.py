import time
from tqdm import tqdm
import torch
from utils import correct_predictions
import torch.nn as nn
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
        description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}"\
                      .format(batch_time_avg/(batch_index+1),
                              running_loss/(batch_index+1))
        tqdm_batch_iterator.set_description(description)
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct_preds / num_labels
    return epoch_time, epoch_loss, epoch_accuracy