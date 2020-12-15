import time
import torch
from utils import correct_predictions
def validate( model, dataloader, criterion,device):
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