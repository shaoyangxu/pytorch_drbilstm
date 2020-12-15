import torch
import torch.nn as nn
def get_mask(sequences_batch, sequences_lengths):
    batch_size = sequences_batch.size()[0]
    max_length = torch.max(sequences_lengths)
    mask = torch.ones(batch_size, max_length, dtype=torch.float)
    mask[sequences_batch[:, :max_length] == 0] = 0.0
    return mask

def masked_softmax(tensor, mask):
    tensor_shape = tensor.size()
    reshaped_tensor = tensor.view(-1, tensor_shape[-1])
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(tensor).contiguous().float()
    reshaped_mask = mask.view(-1, mask.size()[-1])
    result = nn.functional.softmax(reshaped_tensor * reshaped_mask, dim=-1)
    result = result * reshaped_mask
    result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)
    return result.view(*tensor_shape)

def weighted_sum(tensor, weights, mask):
    # tensor
    # weights: batch_size, sen1, sen2    batch_size,1, sen1        batch_size,sen1, dim
    # tensor: batch_size, sen2, dim
    weighted_sum = weights.bmm(tensor)
    while mask.dim() < weighted_sum.dim():
        mask = mask.unsqueeze(1)
    mask = mask.transpose(-1, -2)
    mask = mask.expand_as(weighted_sum).contiguous().float()
    return weighted_sum * mask


def sort_by_seq_lens(batch, sequences_lengths, descending=True):
    sorted_seq_lens, sorting_index =\
        sequences_lengths.sort(0, descending=descending)
    sorted_batch = batch.index_select(0, sorting_index)
    idx_range = torch.arange(0, len(sequences_lengths)).cuda()
    _, reverse_mapping = sorting_index.sort(0, descending=False)
    restoration_index = idx_range.index_select(0, reverse_mapping)
    return sorted_batch, sorted_seq_lens, sorting_index, restoration_index


def check_pool_out_dim(pool_out_dim,hidden_size,methods):
    dim_lst = [method_to_dim(hidden_size, method) for method in methods]
    assert pool_out_dim == sum(dim_lst)


def method_to_dim(hidden_size,method):
    if method == "max":
        return hidden_size
    elif method == "attn":
        return hidden_size
    elif method == "avg":
        return hidden_size
    elif method == "diff":
        return hidden_size
    elif method == "diffsum":
        return 2*hidden_size
    elif method == "endpoint":
        return 2*hidden_size
    elif method == "coherent":
        return hidden_size // 2 + 1
    elif method == "coref":
        return 3*hidden_size