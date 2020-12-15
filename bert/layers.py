import torch
import torch.nn as nn
from model_utils import *
from span_rep import *
class BilstmEncoder(nn.Module):
    def __init__(self,
                 rnn_type,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 bias=True,
                 dropout=0.0,
                 bidirectional=False):
        assert issubclass(rnn_type, nn.RNNBase), \
            "rnn_type must be a class inheriting from torch.nn.RNNBase"

        super(BilstmEncoder, self).__init__()

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional

        self._encoder = rnn_type(input_size,
                                 hidden_size,
                                 num_layers=num_layers,
                                 bias=bias,
                                 batch_first=True,
                                 dropout=dropout,
                                 bidirectional=bidirectional)

    def forward(self, sequences_batch, sequences_lengths, init_state):
        sorted_batch, sorted_lengths, _, restoration_idx = \
            sort_by_seq_lens(sequences_batch, sequences_lengths)
        packed_batch = nn.utils.rnn.pack_padded_sequence(sorted_batch,
                                                         sorted_lengths,
                                                         batch_first=True)
        outputs, state = self._encoder(packed_batch, init_state)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs,
                                                      batch_first=True)
        hidden_state, cell_state = state[0], state[1]
        reordered_outputs = outputs.index_select(0, restoration_idx)
        reordered_hidden_state = hidden_state.index_select(1, restoration_idx) # because state is shaped like: num_layers * bidirectional, batch_size, hidden_size
        reordered_cell_state = cell_state.index_select(1, restoration_idx)
        return reordered_outputs, (reordered_hidden_state,reordered_cell_state)

class SoftmaxAttention(nn.Module):
    def forward(self,
                premise_batch,
                premise_mask,
                hypothesis_batch,
                hypothesis_mask):
        similarity_matrix = premise_batch.bmm(hypothesis_batch.transpose(2, 1)
                                              .contiguous())
        prem_using_hyp_attn = masked_softmax(similarity_matrix, hypothesis_mask)
        hyp_using_prem_attn = masked_softmax(similarity_matrix.transpose(1, 2).contiguous(),
                                                premise_mask)
        attended_premises = weighted_sum(hypothesis_batch,
                                         prem_using_hyp_attn,
                                         premise_mask)
        attended_hypotheses = weighted_sum(premise_batch,
                                           hyp_using_prem_attn,
                                           hypothesis_mask)
        return attended_premises, attended_hypotheses


class attention_pooling(nn.Module):
    
    def __init__(self, hidden_size):
        super(attention_pooling, self).__init__()

        self.word_weight_layer = nn.Linear(hidden_size, 1)

    def forward(self, tensor, mask):
        weight = self.word_weight_layer(tensor).squeeze(-1)  # 句子数*单词数

        weight *= mask

        softmaxed_weight = nn.functional.softmax(weight, dim=-1)  # 句子数*单词数

        softmaxed_weight = softmaxed_weight / (softmaxed_weight.sum(dim=-1, keepdim=True) + 1e-13)

        softmaxed_weight = softmaxed_weight.unsqueeze(1)

        atts = softmaxed_weight.matmul(tensor).squeeze(1)

        return atts



class WordSentencePooling(nn.Module):
    
    def __init__(self, hidden_size, pooling_method1="max",pooling_method2="avg"):
        super(WordSentencePooling, self).__init__()
        self.hidden_size = hidden_size
        self.pooling1 = get_pooling_module(hidden_size,pooling_method1)
        self.pooling2 = get_pooling_module(hidden_size,pooling_method2)

    def forward(self,tensor1, tensor2, start_ids, end_ids):
        pooled_infered_tensor1 = self.pooling1(tensor1, start_ids, end_ids)
        pooled_infered_tensor2 = self.pooling2(tensor2, start_ids, end_ids)
        pooled_tensor = torch.cat([pooled_infered_tensor1, pooled_infered_tensor2], dim=-1)
        return pooled_tensor
    def get_output_dim(self):
        return self.pooling1.get_output_dim() + self.pooling2.get_output_dim()