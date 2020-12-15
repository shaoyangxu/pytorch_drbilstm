from layers import *
from model_utils import *
import torch
import torch.nn as nn
def _init_model_weights(module):
    """
    Initialise the weights of the model.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)

    elif isinstance(module, nn.LSTM):
        nn.init.xavier_uniform_(module.weight_ih_l0.data)
        nn.init.orthogonal_(module.weight_hh_l0.data)
        nn.init.constant_(module.bias_ih_l0.data, 0.0)
        nn.init.constant_(module.bias_hh_l0.data, 0.0)
        hidden_size = module.bias_hh_l0.data.shape[0] // 4
        module.bias_hh_l0.data[hidden_size:(2*hidden_size)] = 1.0
        if (module.bidirectional):
            nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
            nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
            nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
            nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
            module.bias_hh_l0_reverse.data[hidden_size:(2*hidden_size)] = 1.0
class DRLSTM_BASE(nn.Module):
    def __init__(self,
                 embedding_layer,
                 hidden_size,
                 dropout=0.4, # change: dropout=0.5
                 num_classes=3,
                 device="cpu",
                 pooling_method_lst = ("max","avg","max","avg")):
        super(DRLSTM_BASE, self).__init__()
        self.hidden_size = hidden_size  # 450
        self.num_classes = num_classes  # 3
        self.dropout = dropout
        self.device = device
        self.pooling_method_lst = pooling_method_lst
        self._embedding = embedding_layer
        self.embedding_dim = self._embedding.hidden_size
        # 300->2d
        self._encoder1 = BilstmEncoder(
                nn.LSTM,
                self.embedding_dim,
                self.hidden_size,
                bidirectional=True
        )
        # 300->2d
        self._encoder2 = BilstmEncoder(
            nn.LSTM,
            self.embedding_dim,
            self.hidden_size,
            bidirectional=True
        )
        # 2d->8d
        self._attention = SoftmaxAttention()
        # 8d->d  maybe two
        self._projection = nn.Sequential(nn.Dropout(p=self.dropout),
                                         nn.Linear(4*2*self.hidden_size,
                                         self.hidden_size),
                                         nn.ReLU())
        # d->2d
        self._encoder3 = BilstmEncoder(
            nn.LSTM,
            self.hidden_size,
            self.hidden_size,
            bidirectional=True
        )
        # d->2d
        self._encoder4 = BilstmEncoder(
            nn.LSTM,
            self.hidden_size,
            self.hidden_size,
            bidirectional=True
        )

        # 2d->4d(2d -max_pooing-> 2d ; 2d -cat max and avg pooing-> 4d)
        self._pooling_p = WordSentencePooling(2* self.hidden_size, self.pooling_method_lst[0], self.pooling_method_lst[1])
        self._pooling_h = WordSentencePooling(2*self.hidden_size, self.pooling_method_lst[2], self.pooling_method_lst[3])
        # 8d(4d+4d)->d->num_class(3)
        # change1: input_dim: 8 d-> 12d
        # change2: add a linear layer
        pool_out_dim = self._pooling_p.get_output_dim() + self._pooling_h.get_output_dim()
        check_pool_out_dim(pool_out_dim, 2 * self.hidden_size, self.pooling_method_lst)
        self._classification = nn.Sequential(nn.Dropout(p=self.dropout),
                                             # nn.Linear(2 * 4 * self.hidden_size,
                                             #           self.hidden_size),
                                             nn.Linear(pool_out_dim, self.hidden_size),
                                             nn.Tanh(),
                                             nn.Dropout(p=self.dropout),
                                             nn.Linear(self.hidden_size,
                                                       self.num_classes))
        self.apply(_init_model_weights) # new

    def forward(self,
                premises,
                premises_lengths,
                hypotheses,
                hypotheses_lengths):
        raise NotImplementedError


class DRLSTM(DRLSTM_BASE):
    def forward(self,
                premises,
                premises_lengths,
                hypotheses,
                hypotheses_lengths):
        premises_mask = get_mask(premises, premises_lengths).to(self.device)
        hypotheses_mask = get_mask(hypotheses, hypotheses_lengths).to(self.device)
        # embedding
        embedded_premises = self._embedding(premises)
        embedded_hypotheses = self._embedding(hypotheses)
        # encoder
        # maybe: init_state not None here
        _, (final_state_of_hypotheses) = self._encoder1(embedded_hypotheses,
                                                        hypotheses_lengths,
                                                        init_state=None)
        encoded_premises, _ = self._encoder2(embedded_premises,
                                             premises_lengths,
                                             init_state=final_state_of_hypotheses)
        _, (final_state_of_premises) = self._encoder1(embedded_premises,
                                                      premises_lengths,
                                                      init_state=None)
        encoded_hypotheses, _ = self._encoder2(embedded_hypotheses,
                                               hypotheses_lengths,
                                               init_state=final_state_of_premises)
        # attention
        attended_premises, attended_hypotheses = self._attention(encoded_premises,
                                                                 premises_mask,
                                                                 encoded_hypotheses,
                                                                 hypotheses_mask)
        # enhance
        enhanced_premises = torch.cat([encoded_premises,
                                       attended_premises,
                                       encoded_premises - attended_premises,
                                       encoded_premises * attended_premises],
                                      dim=-1)
        enhanced_hypotheses = torch.cat([encoded_hypotheses,
                                         attended_hypotheses,
                                         encoded_hypotheses - attended_hypotheses,
                                         encoded_hypotheses * attended_hypotheses],
                                        dim=-1)
        # projection
        projected_premises = self._projection(enhanced_premises)
        projected_hypotheses = self._projection(enhanced_hypotheses)
        # infer_encoder
        encoded_hypotheses1, (final_state_of_projected_hypotheses) = self._encoder3(projected_hypotheses,
                                                                                    hypotheses_lengths,
                                                                                    init_state=None)
        encoded_premises2, _ = self._encoder4(projected_premises,
                                              premises_lengths,
                                              init_state=final_state_of_projected_hypotheses)
        encoded_premises1, (final_state_of_projected_premises) = self._encoder3(projected_premises,
                                                                                premises_lengths,
                                                                                init_state=None)
        encoded_hypotheses2, _ = self._encoder4(projected_hypotheses,
                                                hypotheses_lengths,
                                                init_state=final_state_of_projected_premises)

        # pooling
        start_ids = torch.zeros(encoded_premises1.shape[0]).long().to(self.device)
        end_ids = premises_lengths - 1
        pooled_premises = self._pooling_p(encoded_premises, # model2
                                        encoded_premises2,
                                        start_ids,
                                        end_ids)
        end_ids = hypotheses_lengths - 1
        pooled_hypotheses = self._pooling_h(encoded_hypotheses,
                                          encoded_hypotheses2,
                                          start_ids,
                                          end_ids)
        # cat
        cat_pooled_p_h = torch.cat([pooled_premises, pooled_hypotheses],
                                   dim=-1)
        # classification
        logits = self._classification(cat_pooled_p_h)
        # softmax
        probabilities = nn.functional.softmax(logits, dim=-1)
        return logits, probabilities