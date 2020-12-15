from drlstm.layers import *
from drlstm.model_utils import *
import torch
import torch.nn as nn
import numpy as np
import random
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def _init_model_weights(module):
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

class DRLSTM(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_size,
                 embeddings=None,
                 padding_idx=0,
                 dropout=0.4, # change: dropout=0.5
                 num_classes=3,
                 device="cpu",
                 pooling_method_lst = ("max","avg","max","avg")):
        super(DRLSTM, self).__init__()
        self.vocab_size = vocab_size  # 42394
        self.embedding_dim = embedding_dim  # 300
        self.hidden_size = hidden_size  # 450
        self.num_classes = num_classes  # 3
        self.dropout = dropout
        self.device = device
        self.pooling_method_lst = pooling_method_lst
        self._embedding = nn.Embedding(self.vocab_size,
                                        self.embedding_dim,
                                        padding_idx=padding_idx,
                                        _weight=embeddings)
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
        pooled_premises = self._pooling_p(encoded_premises,  # model2
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
        return logits


class tanh_DRLSTM(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_size,
                 embeddings=None,
                 padding_idx=0,
                 dropout=0.4, # change: dropout=0.5
                 num_classes=3,
                 device="cpu",
                 pooling_method_lst = ("max","avg","max","avg")):
        super(tanh_DRLSTM, self).__init__()
        self.vocab_size = vocab_size  # 42394
        self.embedding_dim = embedding_dim  # 300
        self.hidden_size = hidden_size  # 450
        self.num_classes = num_classes  # 3
        self.dropout = dropout
        self.device = device
        self.pooling_method_lst = pooling_method_lst
        self._embedding = nn.Embedding(self.vocab_size,
                                        self.embedding_dim,
                                        padding_idx=padding_idx,
                                        _weight=embeddings)
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
                                         nn.Tanh())
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
        pooled_premises = self._pooling_p(encoded_premises,  # model2
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
        return logits


class DRLSTM_1(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_size,
                 embeddings=None,
                 padding_idx=0,
                 dropout=0.4, # change: dropout=0.5
                 num_classes=3,
                 device="cpu",
                 pooling_method_lst = ("max","avg","max","avg")):
        super(DRLSTM_1, self).__init__()
        self.vocab_size = vocab_size  # 42394
        self.embedding_dim = embedding_dim  # 300
        self.hidden_size = hidden_size  # 450
        self.num_classes = num_classes  # 3
        self.dropout = dropout
        self.device = device
        self.pooling_method_lst = pooling_method_lst
        self._embedding = nn.Embedding(self.vocab_size,
                                        self.embedding_dim,
                                        padding_idx=padding_idx,
                                        _weight=embeddings)
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
        encoded_premises2, _ = self._encoder3(projected_premises,
                                              premises_lengths,
                                              init_state=None)
        encoded_hypotheses2, _ = self._encoder3(projected_hypotheses,
                                                hypotheses_lengths,
                                                init_state=None)

        # pooling
        start_ids = torch.zeros(encoded_premises.shape[0]).long().to(self.device)
        end_ids = premises_lengths - 1
        pooled_premises = self._pooling_p(encoded_premises,  # model2
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
        return logits


class DRLSTM_3(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_size,
                 embeddings=None,
                 padding_idx=0,
                 dropout=0.4, # change: dropout=0.5
                 num_classes=3,
                 device="cpu",
                 pooling_method_lst = ("max","avg","max","avg")):
        super(DRLSTM_3, self).__init__()
        self.vocab_size = vocab_size  # 42394
        self.embedding_dim = embedding_dim  # 300
        self.hidden_size = hidden_size  # 450
        self.num_classes = num_classes  # 3
        self.dropout = dropout
        self.device = device
        self.pooling_method_lst = pooling_method_lst
        self._embedding = nn.Embedding(self.vocab_size,
                                        self.embedding_dim,
                                        padding_idx=padding_idx,
                                        _weight=embeddings)
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
        premises_mask = get_mask(premises, premises_lengths).to(self.device)
        hypotheses_mask = get_mask(hypotheses, hypotheses_lengths).to(self.device)
        # embedding
        embedded_premises = self._embedding(premises)
        embedded_hypotheses = self._embedding(hypotheses)
        # encoder
        # maybe: init_state not None here
        _, (final_state1) = self._encoder1(embedded_hypotheses,
                                           hypotheses_lengths,
                                           init_state=None)
        _, (final_state2) = self._encoder1(embedded_premises,
                                           premises_lengths,
                                           init_state=final_state1)
        _, (final_state3) = self._encoder1(embedded_hypotheses,
                                           hypotheses_lengths,
                                           init_state=final_state2)
        encoded_premises, _ = self._encoder2(embedded_premises,
                                             premises_lengths,
                                             init_state=final_state3)

        _, (final_state1) = self._encoder1(embedded_premises,
                                           premises_lengths,
                                           init_state=None)
        _, (final_state2) = self._encoder1(embedded_hypotheses,
                                           hypotheses_lengths,
                                           init_state=final_state1)
        _, (final_state3) = self._encoder1(embedded_premises,
                                           premises_lengths,
                                           init_state=final_state2)
        encoded_hypotheses, _ = self._encoder2(embedded_hypotheses,
                                               hypotheses_lengths,
                                               init_state=final_state3)
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
        pooled_premises = self._pooling_p(encoded_premises,  # model2
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
        return logits


class multi_model(nn.Module):
    def __init__(self, vocab_size,
                 embedding_dim,
                 hidden_size,
                 embeddings=None,
                 padding_idx=0,
                 dropout=0.4, # change: dropout=0.5
                 num_classes=3,
                 device="cpu",
                 pooling_method_lst = ("max","avg","max","avg")):
        super(multi_model,self).__init__()
        set_seed(seed=0)
        self.tanh_DRLSTM = tanh_DRLSTM(vocab_size,embedding_dim,hidden_size,embeddings,padding_idx,dropout,num_classes,device,pooling_method_lst)
        self.DRLSTM_1 = DRLSTM_1(vocab_size,embedding_dim,hidden_size,embeddings,padding_idx,dropout,num_classes,device,pooling_method_lst)
        self.DRLSTM_3 = DRLSTM_3(vocab_size,embedding_dim,hidden_size,embeddings,padding_idx,dropout,num_classes,device,pooling_method_lst)
        set_seed(seed=1)
        self.DRLSTM_1_seed1 = DRLSTM_1(vocab_size,embedding_dim,hidden_size,embeddings,padding_idx,dropout,num_classes,device,pooling_method_lst)
        set_seed(seed=2)
        self.DRLSTM_1_seed2 = DRLSTM_1(vocab_size,embedding_dim,hidden_size,embeddings,padding_idx,dropout,num_classes,device,pooling_method_lst)
        set_seed(seed=3)
        self.DRLSTM_1_seed3 = DRLSTM_1(vocab_size,embedding_dim,hidden_size,embeddings,padding_idx,dropout,num_classes,device,pooling_method_lst)
        set_seed(seed=0)
        self.weight_layer =  nn.Linear(6,1).to(device)

    def forward(self,premises,
                     premises_lengths,
                     hypotheses,
                     hypotheses_lengths):
        result1 = self.tanh_DRLSTM(premises,premises_lengths,hypotheses,hypotheses_lengths).unsqueeze(1)
        result2 = self.DRLSTM_1(premises, premises_lengths, hypotheses, hypotheses_lengths).unsqueeze(1)
        result3 = self.DRLSTM_3(premises, premises_lengths, hypotheses, hypotheses_lengths).unsqueeze(1)
        result4 = self.DRLSTM_1_seed1(premises, premises_lengths, hypotheses, hypotheses_lengths).unsqueeze(1)
        result5 = self.DRLSTM_1_seed2(premises, premises_lengths, hypotheses, hypotheses_lengths).unsqueeze(1)
        result6 = self.DRLSTM_1_seed3(premises, premises_lengths, hypotheses, hypotheses_lengths).unsqueeze(1)
        result = torch.cat([result1,
                                result2,
                                result3,
                                result4,
                                result5,
                                result6], dim = 1).transpose(1,2)
        final_result = self.weight_layer(result).squeeze()
        final_probs = nn.functional.softmax(final_result, dim=-1)
        return final_result, final_probs



class Ensemble_model1(nn.Module):
    def __init__(self, model_n):
        super(Ensemble_model1,self).__init__()
        self.model_n = model_n
        self.weight_layer = nn.Linear(self.model_n, 1)
    def forward(self, result_lst, probs_lst):
        # train
        result = torch.cat(result_lst, dim = 1).transpose(1,2)
        final_result = self.weight_layer(result).squeeze()
        # test
        final_probs = nn.functional.softmax(final_result, dim=-1)
        return final_result, final_probs

class Ensemble_model2(nn.Module):
    def __init__(self, model_n):
        super(Ensemble_model2,self).__init__()
        self.model_n = model_n
        self.weight_layer = nn.Linear(self.model_n, 1)
    def forward(self, result_lst, probs_lst):
        # train
        result = torch.cat(result_lst, dim = 1).transpose(1,2)
        final_result = self.weight_layer(result).squeeze()
        # test
        probs = torch.cat(probs_lst, dim = 1).transpose(1,2)
        final_probs = self.weight_layer(probs).squeeze()
        return final_result, final_probs

class Ensemble_model3(nn.Module):
    def __init__(self, model_n):
        super(Ensemble_model3, self).__init__()
        self.model_n = model_n
        self.weight_layer = nn.Linear(self.model_n, 1)
    def forward(self, result_lst, probs_lst):
        # test
        final_result = torch.mean(torch.cat(result_lst, dim=1), dim=1) # n 3
        final_probs = nn.functional.softmax(final_result, dim=-1)
        return None, final_probs

class Ensemble_model4(nn.Module):
    def __init__(self, model_n):
        super(Ensemble_model4, self).__init__()
        self.model_n = model_n
        self.weight_layer = nn.Linear(self.model_n, 1)
    def forward(self, result_lst, probs_lst):
        # test
        final_probs = torch.mean(torch.cat(probs_lst, dim = 1), dim=1)
        return None,final_probs