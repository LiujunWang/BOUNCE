# coding: utf-8

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from utils.torch_utils import get_device
from torch.nn.utils.rnn import pad_sequence
device = get_device("auto")

class NeuralModel(nn.Module):

    def __init__(self, hidden_size, num_of_label, embedding_path=None, bidirectional=True, lstm_layers=1,
                 num_of_word=None, embedding_dim=None, freeze=False, char_dim=0):
        super().__init__()

        if embedding_path:
            self.embedding = nn.Embedding.from_pretrained(
                embeddings=torch.Tensor(np.load(embedding_path)),
                freeze=freeze
            )

        else:
            self.embedding = nn.Embedding(num_of_word, embedding_dim, padding_idx=0)
            # init_embedding(self.embedding)

        self.embedding_dim = self.embedding.embedding_dim
        self.char_dim = char_dim
        self.word_repr_dim = self.embedding_dim + self.char_dim

        self.char_repr = CharLSTM(
            n_chars=1000,
            embedding_size=char_dim,
            hidden_size=char_dim // 2,
        ) if char_dim > 0 else None

        self.dropout = nn.Dropout(p=0.5)

        self.lstm = nn.LSTM(
            input_size=self.word_repr_dim,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=lstm_layers,
            batch_first=True
        )

        self.lstm_layers = lstm_layers
        self.num_of_label = num_of_label
        self.n_hidden = (1 + bidirectional) * hidden_size

        self.lstm_dropout = nn.Dropout(p = 0.5)
        self.norm = nn.LayerNorm(self.n_hidden)

        self.kernel_sizes = [1, 2]
        self.output_channel = 100
        self.convs = nn.ModuleList([nn.Conv2d(1, self.output_channel, (ks, self.n_hidden)) for ks in self.kernel_sizes])

        self.span_region_clf = RegionCLF(
            n_classes=self.num_of_label,
            kernel_num=len(self.kernel_sizes),
            output_channel=self.output_channel
        )

        self.single_region_labeler = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.num_of_label),
        )

        # start features classifier
        self.ht_conv = nn.Conv1d(self.n_hidden, hidden_size, 2, padding = 1)
        self.ht_labeler = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )


    def forward(self, sentences, sentence_lengths, sentence_words, sentence_word_lengths, sentence_word_indices, masks, span_sentence_labels=None):

        # sentences (batch_size, max_sent_len)
        # sentence_length (batch_size)
        word_repr = self.embedding(sentences)
        # word_feat shape: (batch_size, max_sent_len, embedding_dim)
        # add character level feature
        if self.char_dim > 0:
            # sentence_words (batch_size, *sent_len, max_word_len)
            # sentence_word_lengths (batch_size, *sent_len)
            # sentence_word_indices (batch_size, *sent_len, max_word_len)
            # char level feature
            char_feat = self.char_repr(sentence_words, sentence_word_lengths, sentence_word_indices)
            # char_feat shape: (batch_size, max_sent_len, char_feat_dim)

            # concatenate char level representation and word level one
            word_repr = torch.cat([word_repr, char_feat], dim=-1)
            # word_repr shape: (batch_size, max_sent_len, word_repr_dim)

        word_repr = self.dropout(word_repr)

        packed = nn.utils.rnn.pack_padded_sequence(word_repr, sentence_lengths, batch_first=True)
        out, (hn, _) = self.lstm(packed)
        # out packed_sequence(batch_size, max_sent_len, n_hidden)
        # hn (n_layers * n_directions, batch_size, hidden_size)

        max_sent_len = sentences.shape[1]
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(out, total_length=max_sent_len, batch_first=True)
        # unpacked (batch_size, max_sent_len, n_hidden)

        filter_represent = [conv(unpacked.unsqueeze(1)).squeeze(3).transpose(2, 1) for conv in self.convs]
        # filter_represent[0] (batch_size, max_sent_len, self.output_channel)
        # filter_represent[1] (batch_size, max_sent_len - 1, self.output_channel)
        # filter_represent, _ = attention(word_label_represent, word_label_represent, word_label_represent)
        single_region_outputs = self.single_region_labeler(unpacked).transpose(2, 1)

        # task1: head and tail sequence labeler
        
        span_represent = self.lstm_dropout(self.norm(unpacked))
        # span_sentence_outputs = self.ht_labeler(unpacked).transpose(2, 1)
        conv_represent = self.ht_conv(span_represent.transpose(2, 1))
        # conv_represent (batch_size, hidden_size, max_sent_len + 1)

        span_sentence_outputs = self.ht_labeler(conv_represent.transpose(2, 1)[:, :max_sent_len, :]).transpose(2, 1)
        #print(sentence_outputs.size())
        # shape of sentence_outputs: (batch_size, 2, lengths[0])

        # task2: region classification
        if span_sentence_labels is None:
            span_sentence_labels = torch.argmax(span_sentence_outputs, dim=1) * masks
            # sentence_labels (batch_size, lengths[0])

        regions = list()
        regions.append(list())
        regions.append(list())
        entity_length = list()
        for hidden, hidden_, sentence_label, length in zip(filter_represent[0], filter_represent[1], span_sentence_labels, sentence_lengths):
            for start in range(0, length):
                if sentence_label[start]:
                    for end in range(start+1, length):
                        regions[0].append(hidden[start:end+1])
                        regions[1].append(hidden_[start:end])
                    entity_length += [(i + 1 - start) for i in range(start + 1, length)]
        

        if len(regions[0]) == 0:
            return single_region_outputs, span_sentence_outputs, None
            
        represent = pad_sequence(regions[0], batch_first = True)
        represent_ = pad_sequence(regions[1], batch_first = True)
        entity_length = torch.FloatTensor(entity_length).view(-1, 1).to(device)

        span_region_outputs = self.span_region_clf(represent, represent_, entity_length)
        return single_region_outputs, span_sentence_outputs, span_region_outputs




class RegionCLF(nn.Module):
    def __init__(self, n_classes, kernel_num, output_channel):
        super().__init__()
        self.kernel_num = kernel_num
        self.output_channel = output_channel

        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.output_channel * self.kernel_num * 2, n_classes),
            # nn.Linear(self.repr_dim, n_classes),
        )
        # init_linear(self.fc[1])

    def forward(self, represent, represent_, entity_length):

        # represent(batch_size, length, hidden)
        # represent_(batch_size, length, hidden)
        represent = represent.transpose(2, 1)
        represent_ = represent_.transpose(2, 1)

        output_max = F.max_pool1d(represent, represent.size(2)).squeeze(2)
        output_max_ = F.max_pool1d(represent_, represent_.size(2)).squeeze(2)

        output_sum = torch.sum(represent, dim=-1)
        output_sum_ = torch.sum(represent_, dim=-1)
        entity_length = entity_length.expand_as(output_sum).to(device)

        output_mean = output_sum / entity_length
        output_mean_ = output_sum_ / (entity_length - 1)
        #output_mean = torch.div(output_sum, entity_length)
        #output_mean_ = torch.div(output_sum_, entity_length)
        data_repr = torch.cat([output_max, output_max_, output_mean, output_mean_], dim=-1)
        return self.fc(data_repr)
        # (batch_size, n_classes)



class CharLSTM(nn.Module):

    def __init__(self, n_chars, embedding_size, hidden_size, lstm_layers=1, bidirectional=True):
        super().__init__()
        self.n_chars = n_chars
        self.embedding_size = embedding_size
        self.n_hidden = hidden_size * (1 + bidirectional)

        self.embedding = nn.Embedding(n_chars, embedding_size, padding_idx=0)
        # self.char_dropout = nn.Dropout(p=0.5)
        # init_embedding(self.embedding)
        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=lstm_layers,
            batch_first=True,
        )
        # init_lstm(self.lstm)
    def sent_forward(self, words, lengths, indices):
        sent_len = words.shape[0]
        # words shape: (sent_len, max_word_len)

        embedded = self.embedding(words)
        # in_data shape: (sent_len, max_word_len, embedding_dim)

        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True)
        _, (hn, _) = self.lstm(packed)
        # shape of hn:  (n_layers * n_directions, sent_len, hidden_size)

        hn = hn.permute(1, 0, 2).contiguous().view(sent_len, -1)
        # shape of hn:  (sent_len, n_layers * n_directions * hidden_size) = (sent_len, 2*hidden_size)
        
        temp = torch.ones(hn.size()).to(device)
        # shape of indices: (sent_len, max_word_len)
        temp[indices] = hn  # unsort hn
        # unsorted = hn.new_empty(hn.size())
        # unsorted.scatter_(dim=0, index=indices.unsqueeze(-1).expand_as(hn), src=hn)
        return temp

    def forward(self, sentence_words, sentence_word_lengths, sentence_word_indices):
        # sentence_words [batch_size, *sent_len, max_word_len]
        # sentence_word_lengths [batch_size, *sent_len]
        # sentence_word_indices [batch_size, *sent_len, max_word_len]

        batch_size = len(sentence_words)
        batch_char_feat = torch.nn.utils.rnn.pad_sequence(
            [self.sent_forward(sentence_words[i], sentence_word_lengths[i], sentence_word_indices[i])
             for i in range(batch_size)], batch_first=True)

        return batch_char_feat
        # (batch_size, sent_len, 2 * hidden_size)


def main():
    pass


if __name__ == '__main__':
    main()
