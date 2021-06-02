# coding: utf-8

import os
import copy
import torch
import joblib
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
from gensim.models import KeyedVectors
import utils.file_utils as ft
from utils.file_utils import from_project_root, dirname


class NeuralDataset(Dataset):
    def __init__(self, data_url, device, evaluating=False, cased = True):
        super().__init__()
        self.data_url = data_url
        self.label_dict, self.label_list = get_label_from_data(data_url)
        self.word_dict, _, self.char_dict, _ = get_vocab_from_data(data_url)
        self.sentences, self.records = load_input_data(data_url)
        self.device = device
        self.cased = cased
        self.evaluating = evaluating

    def __getitem__(self, index):
        return self.sentences[index], self.records[index]

    def __len__(self):
        return len(self.sentences)

    def collate_func(self, data_list):
        #print("collate_func")
        data_list = sorted(data_list, key=lambda tup: len(tup[0]), reverse=True)
        #print(data_list)
        sentence_list, records_list = zip(*data_list)  # un zip

        sentence_tensors = self.get_sentence_factors(sentence_list, self.device)
        # (sentences, sentence_lengths, sentence_words, sentence_word_lengths, sentence_word_indices, masks)

        max_sent_len = sentence_tensors[1][0]

        span_sentence_labels = list()
        single_region_labels = list()
        span_region_labels = list()
        for records, length in zip(records_list, sentence_tensors[1]):
            # 0->NL, 1->start
            labels = [0] * max_sent_len  # span's sentence label
            region_labels = [0] * max_sent_len
            for record in records:
                if (record[1] - record[0]) > 1:
                    labels[record[0]] = 1
                else:
                    region_labels[record[0]] = self.label_dict[records[(record[0], record[1])]] # label of single entity layer
            span_sentence_labels.append(labels) # span's sentence label
            single_region_labels.append(region_labels)

            for start in range(0, length):
                if labels[start] == 1:
                    for end in range(start+1, length):
                        span_region_labels.append(self.label_dict[records[(start, end + 1)]] if (start, end + 1) in records else 0)

        # print(region)
        span_sentence_labels = torch.LongTensor(span_sentence_labels).to(self.device)
        span_region_labels = torch.LongTensor(span_region_labels).to(self.device)
        single_region_labels = torch.LongTensor(single_region_labels).to(self.device)
        if self.evaluating:
            return sentence_tensors, single_region_labels, span_sentence_labels, span_region_labels, records_list
        return sentence_tensors, single_region_labels, span_sentence_labels, span_region_labels


    def get_sentence_factors(self, sentence_list, device):
        """ generate input tensors from sentence list

            Args:
                sentence_list: list of raw sentence
                device: torch device
                data_url: data_url used to locate vocab files

            Returns:
                sentences, tensor
                sentence_lengths, tensor
                sentence_words, list of tensor
                sentence_word_lengths, list of tensor
                sentence_word_indices, list of tensor

        """

        sentences = list()
        sentence_words = list()
        sentence_word_lengths = list()
        sentence_word_indices = list()

        unk_idx = self.word_dict["<unk>"]
        for sent in sentence_list:
            if self.cased:
                sentence = torch.LongTensor([self.word_dict[word] if word in self.word_dict else unk_idx for word in sent]).to(device)
            else:
                sentence = torch.LongTensor([self.word_dict[word.lower()] if word.lower() in self.word_dict else unk_idx for word in sent]).to(device)
            words = list() # matrix
            for word in sent:
                # if not self.cased:
                #     word = word.lower()
                # cased for char always
                words.append([self.char_dict[ch] if ch in self.char_dict else unk_idx for ch in word]) # matrix

            word_lengths = torch.LongTensor([len(word) for word in words]).to(device)
            word_lengths, word_indices = torch.sort(word_lengths, descending = True)

            words = np.array(words)[word_indices.cpu().numpy()]  # length sort after
            word_indices = word_indices.to(device)
            words = [torch.LongTensor(word).to(device) for word in words]
            words = pad_sequence(words, batch_first = True).to(device)
            sentences.append(sentence)
            sentence_words.append(words)
            sentence_word_lengths.append(word_lengths)
            sentence_word_indices.append(word_indices)

        sentence_lengths = [len(sentence) for sentence in sentences]
        sentences = pad_sequence(sentences, batch_first = True).to()
        masks = (sentences != 0).type(torch.int64)
        return sentences, sentence_lengths, sentence_words, sentence_word_lengths, sentence_word_indices, masks


def get_label_from_data(data_path, update = False):
    if data_path is None:
        return None
    data_dir_path = os.path.dirname(data_path)
    label_dict_path = os.path.join(data_dir_path, "label_dict.json")
    label_list_path = os.path.join(data_dir_path, "label_list.json")
    if (not update) and os.path.exists(label_dict_path) and os.path.exists(label_list_path):
        return ft.load(label_dict_path), ft.load(label_list_path)
    label_dict = {"O": 0}
    label_list = ["O"]
    file_lines = list()
    with open(data_path, 'r', encoding = 'utf-8') as file:
        for line in file:
            file_lines.append(line.strip())
    for idx in range(int((len(file_lines)) / 3)):
        annotation = file_lines[idx * 3 + 1].strip()
        if len(annotation):
            annotation = annotation.split("|")
            for an in annotation:
                an = an.strip()
                g2 = an.find(" ")
                label = an[g2 + 1:]
                if label in label_dict:
                    continue
                else:
                    label_dict[label] = len(label_dict)
                    label_list.append(label)
    ft.dump(label_dict, label_dict_path)
    ft.dump(label_list, label_list_path)
    return label_dict, label_list

def get_vocab_from_data(data_path_list, min_count = 2, update = False, cased = True):
    """ generate vocabulary and embeddings from data file, generated vocab files will be saved in
            data dir

        Args:
            data_path_list: url to data file(s), list or string
            update: force to update even vocab file exists
            min_count: minimum count of a word

        Returns:
            generated word embedding url
    """
    if isinstance(data_path_list, str):
        data_path_list = [data_path_list]
    data_dir_name = dirname(data_path_list[0])
    word_dict_path = os.path.join(data_dir_name, "word_dict.json")
    char_dict_path = os.path.join(data_dir_name, "char_dict.json")
    word_list_path = os.path.join(data_dir_name, "word_list.json")
    char_list_path = os.path.join(data_dir_name, "char_list.json")
    if (not update) and os.path.exists(word_dict_path) and os.path.exists(char_dict_path):
        return ft.load(word_dict_path), ft.load(word_list_path), ft.load(char_dict_path), ft.load(char_list_path)
    word_set = set()
    char_set = set()
    word_count = defaultdict(int)
    print("get vocab from data")

    for data_path in data_path_list:
        file_lines = list()
        with open(data_path, "r", encoding = "utf-8") as file:
            for line in file:
                file_lines.append(line.strip())
        for idx in range(int((len(file_lines)) / 3)):
            sentence = file_lines[idx * 3].split()
            for word in sentence:
                norm_word = word
                if not cased:
                    norm_word = word.lower()
                word_count[norm_word] += 1
                if word_count[norm_word] >= min_count:
                    word_set.add(norm_word)
                char_set = char_set.union(word)
    word_list = sorted(word_set)
    char_list = sorted(char_set)
    word_list = ["<pad>", "<unk>"] + word_list
    char_list = ['<pad>', '<unk>'] + char_list
    ft.dump(word_list, word_list_path)
    ft.dump(char_list, char_list_path)
    word_dict = ft.list_to_dict(word_list)
    char_dict = ft.list_to_dict(char_list)
    ft.dump(word_dict, word_dict_path)
    ft.dump(char_dict, char_dict_path)
    return word_dict, word_list, char_dict, char_list


def get_pretrained_embedding_path(word_list_path, freeze = False, pretrained_path = None, update = False):
    word_dir_name = os.path.dirname(word_list_path)
    embedding_path = os.path.join(word_dir_name, "embedding.npy") if pretrained_path else None
    if word_list_path is None or pretrained_path is None:
        return None
    if not update and os.path.exists(embedding_path):
        return embedding_path
    print("generate pretrained word vector from", pretrained_path)
    word_dict_path = os.path.join(word_dir_name, "word_dict.json")
    word_list = ft.load(word_list_path)
    word_list_update = list()
    word_list_update.append(word_list[0])
    word_list_update.append(word_list[1])
    binary = pretrained_path.endswith('.bin')
    kvs = KeyedVectors.load_word2vec_format(pretrained_path, binary = binary)
    embeddings = list()
    count = 0
    for index in range(2, len(word_list)):
        word = word_list[index]
        if word in kvs:
            embeddings.append(kvs[word])
            word_list_update.append(word)
        else:
            if freeze:
                # del word_list[index]
                continue
            else:
                count +=1
                embeddings.append(np.random.uniform(-0.25, 0.25, kvs.vector_size))
    embeddings = np.vstack([np.zeros(kvs.vector_size),  # for <pad>
                            np.random.uniform(-0.25, 0.25, kvs.vector_size),  # for <unk>
                            embeddings])
    np.save(embedding_path, embeddings)
    print("unknown word:", count)
    if freeze:
        ## update word_list and word_dict
        word_dict_update = ft.list_to_dict(word_list_update)
        ft.dump(word_list_update, word_list_path)
        ft.dump(word_dict_update, word_dict_path)
    return embedding_path

def load_input_data(data_path, update = False):
    """ load data into sentences and records

        Args:
            data_url: url to data file
            update: whether force to update
        Returns:
            sentences(raw), records
    """
    save_path = data_path.replace('.iob2', '.raw.pkl').replace('.dat', '.raw.pkl')
    if not update and os.path.exists(save_path):
        return joblib.load(save_path)
    file_lines = list()
    with open(data_path, 'r', encoding = 'utf-8') as file:
        for line in file:
            file_lines.append(line.strip())
    sentences = list()
    records = list()
    for idx in range(int((len(file_lines)) / 3)):
        sentence = file_lines[idx * 3]
        sentences.append(sentence.split())
        temp_record = dict()
        annotation = file_lines[idx * 3 + 1].strip()
        if len(annotation):
            annotation = annotation.split("|")
            for an in annotation:
                an = an.strip()
                g1 = an.find(",")
                g2 = an.find(" ")
                start = int(an[:g1])
                end = int(an[g1 + 1:g2])
                tag = an[g2 + 1:]
                key = (start, end)
                temp_record[key] = tag
        records.append(temp_record)
    joblib.dump((sentences, records), save_path)
    return sentences, records

def update_word(data_set_name, cased = True, min_count = 2):
    """ update vocab and embedding

        Args:
            data_set_name: data_set to data file for preparing vocab

    """
    dir_name = os.path.join(ft.from_project_root("data"), data_set_name)
    train_path = os.path.join(dir_name, "train.dat")
    dev_path = os.path.join(dir_name, "dev.dat")
    test_path = os.path.join(dir_name, "test.dat")
    data_path_list = [train_path, dev_path, test_path]
    # print(data_path_list)
    _, _, _, _ = get_vocab_from_data(data_path_list, min_count=min_count, update=True, cased = cased)
    return

def test(path):
    label_dict, label_list = get_label_from_data(path)
    num_of_label = len(label_list)
    device = 'cpu'
    train_set = NeuralDataset(path, device)
    records = train_set.records
    length_record = dict()
    sum = 0
    for i in range(len(records)):
        # print(records[i])
        for record in records[i]:
            start, end = record
            key = str(end - start)
            if key in length_record:
                length_record[key] += 1
            else:
                length_record[key] = 1
            sum += 1
        # if not records[i]:
        #     print("No mention here")
    print("sentence count", len(records))
    print("length count", length_record)
    print("record", sum)
    return sum

if __name__ == '__main__':
    print("genia")
    train_path = ft.from_project_root("data\\genia\\train.dat")
    dev_path = ft.from_project_root("data\\genia\\dev.dat")
    test_path = ft.from_project_root("data\\genia\\test.dat")
    print("train data set:")
    train_count = test(train_path)
    print("dev data set:")
    dev_count = test(dev_path)
    print("test data set:")
    test_count = test(test_path)

    print("\n\nACE2005")
    train_path = ft.from_project_root("data\\ACE2005\\train.dat")
    dev_path = ft.from_project_root("data\\ACE2005\\dev.dat")
    test_path = ft.from_project_root("data\\ACE2005\\test.dat")
    print("train data set:")
    _ = test(train_path)
    print("dev data set:")
    _ = test(dev_path)
    print("test data set:")
    _ = test(test_path)

    print("\n\nKBP")
    train_path = ft.from_project_root("data\\KBP\\train.dat")
    dev_path = ft.from_project_root("data\\KBP\\dev.dat")
    test_path = ft.from_project_root("data\\KBP\\test.dat")
    print("train data set:")
    _ = test(train_path)
    print("dev data set:")
    _ = test(dev_path)
    print("test data set:")
    _ = test(test_path)

