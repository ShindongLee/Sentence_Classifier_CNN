import numpy as np
import sys
import os
import torch
import torch.nn as nn
import time
import random
from gensim.models import KeyedVectors

def set_embedding(model, pretrained_path, train_data_path):

    start_time = time.time()
    
    pretrained_embedding_file = os.path.join(pretrained_path, 'GoogleNews-vectors-negative300.bin')
    train_pos_file = os.path.join(train_data_path, 'rt-polarity.pos')
    train_neg_file = os.path.join(train_data_path, 'rt-polarity.neg')
    
    embeddings_idx = KeyedVectors.load_word2vec_format(pretrained_embedding_file, binary=True)
    
    for word in embeddings_idx.vocab:
        if word.lower() not in embeddings_idx.vocab:
            embeddings_idx.vocab[word.lower()] = embeddings_idx.vocab[word]
            del embeddings_idx.vocab[word]
    
    print('File Loaded')

    word_idx = 1
    w2i = {}
    vector_list = []
    
    train_pos = open(train_pos_file, 'r', encoding = 'latin-1')
    pos_lines = train_pos.readlines()
    train_neg = open(train_neg_file, 'r', encoding = 'latin-1')
    neg_lines = train_neg.readlines()
    
    if model != 'CNN-rand':
        assert model in ['CNN-static', 'CNN-non-static', 'CNN-multichannel']
        
        w2i[' '] = 0
        word_vector = torch.randn([1, 300])
        norm = torch.norm(word_vector)
        word_vector /= norm
        vector_list.append(word_vector)
        
        for line in pos_lines:
            words = line.split(' ')
            for word in words:
                if word in w2i or not any(c.isalpha() for c in word):
                    continue
                w2i[word] = word_idx
                if word in embeddings_idx.vocab:
                    word_vector = torch.from_numpy(embeddings_idx.get_vector(word))
                    word_vector = torch.unsqueeze(word_vector, 0)
                else:
                    word_vector = torch.randn([1, 300])
                
                norm = torch.norm(word_vector)
                word_vector /= norm
                vector_list.append(word_vector)
                word_idx += 1
                
        for line in neg_lines:
            words = line.split(' ')
            for word in words:
                if word in w2i or not any(c.isalpha() for c in word):
                    continue
                w2i[word] = word_idx
                if word in embeddings_idx.vocab:
                    word_vector = torch.from_numpy(embeddings_idx.get_vector(word))
                    word_vector = torch.unsqueeze(word_vector, 0)
                else:
                    word_vector = torch.randn([1, 300])
                    
                norm = torch.norm(word_vector)
                word_vector /= norm    
                vector_list.append(word_vector)
                word_idx += 1
                
    else:
        assert model == 'CNN-rand'
        
        w2i[' '] = 0
        word_vector = torch.randn([1, 300])
        vector_list.append(word_vector)
        
        for line in pos_lines:
            words = line.split(' ')
            for word in words:
                if word in w2i or not any(c.isalpha() for c in word):
                    continue
                w2i[word] = word_idx
                word_vector = torch.randn([1, 300])
                norm = torch.norm(word_vector)
                word_vector /= norm
                vector_list.append(word_vector)
                word_idx += 1
                
        for line in neg_lines:
            words = line.split(' ')
            for word in words:
                if word in w2i or not any(c.isalpha() for c in word):
                    continue
                w2i[word] = word_idx
                word_vector = torch.randn([1, 300])
                norm = torch.norm(word_vector)
                word_vector /= norm
                vector_list.append(word_vector)
                word_idx += 1
                
    embeddings = torch.cat(vector_list, dim = 0)
    print('Word Size:', len(w2i))
    
    end_time = time.time()
    time_elapsed = end_time - start_time
    
    print('Time Elapsed for Setting Embedding: %02d:%02d:%02d' % (time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))
    
    return w2i, embeddings
    
def longest_sentence(path):

    train_pos_file = os.path.join(path, 'rt-polarity.pos')
    train_neg_file = os.path.join(path, 'rt-polarity.neg')
    
    train_pos = open(train_pos_file, 'r', encoding = 'latin-1')
    pos_lines = train_pos.readlines()
    train_neg = open(train_neg_file, 'r', encoding = 'latin-1')
    neg_lines = train_neg.readlines()
    
    longest = 0
    
    for line in pos_lines:
        words = line.split(' ')
        if len(words) > longest:
            longest = len(words)
            
    for line in neg_lines:
        words = line.split(' ')
        if len(words) > longest:
            longest = len(words)
            
    return longest
    
    
def divide_train_and_test_set(path):

    if os.path.exists(os.path.join(path, 'train_data_pos.txt')) and \
        os.path.exists(os.path.join(path, 'train_data_neg.txt')) and \
        os.path.exists(os.path.join(path, 'test_data_pos.txt')) and \
        os.path.exists(os.path.join(path, 'test_data_neg.txt')):
        
        return
        
    else:
        pos = open(os.path.join(path, 'rt-polarity.pos'), 'r', encoding = 'latin-1')
        pos_lines = pos.readlines()
        pos.close()
        
        neg = open(os.path.join(path, 'rt-polarity.neg'), 'r', encoding = 'latin-1')
        neg_lines = neg.readlines()
        neg.close()

        train_pos = open(os.path.join(path, 'train_data_pos.txt'), 'w', encoding = 'latin-1')
        train_neg = open(os.path.join(path, 'train_data_neg.txt'), 'w', encoding = 'latin-1')
        test_pos = open(os.path.join(path, 'test_data_pos.txt'), 'w', encoding = 'latin-1')
        test_neg = open(os.path.join(path, 'test_data_neg.txt'), 'w', encoding = 'latin-1')
        
        num_pos = len(pos_lines)
        num_neg = len(neg_lines)
        num_test_pos = num_pos // 10
        num_test_neg = num_pos // 10
        
        idx_test_pos = random.sample(range(num_pos), num_test_pos)
        idx_test_neg = random.sample(range(num_neg), num_test_neg)
        
        train_lines_pos = [x for i, x in enumerate(pos_lines) if i not in idx_test_pos]
        test_lines_pos = [x for i, x in enumerate(pos_lines) if i in idx_test_pos]
        train_lines_neg = [x for i, x in enumerate(neg_lines) if i not in idx_test_neg]
        test_lines_neg = [x for i, x in enumerate(neg_lines) if i in idx_test_neg]
        
        for line in train_lines_pos:
            train_pos.write(line)
        for line in test_lines_pos:
            test_pos.write(line)
        for line in train_lines_neg:
            train_neg.write(line)
        for line in test_lines_neg:
            test_neg.write(line)

def train_data_ready(path, w2i, length):
    train_data = []
    train_labels = []
    
    train_pos = open(os.path.join(path, 'train_data_pos.txt'), 'r', encoding = 'latin-1')
    train_neg = open(os.path.join(path, 'train_data_neg.txt'), 'r', encoding = 'latin-1')
    
    pos_lines = train_pos.readlines()
    neg_lines = train_neg.readlines()
    
    for line in pos_lines:
        words = line.split(' ')
        indices = []
        for word in words:
            if any(c.isalpha() for c in word):
                indices.append(w2i[word])     
        padding_num = length - len(indices)
        left_half = padding_num // 2
        right_half = padding_num - left_half
        for padding_idx in range(left_half):
            indices.insert(0, 0)
        for padding_idx in range(right_half):
            indices.append(0)
        train_data.append(indices)
        train_labels.append(0)
        
    for line in neg_lines:
        words = line.split(' ')
        indices = []
        for word in words:
            if any(c.isalpha() for c in word):
                indices.append(w2i[word])     
        padding_num = length - len(indices)
        left_half = padding_num // 2
        right_half = padding_num - left_half
        for padding_idx in range(left_half):
            indices.insert(0, 0)
        for padding_idx in range(right_half):
            indices.append(0)
        train_data.append(indices)
        train_labels.append(1)
    
    return train_data, train_labels

def test_data_ready(path, w2i, length):

    test_data = []
    answers = []
    
    test_pos = open(os.path.join(path, 'test_data_pos.txt'), 'r', encoding = 'latin-1')
    test_neg = open(os.path.join(path, 'test_data_neg.txt'), 'r', encoding = 'latin-1')
    
    pos_lines = test_pos.readlines()
    neg_lines = test_neg.readlines()
    
    for line in pos_lines:
        words = line.split(' ')
        indices = []
        for word in words:
            if any(c.isalpha() for c in word):
                indices.append(w2i[word])     
        padding_num = length - len(indices)
        left_half = padding_num // 2
        right_half = padding_num - left_half
        for padding_idx in range(left_half):
            indices.insert(0, 0)
        for padding_idx in range(right_half):
            indices.append(0)
        test_data.append(indices)
        answers.append(0)
        
    for line in neg_lines:
        words = line.split(' ')
        indices = []
        for word in words:
            if any(c.isalpha() for c in word):
                indices.append(w2i[word])     
        padding_num = length - len(indices)
        left_half = padding_num // 2
        right_half = padding_num - left_half
        for padding_idx in range(left_half):
            indices.insert(0, 0)
        for padding_idx in range(right_half):
            indices.append(0)
        test_data.append(indices)
        answers.append(1)
    
    return test_data, answers
    
def shuffle_train_data(train_data, train_labels):

    both = list(zip(train_data, train_labels))
    random.shuffle(both)
    train_data, train_labels = zip(*both)
    
    return train_data, train_labels


    
