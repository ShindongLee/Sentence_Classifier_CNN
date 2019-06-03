import numpy as np
import argparse
import time
import torch
import torch.nn as nn

from preprocess import *
from model import *
from validation import validate
from graph import draw_graph

def train(model, num_epochs, learning_rate, mini_batch_size):

    print('Start Training...\nModel:', model, '\nTotal Epochs:', num_epochs, '\nLearning Rate:', learning_rate, '\nMini Batch Size:', mini_batch_size)

    w2i, embedding = set_embedding(model = model, pretrained_path = './pretrained', train_data_path ='./data/rt-polaritydata')
    longest_setenence_length =  longest_sentence(path = './data/rt-polaritydata')
    divide_train_and_test_set(path = './data/rt-polaritydata')
    
    train_data, train_labels = train_data_ready(path = './data/rt-polaritydata', w2i = w2i, length = longest_setenence_length)
    test_data, answers = test_data_ready(path = './data/rt-polaritydata', w2i = w2i, length = longest_setenence_length)
    
    classifier = Classifier(model, embedding, len(w2i), learning_rate)
    
    epoch_lst = []
    accuracy_lst = []
    best_accuracy = 0.0
    best_epoch = 0
    
    #print(classifier.static_check(0)) #you can check CNN-static's Embedding doesn't change during training
    
    for epoch in range(1, num_epochs + 1):
        
        start_time = time.time()
        
        print('-------------------------------------------------------------------')
        print('Epoch:', epoch)
        
        train_data, train_labels = shuffle_train_data(train_data, train_labels)
        train_num = len(train_data)
        i = 0
        while i < train_num:
            if i+mini_batch_size <= train_num:
                loss = classifier.train(train_data[i:i+mini_batch_size], train_labels[i:i+mini_batch_size], constraint = 3.0)
            else:
                loss = classifier.train(train_data[i:], train_labels[i:], constraint = 3.0)
            i+=mini_batch_size
            if i % 1500 == 0:
                print('loss: %.7f' % loss)
        
        end_time = time.time()
        time_elapsed = end_time - start_time
        print('Time Elapsed for This Epoch: %02d:%02d:%02d\n' % (time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))
        
        test_classified = classifier.test(test_data) #[N 2]
        accuracy = validate(test_classified, answers)
        print('accuracy: %.5f %%' % accuracy)
        
        epoch_lst.append(epoch)
        accuracy_lst.append(accuracy)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            classifier.save(epoch = epoch, model = model)
        print('Best Accuracy: %.5f %%, Best Epoch: %04d' %(best_accuracy, best_epoch))
        print('-------------------------------------------------------------------\n')
        
    #print(classifier.static_check(0)) #you can check CNN-static's Embedding doesn't change during training
    
    print('Training has been Completed')
    print('Best Model has been Saved')
    print(model)
    print('Best Accuracy: %.5f, Best Epoch: %04d' %(best_accuracy, best_epoch))
    
    draw_graph(epoch_lst, accuracy_lst, model)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Train CNN sentence classification model.')
    
    model_default = 'CNN-rand'
    num_epochs_default = 51
    learning_rate_default = 0.0001
    minibatch_size_default = 50
    
    parser.add_argument('--model', type = str, help = 'Type of Model', default = model_default)
    parser.add_argument('--num_epochs', type = int, help = 'Number of Epoch.', default = num_epochs_default)
    parser.add_argument('--learning_rate', type = float, help = 'Learning Rate for training.', default = learning_rate_default)
    parser.add_argument('--mini_batch_size', type = int, help = 'Mini Batch Size for SGD', default = minibatch_size_default)
  
    argv = parser.parse_args()

    model = argv.model
    num_epochs = argv.num_epochs
    learning_rate = argv.learning_rate
    mini_batch_size = argv.mini_batch_size
    
    model_list = ['CNN-rand', 'CNN-static', 'CNN-non-static', 'CNN-multichannel']
    assert model in model_list, 'Model should be one of [CNN-rand, CNN-static, CNN-non-static, CNN-multichannel]'

    train(model, num_epochs, learning_rate, mini_batch_size)     