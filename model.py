import torch
import torch.nn as nn
import torch.optim as optim

from module import *

class Classifier(object):

    def __init__(self, model, embedding, word_size, learning_rate):
        
        super(Classifier, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        print(self.model)
        
        if self.model != 'CNN-multichannel;':
            self.cnn = CNN(word_size).to(self.device)
            self.cnn.embedding.weight.data.copy_(embedding)
            if self.model == 'CNN-static':
                self.cnn.embedding.requires_grad = False
        else:
            self.cnn = CNN_multichannel(word_size).to(self.device)
            self.cnn.embedding_staitc.weight.data.copy_(embedding)
            self.cnn.embedding_non_static.weight.data.copy_(embedding)
            self.cnn.embedding_static.requires_grad = False
        
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.cnn.parameters(), lr = self.learning_rate)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
    
    def train(self, sentence_idx, labels, constraint = 3.0):
    
        self.cnn.train()
        sentence_idx = torch.tensor(sentence_idx, dtype = torch.long).to(self.device)
        labels = torch.tensor(labels, dtype = torch.long).to(self.device)
        self.classified = self.cnn(sentence_idx)
        self.loss = self.criterion(self.classified, labels)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        
        # keep l2 norm of [fullyconnected layer's weight vector] less than constraint = 3.0        
        weight_norm = torch.norm(self.cnn.fullyconnected.weight.data)
        #print(weight_norm)
        if weight_norm > constraint:
            #print('rescale occured')
            rescale = self.cnn.fullyconnected.weight.data * constraint / weight_norm
            self.cnn.fullyconnected.weight.data.copy_(rescale)
        
        return self.loss.item()
    
    def test(self, sentence_idx):
    
        self.cnn.eval()
        sentence_idx = torch.tensor(sentence_idx, dtype = torch.long).to(self.device)
        return self.cnn(sentence_idx)
       
    def save(self, epoch, model):
        path = './model/' + model + '.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.cnn.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
        
        
        
        