import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CNN(nn.Module):

    def __init__(self, word_size):
    
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings = word_size, embedding_dim = 300)
        self.conv1d_3 = nn.Conv1d(in_channels = 300, out_channels = 100, kernel_size = 3, stride = 1, padding = 0)
        self.conv1d_4 = nn.Conv1d(in_channels = 300, out_channels = 100, kernel_size = 4, stride = 1, padding = 0)
        self.conv1d_5 = nn.Conv1d(in_channels = 300, out_channels = 100, kernel_size = 5, stride = 1, padding = 0)
        self.maxpool = nn.AdaptiveMaxPool1d(output_size = 1)
        self.dropout = nn.Dropout(p = 0.5)
        self.fullyconnected = nn.Linear(in_features = 300, out_features = 2, bias = True)
    
    def forward(self, inputs): # input shape [N, T]
        
        #inputs = torch.tensor(inputs, dtype = torch.long) 
        inputs = self.embedding(inputs) #[N, T, 300]
        inputs = inputs.permute(0, 2, 1) #[N, 300, T]
        
        conv3 = F.relu(self.conv1d_3(inputs)) #[N 100 T']
        conv4 = F.relu(self.conv1d_4(inputs)) #[N 100 T'']
        conv5 = F.relu(self.conv1d_5(inputs)) #[N 100 T''']
        
        conv3_max = self.maxpool(conv3) #[N 100 1]
        conv4_max = self.maxpool(conv4) #[N 100 1]
        conv5_max = self.maxpool(conv5) #[N 100 1]
        
        concat = torch.cat([conv3_max, conv4_max, conv5_max], dim = 1) #[N 300 1]
        cocnat_reshape = torch.squeeze(concat) # [N 300]
        
        dropout = self.dropout(cocnat_reshape)
        
        classified = self.fullyconnected(dropout) #[N 2]
        
        return classified
        
class CNN_multichannel(nn.Module):

    def __init__(self, word_size):
    
        super(CNN_multichannel, self).__init__()
        self.embedding_static = nn.Embedding(num_embeddings = word_size, embedding_dim = 300)
        self.embedding_non_static = nn.Embedding(num_embeddings = word_size, embedding_dim = 300)
        
        self.conv1d_3 = nn.Conv1d(in_channels = 300, out_channels = 100, kernel_size = 3, stride = 1, padding = 0)
        self.conv1d_4 = nn.Conv1d(in_channels = 300, out_channels = 100, kernel_size = 4, stride = 1, padding = 0)
        self.conv1d_5 = nn.Conv1d(in_channels = 300, out_channels = 100, kernel_size = 5, stride = 1, padding = 0)
        
        self.maxpool = nn.AdaptiveMaxPool1d(output_size = 1)
        self.dropout = nn.Dropout(p = 0.5)
        self.fullyconnected = nn.Linear(in_features = 600, out_features = 2, bias = True)
        
    def forward(self, inputs): # input shape [N, T]
       
        #inputs = torch.tensor(inputs, dtype = torch.long) # input shape [N, T]
        
        inputs_static = self.embedding_static(inputs) 
        inputs_static = inputs_static.permute(0, 2, 1) #[N, 300, T]
        
        inputs_non_static = self.embedding_non_static(inputs)
        inputs_non_static = inputs_non_static.permute(0, 2, 1) #[N, 300, T]
        
        conv3_static = F.relu(self.conv1d_3(inputs_static)) #[N 100 T']
        conv4_static = F.relu(self.conv1d_4(inputs_static)) #[N 100 T'']
        conv5_static = F.relu(self.conv1d_5(inputs_static)) #[N 100 T''']
        
        conv3_non_static = F.relu(self.conv1d_3(inputs_non_static)) #[N 100 T']
        conv4_non_static = F.relu(self.conv1d_4(inputs_non_static)) #[N 100 T'']
        conv5_non_static = F.relu(self.conv1d_5(inputs_non_static)) #[N 100 T''']
        
        conv3_max_static = self.maxpool(conv3_static) #[N 100 1]
        conv4_max_static = self.maxpool(conv4_static) #[N 100 1]
        conv5_max_static = self.maxpool(conv5_static) #[N 100 1]
        
        conv3_max_non_static = self.maxpool(conv3_non_static) #[N 100 1]
        conv4_max_non_static = self.maxpool(conv4_non_static) #[N 100 1]
        conv5_max_non_static = self.maxpool(conv5_non_static) #[N 100 1]
        
        concat = torch.cat([conv3_max_static, conv4_max_static, conv5_max_static, conv3_max_non_static ,conv4_max_non_static ,conv5_max_non_static ], dim = 1) #[N 600 1]
        cocnat_reshape = torch.squeeze(concat) # [N 600]
        
        dropout = self.dropout(cocnat_reshape)
        
        classified = self.fullyconnected(dropout) # [N 2]
        
        return classified
