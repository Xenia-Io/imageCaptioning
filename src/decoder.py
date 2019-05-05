import torch
import torch.nn as nn
import numpy as np

class Decoder(nn.Module):
    def __init__(self, num_of_features, dim_of_features, hidden_size, vocab_size, embedding_size):
      super(Decoder, self).__init__()
        
      self.num_of_features = num_of_features
      self.dim_of_features = dim_of_features
      self.hidden_size = hidden_size
      self.vocab_size = vocab_size
      self.embedding_size = embedding_size
      self.word_embed = nn.Embedding(self.vocab_size, self.embedding_size)
        
        # LSTM related variables
      self.lstm_cell = nn.LSTMCell(embedding_size+dim_of_features, hidden_size)
      self.deep_output = nn.Linear(self.hidden_size, self.vocab_size)
    

    # Initializes the hidden and memory states of the LSTM    
    def initialize_states(self, features):
      mean_features = features.mean(dim=1)
     
      if self.dim_of_features!= self.hidden_size:
        hidden_mlp_layer = nn.Linear(self.dim_of_features, self.hidden_size)
        memory_mlp_layer = nn.Linear(self.dim_of_features, self.hidden_size)
        # Initial states of the RNN are MLP outputs from a mean feature vector
        h0 = hidden_mlp_layer(mean_features)
        c0 = memory_mlp_layer(mean_features)

      else:
        h0 = mean_features
        c0 = mean_features

      return h0, c0
    
    def call_softMax(self, input):
      a = nn.Softmax(dim=1)(input).float() 
      return a
    
    # Finds the feature to be focused for the current time step 
    def predict_attention(self, input_features, hidden):
      batch_size = input_features.size(0)
      transform_vfeatures = nn.Linear(self.dim_of_features, self.dim_of_features, bias=False)
      output_v = transform_vfeatures(input_features)
      transform_hiddenvec = nn.Linear(self.hidden_size, self.dim_of_features, bias=False)
      output_h = transform_hiddenvec(hidden)
      hidden_visual_input = output_v + output_h
      bias = nn.Parameter(torch.zeros(self.dim_of_features)).view(1, -1, 1)
      activated_out = nn.ReLU()(hidden_visual_input + bias)
    
      # Add last layer for final transformation
      transform_toattention = nn.Linear(self.dim_of_features, 1, bias=False)
      attention_out = transform_toattention(activated_out)
      attention_out = attention_out.squeeze(2)   
      a = self.call_softMax(attention_out).unsqueeze(2)
      weighted_contexts = input_features * a
      z = torch.sum(weighted_contexts, dim=1)
      return a, z
          
    # Runs the forward pass of the LSTM 
    def forward(self, img_features, captions, dropout = False):
      # Initializing the required variables
      batch_size = img_features.size(0)
      h0, c0 = self.initialize_states(img_features)
      max_cap_len = max([len(caption) for caption in captions]) - 1
      predictions = torch.zeros(batch_size, max_cap_len, self.vocab_size).cuda()
      alphas = torch.zeros(batch_size, max_cap_len, img_features.size(1)).cuda()
        
      word_embeddings = self.word_embed(captions)
      if dropout:
        word_embeddings = nn.Dropout(0.5)(word_embeddings)
      
      for t in range(max_cap_len):

        # can change the computation of context vector for first word, may be average of all visual features instead of calling predict_attention
        context, alpha = self.predict_attention(img_features, h0)
        lstm_input = torch.cat((word_embeddings[:, t,:].squeeze(1), context), dim=1)
        
        h0, c0 = self.lstm_cell(lstm_input, (h0, c0))
        output = self.deep_output(nn.Dropout()(h0))

        predictions[:, t, :] = output
        alphas[:, t, :] = alpha
      return predictions, alphas

