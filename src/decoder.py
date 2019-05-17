import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional
import gc

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
      self.lstm_cell = nn.LSTMCell(embedding_size + dim_of_features, hidden_size)
      self.deep_output = nn.Linear(self.hidden_size, self.vocab_size)
      self.transform_vfeatures = nn.Linear(self.dim_of_features, self.dim_of_features, bias=False)
      self.transform_hiddenvec = nn.Linear(self.hidden_size, self.dim_of_features, bias=False)
      self.bias = nn.Parameter(torch.zeros(self.num_of_features))
      self.transform_toattention = nn.Linear(self.dim_of_features, 1, bias=False)

    # Initializes the hidden and memory states of the LSTM    
    def initialize_states(self, features):
      mean_features = features.mean(dim=1).data.cpu()
      if self.dim_of_features != self.hidden_size:
        hidden_mlp_layer = nn.Linear(self.dim_of_features, self.hidden_size)
        memory_mlp_layer = nn.Linear(self.dim_of_features, self.hidden_size)
        # Initial states of the RNN are MLP outputs from a mean feature vector
        h0 = hidden_mlp_layer(mean_features.data)
        c0 = memory_mlp_layer(mean_features.data)

      else:
        h0 = torch.zeros(features.size(0), self.hidden_size)
        c0 = torch.zeros(features.size(0), self.hidden_size)
      if torch.cuda.is_available():
        h0 = h0.cuda(0)
        c0 = c0.cuda(0)
      h0 = Variable(h0)
      c0 = Variable(c0)
      gc.collect()
      if torch.cuda.is_available():
          torch.cuda.empty_cache()
      return h0, c0
    
    def call_softMax(self, input):
      a = nn.Softmax(dim=1)(input).float() 
      return a
    
    # Finds the feature to be focused for the current time step 
    def predict_attention(self, input_features, hidden):
      output_v = self.transform_vfeatures(input_features)
      output_h = self.transform_hiddenvec(hidden).unsqueeze(1)

      activated_out = nn.ReLU()(output_v + output_h + self.bias.view(1, -1, 1))
      attention_out = self.transform_toattention(activated_out).squeeze(2)

      a = self.call_softMax(attention_out)
      z = torch.sum(input_features * a.unsqueeze(2), dim=1)

      del output_v, output_h, activated_out, attention_out
      gc.collect()
      if torch.cuda.is_available():
          torch.cuda.empty_cache()
      return a, z
          
    # Runs the forward pass of the LSTM 
    def forward(self, img_features, captions, dropout = False):
      # Initializing the required variables
      img_features = Variable(img_features.cuda())
      batch_size = img_features.size(0)
      h0, c0 = self.initialize_states(img_features)
      max_cap_len = captions.data.size(1)
      if torch.cuda.is_available():
          predictions = Variable(torch.zeros(batch_size, max_cap_len, self.vocab_size).cuda(0))
          alphas = Variable(torch.zeros(batch_size, img_features.size(1), max_cap_len).cuda(0))
      else:
          predictions = Variable(torch.zeros(batch_size, max_cap_len, self.vocab_size))
          alphas = Variable(torch.zeros(batch_size, img_features.size(1), max_cap_len))

      context = torch.mean(img_features, 1)
      
      word_embeddings = self.word_embed(captions)
      if dropout:
        word_embeddings = nn.Dropout(0.5)(word_embeddings)
      
      for t in range(max_cap_len):
        if t!=0:
          alpha, context = self.predict_attention(img_features[:batch_size, :], h0[:batch_size, :])
          alphas[:batch_size, :,t] = alpha
          
        lstm_input = torch.cat([context, word_embeddings[:, t,:].squeeze(1)], dim=1)
        h0, c0 = self.lstm_cell(lstm_input, (h0, c0))
        output = self.deep_output(nn.Dropout()(h0))

        predictions[:, t, :] = output
        del context, lstm_input, output
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
      return predictions, alphas

    def get_words_inds(self, topk_words,vocab_size):
      prev_word_inds = topk_words / vocab_size
      next_word_inds = topk_words % vocab_size
      return prev_word_inds, next_word_inds

    def beam_search(self, feature, vocab, beam_size=3, max_len=13):
  
      lstm_cell = self.lstm_cell
      fc_out = self.deep_output
      attend = self.predict_attention

      feature = feature.expand(beam_size, self.num_of_features, self.dim_of_features)

      vocab_size = vocab.count
      embed_dim = self.embedding_size
      word_embed = self.word_embed

      k_prev_words = torch.FloatTensor([[vocab.word2id['<start>']]] * beam_size)
      prev_words_embedding = word_embed((k_prev_words.long()).cuda(0)).squeeze(1)


      topk_seqs = torch.cuda.FloatTensor(k_prev_words.cuda(0))
      topk_scores = torch.zeros(beam_size, 1).cuda(0)
      topk_alpha_seqs = torch.ones(beam_size, 1, self.num_of_features).cuda(0)

      seq_list_done = []
      alpha_list_done = []
      score_list_done = []

      h0, c0 = self.initialize_states(feature)
      feas = torch.mean(feature, dim=1)
      alpha = torch.zeros(beam_size, self.num_of_features).cuda(0)

      for t in range(max_len):
        if t != 0:
            prev_words_embedding = word_embed((k_prev_words.long()).cuda(0)).squeeze(1)
            alpha, feas = attend(feature.cuda(0), h0)

        inputs = torch.cat([feas, prev_words_embedding], dim=1)
        h0, c0 = lstm_cell(inputs, (h0, c0))
        scores = fc_out(h0)
        scores = torch.nn.functional.log_softmax(scores, dim=1)
        scores = topk_scores.expand_as(scores) + scores

        if t == 0:
            topk_scores, topk_words = scores[0].topk(beam_size, 0, True, True)
        else:
            topk_scores, topk_words = scores.view(-1).topk(beam_size, 0, True, True)

        prev_word_inds,next_word_inds = self.get_words_inds(topk_words,vocab_size)

        topk_seqs = torch.cat([topk_seqs[prev_word_inds], next_word_inds.unsqueeze(1).float()], dim=1)
        topk_alpha_seqs = torch.cat([topk_alpha_seqs[prev_word_inds], alpha[prev_word_inds].unsqueeze(1).float()], dim=1)
        incomp_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != vocab.word2id['<end>']]
        comp_inds = list(set(range(len(next_word_inds))) - set(incomp_inds))

        if len(comp_inds) > 0:
            seq_list_done.extend(topk_seqs[comp_inds])
            alpha_list_done.extend(topk_alpha_seqs[comp_inds])
            score_list_done.extend(topk_scores[comp_inds])
            beam_size -= len(comp_inds)

        if beam_size == 0:
            break
        if t + 1 == max_len:
            seq_list_done.extend(topk_seqs[incomp_inds])
            alpha_list_done.extend(topk_alpha_seqs[incomp_inds])
            score_list_done.extend(topk_scores[incomp_inds])
            break

        topk_seqs = topk_seqs[incomp_inds]
        topk_alpha_seqs = topk_alpha_seqs[incomp_inds]
        h0 = h0[prev_word_inds[incomp_inds]]
        c0 = c0[prev_word_inds[incomp_inds]]
        feature = feature[prev_word_inds[incomp_inds]]
        topk_scores = topk_scores[incomp_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomp_inds].unsqueeze(1)

      top_score_idx = score_list_done.index(max(score_list_done))
      alphas = alpha_list_done[top_score_idx][1:]
      seq = seq_list_done[top_score_idx][1:]
      

      return seq, alphas

