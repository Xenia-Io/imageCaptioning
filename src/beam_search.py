import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional
from torch.autograd import Variable


def beam_search(decoder, input_features, vocabulary, beam_size):
  batch_size = input_features.size(0)
  num_of_features = input_features.size(1)
  dim_of_features = input_features.size(2)
  vocab_size = len(vocabulary)
  max_cap_len = 13

  # expanding input features by replicating beam_size times
  input_features = input_features.expand(beam_size,num_of_features,dim_of_features)

  # topk words in the beginning are assigned as start tag
  prev_topk_words = torch.LongTensor([[vocabulary['<start>']]] * beam_size).cuda()
  prev_topk_wordembeds = decoder.word_embed(Variable(prev_topk_words)).squeeze(1)

  # keeping track of best k sequences at any time step
  squences = prev_topk_words

  # initializing top k scores with zero
  topk_scores = torch.zeros(beam_size, 1).cuda()

  # keeping track of alpha (attention weights) corresponding to best k sequences
  alpha_squences = torch.ones(beam_size,1,num_of_features).cuda()

  hidden_dim = decoder.hidden_size
  h_curr = Variable(torch.zeros(beam_size, hidden_dim).cuda())
  c_curr = Variable(torch.zeros(beam_size, hidden_dim).cuda())

  # initializing context as mean of all visual features
  context = torch.mean(input_features, 1)
  alpha = torch.zeros(beam_size, num_of_features).cuda()

  comp_seq_lst = []
  comp_alpha_lst = []
  comp_score_lst = []

  for t in range(max_cap_len):
    if(t!=0):
      prev_topk_wordembeds = decoder.word_embed(Variable(prev_topk_words)).squeeze(1)
      context, alpha = decoder.predict_attention(input_features.cuda(), h_curr)

    decoder_input = torch.cat([context, prev_topk_wordembeds], dim=1)
    h_curr, c_curr = decoder.lstm_cell(decoder_input, (h_curr, c_curr))
    scores = decoder.deep_output(h_curr)
    scores = torch.nn.functional.log_softmax(scores, dim=1)

    # update the scores by adding current step scores
    scores = topk_scores.expand_as(scores) + scores

    # update top k scores and best k words for t-th step
    topk_scores, topk_words = scores.view(-1).topk(beam_size, 0, True, True)

    squences, alpha_squences= findk_wordseq_alphas(topk_words, squences, alpha_squences, vocab_size, alpha)

    # find complete and incomplete sequences
    incomp_captions, comp_captions = find_sequences(topk_words,vocabulary)

    if(len(comp_captions)>0):
      comp_seq_lst.extend(squences[comp_captions])
      comp_alpha_lst.extend(alpha_squences[comp_captions])
      comp_score_lst.extend(topk_scores[comp_captions])
      beam_size = beam_size - len(comp_captions)


    if(beam_size == 0):
      # if all top k captions have been generated
      break

    if(t+1==max_cap_len):
      # if maximum length of captions have been achieved, stop it and store all generated captions
      comp_seq_lst.extend(squences[incomp_captions])
      comp_alpha_lst.extend(alpha_squences[incomp_captions])
      comp_score_lst.extend(topk_scores[incomp_captions])
      break

    # removing completed sequences and related columns in features and other variables
    squences = squences[incomp_captions] 
    alpha_squences = seqs_alpha[incomp_captions]
    prev_words = topk_words / vocab_size
    curr_words = topk_words % vocab_size
    h_curr = h_curr[prev_words[incomp_captions]]
    c_curr = c_curr[prev_words[incomp_captions]]
    input_features = input_features[prev_words[incomp_captions]]
    topk_scores = topk_scores[incomp_captions].unsqueeze(1)
    topk_words = curr_words[incomp_captions].unsqueeze(1)

  top_captionid = comp_score_lst.index(max(comp_score_lst))
  caption = comp_seq_lst[top_captionid][1:]
  alphas = comp_alpha_lst[top_captionid][1:]

  return caption, alphas


def findk_wordseq_alphas(topk_words, squences, alpha_squences, vocab_size, alpha):
  prev_words = topk_words/vocab_size
  curr_words = topk_words%vocab_size
  squences = torch.cat([squences[prev_words], curr_words.unsqueeze(1).float()], dim=1)
  alpha_squences = torch.cat([alpha_squences[prev_words], alpha[prev_words].unsqueeze(1).float()], dim=1)
  return squences, alpha_squences

def find_sequences(topk_words, vocabulary):
  curr_words = topk_words%len(vocabulary)
  incomp_captions = [ind for ind, curr_word in enumerate(curr_words) if curr_word != vocabulary("<end>")]
  comp_captions = list(set(range(len(curr_words))) - set(incomp_captions))
  return incomp_captions, comp_captions
