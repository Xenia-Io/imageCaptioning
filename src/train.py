import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
# import utils.py
from beam_search import beamsearch
from torch.autograd import Variable

def save_predictions_lengths(data_loader):
  
  predictions_lengths = []
  
  for images, target_captions in tqdm(data_loader):
    max_cap_len = max([len(target_captions) for caption_i in target_captions]) - 1
    
    for caption_i in target_captions:
      predictions_lengths[i] = len(caption_i)
      
  return predictions_lengths


def clip_gradient(optimizer, grad_clip):
  
    for params in optimizer.param_groups:
      
        for param in params["params"]:
          
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def train(encoder, decoder, enc_optimizer, dec_optimizer, data_loader, epoch, grad_clip):

  # start training time
  start_train = time.time()
  
  # start time
  start = time.time()
  
  
  # define the mode of the models
  encoder = encoder.train().cuda()
  decoder = decoder.train().cuda()
  
  predictions_lengths = save_predictions_lengths(data_loader)
    
  # loss function
  loss_function = nn.CrossEntropyLoss()
  
  for image, target_caption in enumerate(tqdm(data_loader)):
    
    # clear the gradients of all optimizers
    enc_optimizer.zero_grad()
    dec_optimizer.zero_grad()
    
    # move variables to cuda
    if torch.cuda.is_available():
        image = Variable(image.cuda())
        target_caption = Variable(target_caption.cuda())
        
    # extract features from encoder
    img_features = encoder.forward(image)
    img_features = img_features.view(img_features.size(0), dim_of_features, num_of_features).transpose(1,2)
    
    # do forward for captioning
    predictions, alphas = decoder.forward(img_features, target_caption[:, :-1])
    predictions = pack_padded_sequence(predictions, predictions_lengths, batch_first=True)[0]
    captions = pack_padded_sequence(target_caption[:, 1:], predictions_lengths, batch_first=True)[0]
    
    
    # calculate each loss of train
    loss = loss_function(predictions, captions)
    loss = loss + alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
    
    # back propagate
    loss.backward()
    
    # clip the gradient
    if grad_clip is not None:
        clip_gradient(dec_optimizer, grad_clip)
   
        if encoder_optimizer is not None:
          clip_gradient(enc_optimizer, grad_clip)

  
  
    # update the weight in the optimizers
    if enc_optimizer is not None:
       enc_optimizer.step()
   
    dec_optimizer.step()
        
    start = time.time()
    
  end_training = time.time()
  print("Training time was: ", end_training-start_training)

  
    
def prepare_training(input_csv, img_dir, batch_size, shuffle, captions, vocab_path, num_of_features, 
                     dim_of_features, hidden_size, vocab_size, embedding_size, learning_rate, epochs, grad_clip):
  
  # set device 
  if torch.cuda.is_available():
    device = torch.device("cuda")
  else:
    device = torch.device("cpu")
  
  # load vocabulary
  with open(vocab_path, "rb") as vcb:
        vocabulary = pickle.load(vocab, vcb)
  
  
  # load images
  max_caption_len = max([len(caption) for caption in captions]) - 1
  data_loader = load_dataset(input_csv, img_dir, vocabulary, max_caption_len, batch_size, shuffle)
  
  
  # initialize encoder
  encoder = VGGNetEncoder()
  
  enc_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                        lr = learning_rate)
  
  
  # initialize decoder
  decoder = Decoder(num_of_features, dim_of_features, hidden_size, vocab_size, embedding_size)
  
  dec_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                        lr = learning_rate)
  
  
  
  # train the model
  for epoch_i in range(epochs):
    
    # start training
        train(encoder, decoder, enc_optimizer, dec_optimizer, data_loader, epoch_i, grad_clip)
      
      
def validation(encoder, decoder, val_input, loss_fn, epoch, vocab, beam_size, feature_dim, num_features):
  predictions = list()
  encoder.eval()
  decoder.eval()
  for images, captions in val_input:
      if torch.cuda.is_available():
        images = Variable(images.cuda())
        captions = Variable(captions.cuda())
      img_features = encoder(images)
      img_features = img_features.view(img_features.size(0), feature_dim, num_features).transpose(1,2)
      prediction = beam_search(decoder, img_features, vocab, beam_size)

      predictions.append(prediction)

      # delete temporary data
      del images, img_features, prediction, captions
      gc.collect()

  return predictions
      
