import config
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from beam_search import beam_search
from torch.autograd import Variable
from encoder import VGGNetEncoder
from decoder import Decoder
from utils import *

def clip_gradient(optimizer, grad_clip):
  
    for ps in optimizer.param_groups:
      for p in ps["params"]:
        if p.grad is not None:
            p.grad.data.clamp_(-grad_clip, grad_clip)

                
def train_one_epoch(encoder, decoder, enc_optimizer, dec_optimizer, data_loader, grad_clip, loss_function):

  # start training time
  start_training = time.time()

  
  # define the mode of the models
  encoder = encoder.train()
  decoder = decoder.train()
  
  train_loss = []
  
  for images, target_captions in tqdm(data_loader):
    # Length of each caption
    caption_lengths = [len(caption) for caption in target_captions]
    
    # clear the gradients of all optimizers
    enc_optimizer.zero_grad()
    dec_optimizer.zero_grad()
    
    # move variables to cuda
    if torch.cuda.is_available():
      images = Variable(images.cuda())
      target_captions = Variable(target_captions.cuda())
    else:
      images = Variable(images)
      target_captions = Variable(target_captions)
      
    # extract features from encoder
    img_features = encoder(images)
    img_features = img_features.view(img_features.size(0), dim_of_features, num_of_features).transpose(1,2)
    
    # do forward for captioning
    predictions, alphas = decoder(img_features, target_captions[:, :-1])
    predictions = pack_padded_sequence(predictions, caption_lengths, batch_first=True)[0]
    captions = pack_padded_sequence(target_caption[:, 1:], caption_lengths, batch_first=True)[0]
      
    # calculate each loss of train
    loss = loss_function(predictions, captions)
    loss = loss + alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
    
    # back propagate
    loss.backward(retain_graph=False)
    
    # clip the gradient
    if grad_clip is not None:
        clip_gradient(dec_optimizer, grad_clip)
   
        if enc_optimizer is not None:
          clip_gradient(enc_optimizer, grad_clip)

    # update the weight in the optimizers
    if enc_optimizer is not None:
       enc_optimizer.step()
   
    dec_optimizer.step()
    train_loss.append(float(loss))
    del images, target_captions, img_features, predictions, alphas, loss
    gc.collect()        
    
  end_training = time.time()

  print("Training time was: ", end_training-start_training)
  print("Training loss is : ", np.sum(train_loss))
  
      
def validation(encoder, decoder, val_input, loss_fn, vocab, beam_size, feature_dim, num_features):
  predictions = list()
  encoder.eval()
  decoder.eval()
  for images, captions in val_input:
      if torch.cuda.is_available():
        images = Variable(images.cuda())
        captions = Variable(captions.cuda())
      else:
        images = Variable(images)
        captions = Variable(captions)
      img_features = encoder(images)
      img_features = img_features.view(img_features.size(0), feature_dim, num_features).transpose(1,2)
      prediction = beam_search(decoder, img_features, vocab, beam_size)

      predictions.append(prediction)

      # delete temporary data
      del images, img_features, prediction, captions
      gc.collect()

  return predictions


def train_all(params):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  with open(params["vocab_path"], "rb") as vocab_file:
    vocab = pickle.load(vocab_file)

  encoder = VGGNetEncoder()
  decoder = Decoder(params['num_of_features'],params['dim_of_features'],params['hidden_size'],params['vocab_size'],params['embed_size'])

  ecoder_optim = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                        lr=params["enc_lr"]) 
  decoder_optim = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters())
                                         lr=params["dec_lr"])
  encoder = nn.DataParallel(encoder).to(device)
  decoder = nn.DataParallel(decoder).to(device)

  transform_train = process_image(params['resize'],params['crop_size'], split = "Train")
  transform_val = process_image(params['resize'],params['crop_size'], split = "Val")

  train_dataloader = load_dataset(params['train_csv'],params['train_dir'],vocab,params['batch_size'],transform_train,shuffle=True)
  val_dataloader = load_dataset(params['val_csv'],params['val_dir'],vocab,params['batch_size'],transform_val,shuffle=True)

  grad_clip = 5.0
  loss_func = nn.CrossEntropyLoss()
  max_bleu = 0
  score_tolerance_count = 0
  max_tolerance_count = 10

  for epoch in range(params["n_epochs"]):
    print("Epoch : ",epoch+1)
    train_one_epoch(encoder, decoder, ecoder_optim, decoder_optim, train_dataloader, grad_clip, loss_func)
    predictions, truecaptions = validation(encoder, decoder, val_dataloader, loss_func, vocab, params['beam_size'],params['dim_of_features'],params['num_of_features'])
    predicted_captions = id2caption(predictions,vocab)
    true_captions = id2caption(truecaptions,vocab)
    bleu_scores = []
    
    for i in range(len(predicted_captions)):
      bleu_scores.append(compute_bleu_score(predicted_captions[i],true_captions[i]))
    curr_bleu_score =  np.mean(bleu_scores)

    if(curr_bleu_score<max_bleu):
      score_tolerance_count += 1
    else :
      max_bleu = curr_bleu_score
      score_tolerance_count = 0
      torch.save(encoder.state_dict(),params['encoder_weights_path'])
      torch.save(decoder.state_dict(),params['decoder_weights_path'])

    if(score_tolerance_count==max_tolerance_count):
      print("Early stopping")
      break


if __name__ == '__main__':
  args = config.parse_opt()
  params = vars(args) # convert to ordinary dict
  train_all(params)


