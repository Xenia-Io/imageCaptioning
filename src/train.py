import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
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

                
def train(encoder, decoder, enc_optimizer, dec_optimizer, data_loader, epoch, grad_clip, loss_function):

  # start training time
  start_train = time.time()
  
  # start time
  start = time.time()
  
  # define the mode of the models
  encoder = encoder.train().cuda()
  decoder = decoder.train().cuda()
  
  predictions_lengths = save_predictions_lengths(data_loader)
  
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


def prepare_training(input_csv, img_dir_train, img_dir_val, batch_size, shuffle, captions, vocab_path, num_of_features, beam_size,
                     dim_of_features, hidden_size, vocab_size, embedding_size, learning_rate, epochs, grad_clip):
  
  # set device 
  if torch.cuda.is_available():
    device = torch.device("cuda")
  else:
    device = torch.device("cpu")
  
  # load vocabulary
  with open(vocab_path, "rb") as vcb:
        vocabulary = pickle.load(vocab, vcb)
  
  
  # load images for training
  max_caption_len = max([len(caption) for caption in captions]) - 1
  train_data_loader = load_dataset(input_csv, img_dir_train, vocabulary, max_caption_len, batch_size, shuffle)
  
  # load images for validation
  max_caption_len = max([len(caption) for caption in captions]) - 1
  val_data_loader = load_dataset(input_csv, img_dir_val, vocabulary, max_caption_len, batch_size, shuffle)
  
  
  # initialize encoder
  encoder = VGGNetEncoder()
  
  enc_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                        lr = learning_rate)
  
  # initialize decoder
  decoder = Decoder(num_of_features, dim_of_features, hidden_size, vocab_size, embedding_size)
  
  dec_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                        lr = learning_rate)
   
  # move to GPU
  encoder = nn.DataParallel(encoder).to(device)
  decoder = nn.DataParallel(decoder).to(device)

  # loss function
  loss_function = nn.CrossEntropyLoss()
  
  # initialize scores
  best_bleu_score = -1000
  not_improved_counter = 0
  
  # train the model
  for epoch_i in range(epochs):
    
    # start training
    train(encoder, decoder, enc_optimizer, dec_optimizer, train_data_loader, epoch_i, grad_clip, loss_function)
  
    # start validation
    predictions_v = validation(encoder, decoder, val_data_loader, loss_function, epoch_i, vocabulary, beam_size, dim_of_features, num_of_features)

    # calculate blue score
    predicted_caps = decoding_caption(predictions_v["predicted"], vocabulary.idx2word)
    true_caps = decoding_caption(predictions_v["true"], vocabulary.idx2word)
    
    bleu_score_lst = []
    for i in range(len(predicted_caps)):
        bleu_score_lst.append(compute_bleu_score(predicted_caps[i], true_caps[i], mode="4-gram"))
    bleu_score = np.mean(bleu_score_lst)

    # increase counter to affect early stopping
    if bleu_score < best_bleu_score:
        not_improved_cnt += 1
    else:
        # learning is going well
        best_bleu_score = bleu_score
        not_improved_cnt = 0
        
    if not_improved_cnt == 10000:
        break

       
def main(args):
  
  prepare_training(args.input_csv, args.img_dir_train, args.img_dir_val, args.batch_size, args.shuffle, args.captions, args.vocab_path, args.num_of_features, args.beam_size,
                     args.dim_of_features, args.hidden_size, args.vocab_size, args.embedding_size, args.learning_rate, args.epochs, args.grad_clip)
  
  
if __name__ == "__main__":
    
    logger.debug("Running with args: {0}".format(args))
    main(args)
