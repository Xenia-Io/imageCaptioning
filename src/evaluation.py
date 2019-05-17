import numpy as np
import pandas as pd
import argparse
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms
import pickle
from tqdm import tqdm
from PIL import Image

from utils import *
from encoder import ResNetEncoder
from decoder import Decoder
import config
from bleu import bleu


def get_test_data(params):

	test_df = pd.read_csv(params["test_csv"])
	images_files = list(test_df["file_name"])
	test_path_arr = list([(params['test_dir']+"/")*len(images_files)])
	images_files = test_path_arr + images_files
	captions = list(test_df["caption"])
	vocab = pickle.load(open(params["vocab_path"], "rb"))
	return images_files, captions, vocab

def load_models(params):
	encoder = ResNetEncoder()
	decoder = Decoder(params['num_of_features'],params['dim_of_features'],params['hidden_size'],params['vocab_size'],params['embed_size'])
    encoder = nn.DataParallel(encoder_model)
    decoder = nn.DataParallel(decoder_model)
    encoder.load_state_dict(torch.load(params["encoder_weights_path"]))
	decoder.load_state_dict(torch.load(params["decoder_weights_path"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE: {}".format(device))
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    return encoder, decoder


def eval_one(img_path, encoder, decoder, transform,num_of_dim,num_of_feat):

	image = Image.open(img_path).convert("RGB")
    image = transform(image)
    image = Variable(image.cuda())
    feature = encoder(image.unsqueeze(0))
    feature = feature.view(feature.size(0), num_of_dim, num_of_feat).transpose(1,2)

    word_ids, _ = decoder.module.beam_search(feature, vocab, beam_size=3)
	return word_ids    


def eval_all(params):

	images_files, captions, vocab = get_test_data(params)
    encoder, decoder = load_models(params)
    encoder.eval()
    decoder.eval()
    predicted_caption_ids = []
    true_captions =[]
    transform = set_transform(resize=(224, 224), crop_size=None, horizontal_flip=False, normalize=True)
    file_not_exist = 0

    # prediction
    with torch.no_grad():
        for img_path,cap in tqdm(zip(images_files,captions), total=len(images_files)):
            if os.path.exists(img_path):
                word_ids = eval_one(img_path, encoder, decoder, transform,params['dim_of_features'],params['num_of_features'])
                predicted_caption_ids.append(word_ids)
                true_captions.append(cap.lower())
            else:
                file_not_exist += 1

    predicted_captions = decode_caption(predicted_caption_ids, vocab.id2word)
    output_df = {"predicted": [], "true": []}
    bleu_score_list = []
    for i in range(len(predicted_captions)):
        output_df["predicted"].append(predicted_captions[i])
        output_df["true"].append(true_captions[i])
        bleu_score_list.append(bleu(predicted_captions[i], true_captions[i], mode="4-gram"))
        
    bleu_score = np.mean(bleu_score_list)
    print("bleu score on test dataset is : ",bleu_score)
    save_path = os.path.join(params["test_dir"], "prediction.csv")
    pd.DataFrame(output_df).to_csv(save_path, index=False)
    print("results saved as prediction.csv")




if __name__ == '__main__':

    args = config.parse_opt()
    params = vars(args) 
    eval_all(params)