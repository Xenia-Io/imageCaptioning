import skimage
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
from build_vocab import vocab
from utils import *
from encoder import ResNetEncoder
from decoder import Decoder
import config
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


def get_test_data(params):
    test_df = pd.read_csv(params["test_csv"])
    images_filenames = list(test_df["file_name"])
    images_files = [params['test_dir'] + "/" + images_filename for images_filename in images_filenames]
    captions = list(test_df["caption"])
    vocab = pickle.load(open(params["vocab_path"], "rb"))
    return images_files, captions, vocab


def load_models(params, vocab):
    encoder_model = ResNetEncoder()
    decoder_model = Decoder(params['num_of_features'], params['dim_of_features'], params['hidden_size'], vocab.count,
                            params['embed_size'])

    encoder = nn.DataParallel(encoder_model)
    decoder = nn.DataParallel(decoder_model)
    encoder.load_state_dict(torch.load(params["encoder_weights_path"]))
    decoder.load_state_dict(torch.load(params["decoder_weights_path"]))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("DEVICE: {}".format(device))
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    return encoder, decoder


def eval_one(img_path, encoder, decoder, transform, num_of_dim, num_of_feat, vocab):
    image = Image.open(img_path).convert("RGB")
    image = transform(image)
    image = Variable(image.cuda(0))
    feature = encoder(image.unsqueeze(0))
    feature = feature.view(feature.size(0), num_of_dim, num_of_feat).transpose(1, 2)

    word_ids, alpha = decoder.module.beam_search(feature, vocab, beam_size=3)
    return word_ids, alpha


def visualize_results(img_path, img_alpha, predicted_caption, savefile):
    image = Image.open(img_path).convert("RGB")
    image = image.resize((224, 224))
    plt.figure(figsize=(12, 12))
    plt.subplot(4, 5, 1)
    plt.text(0, 1, "<start>", color='black', backgroundcolor='white', fontsize=8)
    plt.imshow(image)
    plt.axis('off')

    words = predicted_caption
    words = words.split(' ')
    alpha_sum = np.zeros((14, 14))
    for t in range(len(words)):
        if words[t] == "<end>":
            break
        if t > 14:
            break
        plt.subplot(4, 5, t + 2)
        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=14)
        plt.imshow(image)

        alp_curr = img_alpha[t, :].view(14, 14)
        alpha_sum += alp_curr.data.cpu().numpy()
        alp_img = skimage.transform.pyramid_expand(alp_curr.data.cpu().numpy(), upscale=16, sigma=20)
        plt.gray()
        plt.imshow(alp_img, alpha=0.8)
        plt.axis('off')

    plt.figure()
    plt.imshow(image)
    alpha_sum = skimage.transform.pyramid_expand(alpha_sum, upscale=16, sigma=20)
    plt.imshow(alpha_sum, alpha=0.8)
    plt.show()

def eval_all(params):
    images_files, captions, vocab = get_test_data(params)
    encoder, decoder = load_models(params, vocab)
    encoder.eval()
    decoder.eval()
    predicted_caption_ids = []
    true_captions = []
    transform = process_image((224, 224), 224, "Val")
    file_not_exist = 0
    alphas = []
    # prediction
    with torch.no_grad():
        for img_path, cap in tqdm(zip(images_files, captions), total=len(images_files)):
            if os.path.exists(img_path):
                word_ids, alpha = eval_one(img_path, encoder, decoder, transform, params['dim_of_features'],
                                           params['num_of_features'], vocab)
                predicted_caption_ids.append(word_ids)
                true_captions.append(cap.lower())
                alphas.append(alpha)
            else:
                file_not_exist += 1

    predicted_captions = id2word_captions(predicted_caption_ids, vocab)
    output_df = {"predicted": [], "true": []}
    gram4_bleu_score_list = []
    gram3_bleu_score_list = []
    gram2_bleu_score_list = []
    gram1_bleu_score_list = []

    for i in range(len(predicted_captions)):
        output_df["predicted"].append(predicted_captions[i])
        output_df["true"].append(true_captions[i])
        gram4_bleu_score_list.append(compute_bleu_score(predicted_captions[i], true_captions[i], mode="4-gram"))
        gram3_bleu_score_list.append(compute_bleu_score(predicted_captions[i], true_captions[i], mode="3-gram"))
        gram2_bleu_score_list.append(compute_bleu_score(predicted_captions[i], true_captions[i], mode="2-gram"))
        gram1_bleu_score_list.append(compute_bleu_score(predicted_captions[i], true_captions[i], mode="1-gram"))

    bleu_score4 = np.mean(gram4_bleu_score_list)
    bleu_score3 = np.mean(gram3_bleu_score_list)
    bleu_score2 = np.mean(gram2_bleu_score_list)
    bleu_score1 = np.mean(gram1_bleu_score_list)
    print("4-gram bleu score on test dataset is : ", bleu_score4)
    print("3-gram bleu score on test dataset is : ", bleu_score3)
    print("2-gram bleu score on test dataset is : ", bleu_score2)
    print("1-gram bleu score on test dataset is : ", bleu_score1)
    save_path = os.path.join(params["test_dir"], "prediction.csv")
    pd.DataFrame(output_df).to_csv(save_path, index=False)
    print("results saved as prediction.csv")

    top_bleu4_inds = sorted(range(len(gram4_bleu_score_list)), key=lambda i: gram4_bleu_score_list[i])[-3:]
    bottom_bleu4_inds = sorted(range(len(gram4_bleu_score_list)), key=lambda i: -gram4_bleu_score_list[i])[-3:]
    top_bleu3_inds = sorted(range(len(gram3_bleu_score_list)), key=lambda i: gram3_bleu_score_list[i])[-3:]
    bottom_bleu3_inds = sorted(range(len(gram3_bleu_score_list)), key=lambda i: -gram3_bleu_score_list[i])[-3:]
    top_bleu2_inds = sorted(range(len(gram2_bleu_score_list)), key=lambda i: gram2_bleu_score_list[i])[-3:]
    bottom_bleu2_inds = sorted(range(len(gram2_bleu_score_list)), key=lambda i: -gram2_bleu_score_list[i])[-3:]
    top_bleu1_inds = sorted(range(len(gram1_bleu_score_list)), key=lambda i: gram1_bleu_score_list[i])[-3:]
    bottom_bleu1_inds = sorted(range(len(gram1_bleu_score_list)), key=lambda i: -gram1_bleu_score_list[i])[-3:]


    for i in range(3):
        savefile = 'top_bleu1_' + str(i) + '.png'
        ind = top_bleu1_inds[i]
        img_file = images_files[ind]
        img_alpha = alphas[ind]
        predicted_caption = predicted_captions[ind]
        visualize_results(img_file, img_alpha, predicted_caption, savefile)


    for i in range(3):
        savefile = 'top_bleu4_' + str(i) + '.png'
        ind = top_bleu4_inds[i]
        img_file = images_files[ind]
        img_alpha = alphas[ind]
        predicted_caption = predicted_captions[ind]
        visualize_results(img_file, img_alpha, predicted_caption, savefile)

    for i in range(3):
        savefile = 'bottom_bleu4_' + str(i) + '.png'
        ind = bottom_bleu4_inds[i]
        img_file = images_files[ind]
        img_alpha = alphas[ind]
        predicted_caption = predicted_captions[ind]
        visualize_results(img_file, img_alpha, predicted_caption, savefile)


if __name__ == '__main__':
    args = config.parse_opt()
    params = vars(args)
    eval_all(params)
