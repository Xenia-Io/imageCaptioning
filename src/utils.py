import os
import torchvision.transforms as transforms
import torch.utils.data as datautil
import nltk
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from nltk import word_tokenize
from nltk import bleu_score
import re


# Transforms the image to match VGG16 inputs
def process_image(resize, crop_size, split):
    image_mean = (0.485, 0.456, 0.406)
    image_std = (0.229, 0.224, 0.225)
    if split == "Train":
        transformed_image = transforms.Compose([transforms.Resize(resize),
                                      transforms.RandomCrop(crop_size),
                                      transforms.ToTensor(),
                                      transforms.Normalize(image_mean, image_std)])
    elif split == "Val":
        transformed_image = transforms.Compose([transforms.Scale(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(image_mean, image_std)])
     
    return transformed_image
  
  

def compute_bleu_score(predictedCaptions, trueCaptions, mode="4-gram"):
    if mode == "1-gram":
        weights = [1.0]
    elif mode == "2-gram":
        weights = [0.5, 0.5]
    elif mode == "3-gram":
        weights = [0.33, 0.33, 0.33]
    elif mode == "4-gram":
        weights = [0.25, 0.25, 0.25, 0.25]
    else:
        sys.stdout.write("Not support mode")
        sys.exit()
        
    return bleu_score.sentence_bleu([predictedCaptions], trueCaptions, weights=weights)


# Creates a dataset that returns the images and the corresponding captions.
class Flickr8KDataset(datautil.Dataset):
    def __init__(self, csv_file, root_dir, vocabulary, transform=None):
            """
            Args:
                csv_file (string): Path to the csv file with annotations.
                root_dir (string): Directory with all the images.
                vocabulary (vocab object): Object with the word_to_ind mappings.
                transform (callable, optional): Optional transform to be applied
                    on a sample.
            """
            self.input_frame = pd.read_csv(csv_file)
            self.root_dir = root_dir
            self.transform = transform
            self.vocab = vocabulary

    def __len__(self):
        return len(self.input_frame)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.input_frame.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform (image)
        caption = self.input_frame.iloc[idx, 1]
        # Tokenize the word in the captions
        caption_tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        max_caption_len = max([len(tokens) for token in caption_tokens])
        # Convert the captions to the corresponding word ids from the built vocabulary.
        captions = []
        captions.append(self.vocab['<start>'])
        for tokens in caption_tokens:
            captions.append(self.vocab[token] if token in self.vocab else self.vocab['<unk>'] for token in tokens)
        captions.append(self.vocab['<end>'] + word_dict['<pad>'] * (max_caption_len - len(tokens)))
        target = torch.Tensor(captions)

        return image, target

def collate_fn(data):
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: -len(x[1]))
    # unzipping the tuples
    images, captions = zip(*data)
    # Converting tuple to a torch variable
    images = torch.stack(images, dim=0)
    # Merge captions (from tuple of 1D tensor to 2D tensor).
    captions = nn.utils.rnn.pad_sequence(list(captions[:]), batch_first=True)
    captions = captions.type(torch.IntTensor).long()

    return torch.Tensor(images), captions

# Returns the input data according to the batch size
def load_dataset(input_csv, img_dir, vocab, batch_size, shuffle):
    flickr_data = Flickr8KDataset(input_csv, img_dir, vocab, transform)
    data_loader = datautil.DataLoader(dataset=flickr_data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

    return data_loader

# Converts the captions generated as ids from decoder to words.
def id2word_captions(captions_list, id2word_map):
  parsed_captions = []
  for caption in caption_list:
    caption = caption.data.cpu().numpy()
    caption = [id2word_map[id] for id in caption if id2word_map[id]!= "<end>"]
    caption = ' '.join(caption)
    caption = re.sub(r' \.', '.', caption)
    parsed_captions.append(caption)
    
  return parsed_captions
