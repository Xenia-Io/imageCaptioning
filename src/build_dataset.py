import pandas as pd
import os
import pickle
import numpy as np
import nltk
import argparse

def make_parse():
    parser = argparse.ArgumentParser(
        prog="build_dataset.py",
        usage="build dataset by generating csv file with filename and caption for train, test and validation",
        description="description",
        epilog="end",
        add_help=True
    )

    parser.add_argument("--source_data_dirc", type=str, default="~/ImageCaptioning/Flickr8kDataset/")
    parser.add_argument("--dest_data_dirac", type=str, default="~/ImageCaptioning/data/")
    args = parser.parse_args()
    return args

def split_dataset(source_dir, dest_dir):
    of = source_dir+"Flickr_8k.trainImages.txt"
    print(of)
    trainf = open(of,"r")
    train_images = trainf.readlines()
    print("splitting dataset for training")
    for img in train_images:
        oldimg = source_dir + "images/"+img.strip("\n")
        newimg = dest_dir + "train/images/"+img.strip("\n")
        os.rename(oldimg,newimg)

    validf = open(source_dir+"Flickr_8k.devImages.txt","r")
    valid_images = validf.readlines()
    print("splitting dataset for validation")
    for img in valid_images:
        oldimg = source_dir + "images/"+img.strip("\n")
        newimg = dest_dir + "valid/images/"+img.strip("\n")
        os.rename(oldimg,newimg)

    testf = open(source_dir+"Flickr_8k.testImages.txt","r")
    test_images = testf.readlines()
    print("splitting dataset for testing")
    for img in test_images:
        oldimg = source_dir + "images/"+img.strip("\n")
        newimg = dest_dir + "test/images/"+img.strip("\n")
        os.rename(oldimg,newimg)

def build_dataset(source_dir,dest_dir):
    token_file = source_dir+'Flickr8k.token.txt'
    annotations = open(token_file, 'r').read().strip().split('\n')
    id2caption = {}
    for i, row in enumerate(annotations):
        row = row.split('\t')
        caption_id = row[0]
        caption = row[1][:len(row[1])-2]
        id2caption[caption_id] = caption

    # building csv for training
    print("building csv files for training")
    trainImages = os.listdir(dest_dir+"train/images/")
    train_images = []
    train_captions = []
    for name in trainImages:
        for i in range(5):
            train_images.append(name)
            train_captions.append(id2caption[name+"#"+str(i)])

    train_dataset = {"file_name": train_images, "caption": train_captions}
    train_df = pd.DataFrame(train_dataset)
    train_df.to_csv(dest_dir+"train/image_captions_train.csv", index=False)

    # building csv for validation
    print("building csv file for validation")
    validImages = os.listdir(dest_dir+"valid/images/")
    valid_images = []
    valid_captions = []
    for name in validImages:
        for i in range(5):
            valid_images.append(name)
            valid_captions.append(id2caption[name+"#"+str(i)])

    valid_dataset = {"file_name": valid_images, "caption": valid_captions}
    valid_df = pd.DataFrame(valid_dataset)
    valid_df.to_csv(dest_dir+"valid/image_captions_valid.csv", index=False)

    # building csv for testing
    print("building csv file for testing")
    testImages = os.listdir(dest_dir+"test/images/")
    test_images = []
    test_captions = []
    for name in testImages:
        for i in range(5):
            test_images.append(name)
            test_captions.append(id2caption[name+"#"+str(i)])

    test_dataset = {"file_name": test_images, "caption": test_captions}
    test_df = pd.DataFrame(test_dataset)
    test_df.to_csv(dest_dir+"test/image_captions_test.csv", index=False)

if __name__ == "__main__":
    print("parsing arguments")
    # args = make_parse()
    # print(args)
    sourcedir = "/Users/shiprajain/ImageCaptioning/Flickr8kDataset/"
    destdir = "/Users/shiprajain/ImageCaptioning/data/"
    print("splitting dataset")
    split_dataset(sourcedir,destdir)
    print("building csv files")
    build_dataset(sourcedir,destdir)
                


