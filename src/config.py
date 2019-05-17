import argparse


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_path', type=str, default='~/ImageCaptioning/data/vocab.pkl',
						help='pickle file containing vocabulary')
    parser.add_argument('--num_of_features',type=int, default=196, help='14*14 image feature as output')
    parser.add_argument('--dim_of_features',type=int, default=512, help='depth of output feature from vggnet')
    parser.add_argument('--hidden_size',type=int, default=256, help='hidden vector size of rnn cell')
    parser.add_argument('--vocab_size',type=int, default=9000, help='number of unique words in vocabulary')
    parser.add_argument('--embed_size',type=int, default=256, help='size of word embeddings')
    parser.add_argument('--enc_lr',type=float, default=1e-4, help='learning rate for encoder')
    parser.add_argument('--dec_lr',type=float, default=4e-4, help='learning rate for decoder')
    parser.add_argument('--beam_size', type=int, default=3, help='for beam search')
    parser.add_argument('--batch_size',type=int, default=32, help='number of images in one batch')
    parser.add_argument('--resize',type=tuple, default=(256,256), help='size of transformed image')
    parser.add_argument('--crop_size',type=tuple, default=(224,224), help='crop the image while transformation')
    parser.add_argument('--n_epochs',type=int, default=200, help='number of epochs')
    parser.add_argument('--train_csv',type=str, default='~/ImageCaptioning/data/train/image_captions_train.csv', help='path to training file with image id and captions')
    parser.add_argument('--val_csv', type=str, default='~/ImageCaptioning/data/valid/image_captions_valid.csv', help='path to validation file with image id and captions')
    parser.add_argument('--test_csv', type=str, default='~/ImageCaptioning/data/test/image_captions_test.csv', help='path to test file with image id and captions')
    parser.add_argument('--train_dir', type=str, default='~/ImageCaptioning/data/train/images', help='path to training images directory')
    parser.add_argument('--val_dir', type=str, default='~/ImageCaptioning/data/valid/images', help='path to validation images directory')
    parser.add_argument('--test_dir', type=str, default='~/ImageCaptioning/data/test/images', help='path to test images directory')
    parser.add_argument('--encoder_weights_path',type=str, default='~/ImageCaptioning/model_weights/encoder_weights.pth', help='weight of trained encoder model')
    parser.add_argument('--decoder_weights_path',type=str, default='~/ImageCaptioning/model_weights/decoder_weights.pth', help='weight of trained decoder model')

    args = parser.parse_args()
    return args
