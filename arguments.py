from argparse import ArgumentParser

parser = ArgumentParser(description='ConvLSTM model parameters')

parser.add_argument('--input_size', default=2, type=int,
                    metavar='N', help='input size size for ConvLSTM')
parser.add_argument('--hidden_size', default=6, type=int,
                    metavar='N', help='hidden size for ConvLSTM')
parser.add_argument('--num_layers', default=1, type=int,
                    metavar='N', help='layers for ConvLSTM')
parser.add_argument('--batch_size', '-b', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--learning_rate', default=1e-3,
                    type=float, help='learning rate of model')
parser.add_argument('--epoches', default=100, type=int,
                    metavar='N', help='train time of each epoch')

parser.add_argument('--gblur_size_width', default=5,
                    type=int, help='gblur_size_width')
parser.add_argument('--gblur_size_high', default=5,
                    type=int, help='gblur_size_high')
parser.add_argument('--windows', '-w', default=4, type=int,
                    help='prediction window size')
parser.add_argument('--threshold', default=0.2, type=float,
                    help='control predict_tile choose')
parser.add_argument('--show_image', default=False,
                    type=bool, help='show test image or not')

# path setting
parser.add_argument('--self_predict', default=True, type=bool)
# parser.add_argument('--model_path', default='./model/', type=str)
parser.add_argument(
    '--model_path', default='./model/', type=str)
parser.add_argument(
    '--sal_path', default='./SalData/', type=str)
# parser.add_argument('--sal_path', default='./model/Saliency/', type=str)


def get_args():
    args = parser.parse_args()
    return args
