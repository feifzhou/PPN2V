import argparse
# import template

parser = argparse.ArgumentParser()
parser.add_argument('--jobid', type=str, default='jobid', help='job id')
parser.add_argument('--mode', type=str, default='N2V', help='supervised(with GT as second channel), N2V, N2N(not implemented yet)')
parser.add_argument('--data', type=str, default='train.npy', help='dataset directory')
parser.add_argument('--nvalid', type=int, default=5, help='validation set')
parser.add_argument('--GT', type=str, default='GT.npy', help='GT dataset directory')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
parser.add_argument('--step_per_epoch', type=int, default=10, help='steps per epoch')
parser.add_argument('--minibatch_size', type=int, default=4, help='minibatch size for training')
parser.add_argument('--remove_edge', type=int, default=0, help='crop edge by how many pixels')
parser.add_argument('--virtualbatch_size', type=int, default=20, help='virtual batch size for training')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--GT_from_average', action='store_true', help='GT from average of dataset')
parser.add_argument('--patch_size', type=int, default=100, help='cropped patch size for training')
parser.add_argument('--test_only', action='store_true', help='prediction job')
parser.add_argument('--load_model', action='store_true', help='Load model')
parser.add_argument('--unet_depth', type=int, default=3, help='unet depth')

args = parser.parse_args()
# template.set_template(args)

def str2list(x, typ=int): return list(map(typ, filter(bool, x.split(','))))

if args.epochs == 0:
    args.epochs = 1e8

# if not args.dir: args.dir='../experiment/'+args.jobid
args.dir='../experiment/'+args.jobid
