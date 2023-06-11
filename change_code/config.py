import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--noisetype", type=str, default="gauss25", choices=['gauss25', 'gauss5_50', 'poisson30', 'poisson5_50'])
parser.add_argument('--resume', type=str)
parser.add_argument('--checkpoint', type=str)
parser.add_argument('--data_dir', type=str,
                    default='./data/train/Imagenet_val')
parser.add_argument('--val_dirs', type=str, default='./data/validation')
parser.add_argument('--save_model_path', type=str,
                    default='../experiments/results')
parser.add_argument('--log_name', type=str,
                    default='b2u_unet_gauss25_112rf20')
parser.add_argument('--gpu_devices', default='0', type=str)
parser.add_argument('--parallel', action='store_true')
parser.add_argument('--n_feature', type=int, default=48)
parser.add_argument('--n_channel', type=int, default=3)

parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--w_decay', type=float, default=1e-8)

parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--n_snapshot', type=int, default=1)
parser.add_argument('--batchsize', type=int, default=4)
parser.add_argument('--patchsize', type=int, default=128)
parser.add_argument("--Lambda1", type=float, default=1.0)
parser.add_argument("--Lambda2", type=float, default=2.0)
parser.add_argument("--increase_ratio", type=float, default=20.0)

opt, _ = parser.parse_known_args()