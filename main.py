from UGATIT import UGATIT
import argparse
from utils import *

"""parsing and configuration"""

def parse_args():
    desc = "Pytorch implementation of U-GAT-IT"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='[train / test]')
    parser.add_argument('--light', type=str2bool, default=False, help='[U-GAT-IT full version / U-GAT-IT light version]')
    parser.add_argument('--dataset', type=str, default='YOUR_DATASET_NAME', help='dataset_name')
    parser.add_argument('--testA_dir', type=str, default=None,
                        help='Optional path to A-domain test images')
    parser.add_argument('--testB_dir', type=str, default=None,
                        help='Optional path to B-domain test images')

    parser.add_argument('--iteration', type=int, default=1000000, help='The number of training iterations')
    parser.add_argument('--batch_size', type=int, default=1, help='The size of batch size')
    parser.add_argument('--print_freq', type=int, default=1000, help='The number of image print freq')
    parser.add_argument('--save_freq', type=int, default=100000, help='The number of model save freq')
    parser.add_argument('--decay_flag', type=str2bool, default=True, help='The decay_flag')

    parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='The weight decay')
    parser.add_argument('--adv_weight', type=int, default=1, help='Weight for GAN')
    parser.add_argument('--cycle_weight', type=int, default=10, help='Weight for Cycle')
    parser.add_argument('--identity_weight', type=int, default=10, help='Weight for Identity')
    parser.add_argument('--cam_weight', type=int, default=1000, help='Weight for CAM')
    parser.add_argument('--global_dis_ratio', type=float, default=0.5,
                        help='Ratio of global discriminator loss (0~1)')

    parser.add_argument('--use_ds', type=str2bool, default=False,
                        help='enable style diversity with random style vectors')
    parser.add_argument('--style_dim', type=int, default=8, help='dimension of style vector')
    parser.add_argument('--ds_weight', type=float, default=1.0, help='diversity sensitive loss weight')

    parser.add_argument('--use_spade_adalin', type=str2bool, default=False,
                        help='enable SPADE-AdaLIN in generator')
    parser.add_argument('--style_nc', type=int, default=256, help='style code dimension')
    parser.add_argument('--lambda_style', type=float, default=1.0, help='style consistency loss weight')
    parser.add_argument('--lambda_lowpass', type=float, default=5.0, help='low-pass tone loss weight')

    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    parser.add_argument('--n_res', type=int, default=4, help='The number of resblock')
    parser.add_argument('--n_dis', type=int, default=6, help='The number of discriminator layer')

    parser.add_argument('--img_size', type=int, default=256, help='Image height')
    parser.add_argument('--aspect_ratio', type=float, default=1.0, help='Width / height ratio')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')
    parser.add_argument('--center_crop', type=str2bool, default=False,
                        help='Center crop to maintain aspect ratio instead of simple resize')

    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the results')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Set gpu mode; [cpu, cuda]')
    parser.add_argument('--benchmark_flag', type=str2bool, default=False)
    parser.add_argument('--resume', type=str2bool, default=False)
    parser.add_argument('--resume_iter', type=int, default=0,
                        help='The iteration of checkpoints to load for testing')
    parser.add_argument('--use_checkpoint', type=str2bool, default=False,
                        help='enable gradient checkpointing')

    args = parser.parse_args()
    args.img_w = int(args.img_size * args.aspect_ratio)
    return check_args(args)

"""checking arguments"""
def check_args(args):
    # --result_dir
    check_folder(os.path.join(args.result_dir, args.dataset, 'model'))
    check_folder(os.path.join(args.result_dir, args.dataset, 'img'))
    check_folder(os.path.join(args.result_dir, args.dataset, 'test', 'A2B'))
    check_folder(os.path.join(args.result_dir, args.dataset, 'test', 'B2A'))

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    if not 0.0 <= args.global_dis_ratio <= 1.0:
        raise ValueError('global_dis_ratio must be between 0 and 1')
    if args.img_w % 4 != 0:
        raise ValueError('img_size * aspect_ratio must be divisible by 4')
    if args.use_ds and args.style_dim <= 0:
        raise ValueError('style_dim must be positive when use_ds is enabled')
    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    # open session
    gan = UGATIT(args)

    # build graph
    gan.build_model()

    if args.phase == 'train' :
        gan.train()
        print(" [*] Training finished!")

    if args.phase == 'test' :
        try:
            gan.test()
        except FileNotFoundError as e:
            print(f'Error: {e}')
            return
        print(" [*] Test finished!")

if __name__ == '__main__':
    main()
