import argparse
from compute_stats import run_nuclei_inst_stat
from compute_stats import run_nuclei_inst_stat_one

parser = argparse.ArgumentParser(description='Execute panoptic segmentation over a all test set or a chosen test image')

parser.add_argument('--mode', type=str, help='an string indicating either if you want to obtain metrics for a single image (demo), or for all the test set (test)' , default=0,
                       )
parser.add_argument('--img', type=int, help='an integer indicating the image number (between 0 and 14)',
                       default = 0,
                       )

args = parser.parse_args()

pred_dir = '/media/user_home0/lvacostac/Vision/Final_Project/PROYECTO/hover_net/dataset/sample__tiles/pred/ResNet101_16_0.001_Adam_50/mat/'
true_dir = '/media/user_home0/jpuentes/CoNSeP/Test/Labels/'

if args.mode == 'test':
    run_nuclei_inst_stat(pred_dir, true_dir, print_img_stats=False)
    
elif args.mode == 'demo':
    run_nuclei_inst_stat_one(pred_dir, true_dir, args.img, print_img_stats=False)
    

