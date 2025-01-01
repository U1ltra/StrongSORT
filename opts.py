"""
@Author: Du Yunhao
@Filename: opts.py
@Contact: dyh_bupt@163.com
@Time: 2022/2/28 19:41
@Discription: opts
"""
import json
import argparse
from os.path import join

data = {
    'MOT17': {
        'val':[
            'MOT17-02-FRCNN',
            'MOT17-04-FRCNN',
            'MOT17-05-FRCNN',
            'MOT17-09-FRCNN',
            'MOT17-10-FRCNN',
            'MOT17-11-FRCNN',
            'MOT17-13-FRCNN'
        ],
        'test':[
            'MOT17-01-FRCNN',
            'MOT17-03-FRCNN',
            'MOT17-06-FRCNN',
            'MOT17-07-FRCNN',
            'MOT17-08-FRCNN',
            'MOT17-12-FRCNN',
            'MOT17-14-FRCNN'
        ]
    },
    'MOT20': {
        'test':[
            'MOT20-04',
            'MOT20-06',
            'MOT20-07',
            'MOT20-08'
        ]
    }
}

import os

def get_directory_names(path):
    """
    Returns a list of directory names in the specified path.
    
    Args:
        path (str): The path to search for directories
        
    Returns:
        list: A list of directory names (excluding files)
        
    Example:
        dirs = get_directory_names('/home/user/documents')
        # Returns: ['folder1', 'folder2', etc.]
    """
    try:
        # Get all items in the directory
        all_items = os.listdir(path)
        
        # Filter to only include directories using full paths for checking
        directories = [item for item in all_items 
                      if os.path.isdir(os.path.join(path, item))]
        
        return directories
        
    except FileNotFoundError:
        print(f"Error: Path '{path}' not found")
        return []
    except PermissionError:
        print(f"Error: Permission denied for path '{path}'")
        return []

class opts:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            'dataset',
            type=str,
            help='MOT17 or MOT20',
        )
        self.parser.add_argument(
            'mode',
            type=str,
            help='val or test',
        )
        self.parser.add_argument(
            '--BoT',
            action='store_true',
            help='Replacing the original feature extractor with BoT'
        )
        self.parser.add_argument(
            '--ECC',
            action='store_true',
            help='CMC model'
        )
        self.parser.add_argument(
            '--NSA',
            action='store_true',
            help='NSA Kalman filter'
        )
        self.parser.add_argument(
            '--EMA',
            action='store_true',
            help='EMA feature updating mechanism'
        )
        self.parser.add_argument(
            '--MC',
            action='store_true',
            help='Matching with both appearance and motion cost'
        )
        self.parser.add_argument(
            '--woC',
            action='store_true',
            help='Replace the matching cascade with vanilla matching'
        )
        self.parser.add_argument(
            '--AFLink',
            action='store_true',
            help='Appearance-Free Link'
        )
        self.parser.add_argument(
            '--GSI',
            action='store_true',
            help='Gaussian-smoothed Interpolation'
        )
        self.parser.add_argument(
            '--root_dataset',
            type=str,
            default='/home/jiaruili/Documents/exp/advTraj/baselines/surveillance_camera',
            required=True
            # default='/home/share/MOT'
        )
        self.parser.add_argument(
            '--path_AFLink',
            type=str,
            default='/data/dyh/results/StrongSORT_Git/AFLink_epoch20.pth'
        )
        self.parser.add_argument(
            '--dir_save',
            type=str,
            required=True
        )
        self.parser.add_argument(
            '--EMA_alpha',
            type=float,
            default=0.9
        )
        self.parser.add_argument(
            '--MC_lambda',
            type=float,
            default=0.98
        )

    def parse(self, args=''):
        if args == '':
          opt = self.parser.parse_args()
        else:
          opt = self.parser.parse_args(args)
        opt.min_confidence = 0.6
        opt.nms_max_overlap = 1.0
        opt.min_detection_height = 0
        if opt.BoT:
            opt.max_cosine_distance = 0.4
            opt.dir_dets = join(opt.root_dataset, "det_feats")
            # opt.dir_dets = '/home/jiaruili/Documents/exp/StrongSORT/{}_{}_YOLOX+BoT'.format(opt.dataset, opt.mode)
        else:
            opt.max_cosine_distance = 0.3
            opt.dir_dets = '/data/dyh/results/StrongSORT_Git/{}_{}_YOLOX+simpleCNN'.format(opt.dataset, opt.mode)
        if opt.MC:
            opt.max_cosine_distance += 0.05
        if opt.EMA:
            opt.nn_budget = 1
        else:
            opt.nn_budget = 100
        if opt.ECC:
            path_ECC = join(opt.root_dataset, "ecc.json")
            # path_ECC = '/home/jiaruili/Documents/exp/advTraj/baselines/surveillance_camera/ecc.json'
            # path_ECC = '/home/jiaruili/Documents/exp/StrongSORT/{}_ECC_{}.json'.format(opt.dataset, opt.mode)
            opt.ecc = json.load(open(path_ECC))
        
        if opt.dataset not in data or opt.mode not in data[opt.dataset]:
            opt.sequences = get_directory_names(join(opt.root_dataset, "imgs"))
            print(f"{len(opt.sequences)} sequences found in {opt.root_dataset}")
        else:
            opt.sequences = data[opt.dataset][opt.mode]
        # opt.dir_dataset = join(
        #     opt.root_dataset,
        #     opt.dataset,
        #     'train' if opt.mode == 'val' else 'test'
        # )
        opt.dir_dataset = join(opt.root_dataset, "imgs")

        # create dir_save if not exist
        opt.dir_save = join(opt.dir_save, f"strongSORT_det")
        if not os.path.exists(opt.dir_save):
            os.makedirs(opt.dir_save)
        else:
            raise ValueError(f"{opt.dir_save} already exists")

        return opt

opt = opts().parse()
