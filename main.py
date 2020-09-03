import torch
import argparse
from Face_parsing.test import parsing
from SEAN.options.test_options import TestOptions
from SEAN.test import reconstruct

def main(args):

    print(args)
    torch.manual_seed(args.seed)

    if args.mode.lower() in ['black','brown','blond','red','blue']:
        # parsing only src_image
        parsing(respth='./results/label', dspth='./data/src')

        reconstruct(args)

    elif args.mode.lower() == 'custom':
        # parsing src_image
        parsing(respth='./results/label', dspth='./data/src') # results) results/label/src.~
        # parsing dyeing_target_image
        parsing(respth='./results/label', dspth='./data/ref') # results) results/label/ref.~
    
        reconstruct(args)

    else:
        raise NotImplementedError


if __name__ == '__main__':

    opt = TestOptions().parse()
    
    opt.status = 'test'
    opt.contain_dontcare_label = True
    opt.no_instance = True

    main(opt)