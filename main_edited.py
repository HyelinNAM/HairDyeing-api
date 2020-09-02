import torch
import argparse
from Face_parsing.test import parsing
from SEAN.options.test_options import TestOptions
from SEAN.test import reconstruct

def main(args):

    print(args)
    torch.manual_seed(args.seed)

    # 현재 결과물 폴더에 있는 것들 모두 삭제 단계 추가?
    # 혹은 사진 이름 일괄 맞춰서 실행?

    parsing(respth='./results/label', dspth='./data/src/src') 
    # parsing src_image (StarGAN 없이도 src/src 이중 처리해야하는지 체크)
    parsing(respth='./results/label', dspth='./data/src/src')
    # parsing dyeing_target_image
    
    


    '''
    if args.mode =='reference':

        parsing(respth='./results/label/src', dspth='./data/src/src') # parsing src_image
        parsing(respth='./results/label/')


        parsing(respth='./results/label/src' ,dspth='./data/src/src') # parsing src_image
        parsing(respth='./results/label/others', dspth='./data/dyeing') # parsing ref_image
        reconstruct(args.mode)
        
    elif args.mode =='choice_color':
        reconsturct(args.mode, args.)

    else:
        raise NotImplementedError
    '''

if __name__ == '__main__':

    # SEAN의 argparse 끌어와서? 
    opt = TestOptions().parse()
    
    opt.status = 'test'
    opt.contain_dontcare_label = True
    opt.no_instance = True    

    # implement
    opt.add_argument('--mode', type=str, required=True,
                        choices=['black','brown','blond','red','blue','custom'], help='set mode(color)')
    opt.add_argument('--seed', type=int, default=777,
                        help='Seed for random number generator')