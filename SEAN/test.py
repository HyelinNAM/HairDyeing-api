import os
import os.path as osp
from collections import OrderedDict
from itertools import cycle
from PIL import Image
import torch

from SEAN import data
from SEAN.options.test_options import TestOptions
from SEAN.models.pix2pix_model import Pix2PixModel
from SEAN.util.visualizer import Visualizer

from SEAN.data.base_dataset import get_params, get_transform

def preprocess(opt,image_path,label_path): ## __getitem__
    # label
    label = Image.open(label_path)
    params = get_params(opt,label.size)
    transform_label = get_transform(opt, params, method=Image.NEAREST, normalize=False)
    label_tensor = transform_label(label) * 255.0
    label_tensor[label_tensor == 255] = opt.label_nc

    # image (Real)
    for path in os.listdir(image_path):
        image = Image.open(osp.join(image_path, path))
    
    image = image.convert('RGB')

    transform_image = get_transform(opt,params)
    image_tensor = transform_image(image)

    instance_tensor = torch.Tensor([0])

    return {'label':label_tensor, 'instance':instance_tensor, 'image':image_tensor, 'path':image_path}

def reconstruct(opt):

    model = Pix2PixModel(opt)
    model.eval()

    visualizer = Visualizer(opt)

    if opt.mode.lower() in ['black','brown','blond','red','blue']:

        src = preprocess(opt, image_path='data/src', label_path='results/label/src.png')

        generated = model(src,None,opt) # image, segmap 붙여야

        img_path = src['path']

        print('process image...')
        visuals = OrderedDict([('input_label', src['label']),
                            ('synthesized_image', generated.squeeze(0))])

        visualizer.save_images(visuals, img_path,opt.results_dir,'results')   

    elif opt.mode.lower() == 'custom':
        
        src = preprocess(opt, image_path='data/src', label_path='results/label/src.png')
        ref = preprocess(opt, image_path='data/ref', label_path='results/label/ref.png')

        generated = model(src,ref,opt)

        img_path = src['path']

        print('process image...')
        visuals = OrderedDict([('input_label', src['label']),
                            ('synthesized_image', generated.squeeze(0))])

        visualizer.save_images(visuals, img_path,opt.results_dir,'results')

    elif opt.mode.lower() == 'save_color':

        ref = preprocess(opt, image_path='data/ref', label_path='results/label/ref.png')

        generated = model(None,ref,opt)




        








    