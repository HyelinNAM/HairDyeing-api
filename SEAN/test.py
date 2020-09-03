import os
from collections import OrderedDict
from itertools import cycle
from PIL import Image

from SEAN import data
from SEAN.options.test_options import TestOptions
from SEAN.models.pix2pix_model import Pix2PixModel
from SEAN.util.visualizer import Visualizer

from SEAN.data.base_dataset import get_params, get_transform

def preprocess(opt,image_path,label_path):

    # label
    label = Image.open(label_path)
    params = get_params(opt.preprocess_mode,label.size)
    transform_label = get_transform(opt, params, method=Image.NEAREST, normalize=False)
    label_tensor = transform_label(label) * 255.0
    label_tensor[label_tensor == 255] = opt.label_nc

    # image (Real)
    image = Image.open(image_path)
    image = image.convert('RGB')

    transform_image = get_transform(opt,params)
    image_tensor = transform_image(image)

    instance_tensor = 0

    return {'label':label_tensor, 'instance':instance_tensor, 'image':image_tensor, 'path':image_path}


def reconstruct(opt):

    model = Pix2PixModel(opt)
    model.eval()

    visualizer = Visualizer(opt)

    if opt.mode.lower() in ['black','brown','blond','red','blue']:

        src = preprocess(opt, image_path='', label_path='')

        generated = model(src, None,  mode=opt.mode) # image, segmap 붙여야?

        img_path = src['path']

        print('process image...')
        visuals = OrderedDict([('input_label', src['label']),
                            ('synthesized_image', generated)])

        visualizer.save_images(visuals, img_path,opt.results_dir,'results') #        


    elif opt.lower() == 'custom':
        
        src = preprocess(opt, image_path='', label_path='')
        ref = preprocess(opt, image_path='', label_path='')

        generated = model(src,ref, mode=opt.styling_mode)

    else:
        raise NotImplementedError



    for i, data_i in enumerate(zip(cycle(src_dataloader),oth_dataloader)):
        src_data = data_i[0]
        oth_data = data_i[1]
        generated = model(src_data,oth_data, mode=opt.styling_mode)

        img_path = src_data['path']

        for b in range(generated.shape[0]):
            print('process image... %s' % img_path[b])
            visuals = OrderedDict([('input_label', src_data['label'][b]),
                               ('synthesized_image', generated[b])])

            visualizer.save_images(visuals, img_path[b:b + 1],opt.results_dir,f'results_{i}')
    
        








    