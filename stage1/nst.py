import utils.image_utils as image_utils
import utils.model_utils
from utils.video_utils import create_viz_video

import torch
from torch.autograd import Variable
import os
import argparse
from optimizer.lbfgs import lbfgs_optimize

def neural_style_transfer(config):

    # handle nst output storage
    content_img_path = os.path.join(config['content_images_dir'], config['content'])
    style_img_path = os.path.join(config['style_images_dir'], config['style'])
    out_dir_name = 'combined_' + os.path.split(content_img_path)[1].split('.')[0] + '_' + os.path.split(style_img_path)[1].split('.')[0]
    result_path = os.path.join(config['output_img_dir'], out_dir_name)
    os.makedirs(result_path, exist_ok=True)

    # setup GPU for inference
    print("GPU Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # prepare inputs for nst
    content_img = image_utils.normalize_nst_input(content_img_path, config['height'], device)
    style_img = image_utils.normalize_nst_input(style_img_path, config['height'], device)

    # initialize with content image
    init_img = content_img
    starting_img = Variable(init_img, requires_grad=True)

    # prepare model for inference
    model, content_feature_maps_index_name, style_feature_maps_indices_names = utils.model_utils.prepare_nst_model(device)

    # extract feature maps for content and style image
    content_img_set_of_feature_maps = model(content_img)
    style_img_set_of_feature_maps = model(style_img)

    target_content_representation = content_img_set_of_feature_maps[content_feature_maps_index_name[0]].squeeze(axis=0)
    target_style_representation = [image_utils.generate_gram_matrix(x) for cnt, x in enumerate(style_img_set_of_feature_maps) if cnt in style_feature_maps_indices_names[0]]
    target_representations = [target_content_representation, target_style_representation]

    num_of_iterations = config['iterations']
    
    lbfgs_optimize(model, starting_img, target_representations, content_feature_maps_index_name, style_feature_maps_indices_names, config, result_path, num_of_iterations)

    return result_path


if __name__ == "__main__":
    # default args
    default_resource_dir = os.path.join(os.path.dirname(__file__), 'data')
    content_images_dir = os.path.join(default_resource_dir, 'content-images')
    style_images_dir = os.path.join(default_resource_dir, 'style-images')
    output_img_dir = os.path.join(default_resource_dir, 'output-images')
    img_format = (4, '.jpg')  # saves images in the format: %04d.jpg


    parser = argparse.ArgumentParser()
    parser.add_argument("--content", type=str, help="content image name", default='lion.jpg')
    parser.add_argument("--style", type=str, help="style image name", default='ben_giles.jpg')
    args = parser.parse_args()

    nst_config = dict()
    
    nst_config = {
        'height': 400,
        'content_weight': 100000.0,
        'style_weight': 30000.0,
        'tv_weight': 1.0,
        'optimizer': 'lbfgs',
        'iterations': 1000,
        'model': 'vgg19',
        'saving_freq': -1
    }
    for arg in vars(args):
        nst_config[arg] = getattr(args, arg)
    nst_config['content_images_dir'] = content_images_dir
    nst_config['style_images_dir'] = style_images_dir
    nst_config['output_img_dir'] = output_img_dir
    nst_config['img_format'] = img_format

    results_path = neural_style_transfer(nst_config)

    create_viz_video(results_path, img_format)
