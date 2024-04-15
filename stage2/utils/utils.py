import os
import cv2 as cv
import numpy as np
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, Sampler

IMAGENET_MEAN_1 = np.array([0.485, 0.456, 0.406])
IMAGENET_STD_1 = np.array([0.229, 0.224, 0.225])

def load_image(img_path, target_shape=None):

    img = cv.imread(img_path)[:, :, ::-1]  

    if target_shape is not None:  # resize section
        if isinstance(target_shape, int) and target_shape != -1:  
            current_height, current_width = img.shape[:2]
            new_width = target_shape
            new_height = int(current_height * (new_width / current_width))
            img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_CUBIC)
        else:  # set both dimensions to target shape
            img = cv.resize(img, (target_shape[1], target_shape[0]), interpolation=cv.INTER_CUBIC)

    img = img.astype(np.float32) 
    img /= 255.0  # get to [0, 1] range
    return img


def prepare_img(img_path, target_shape, device, batch_size=1, should_normalize=True):
    img = load_image(img_path, target_shape=target_shape)
    transform_list = [transforms.ToTensor()]
    if should_normalize:
        transform_list.append(transforms.Normalize(mean=IMAGENET_MEAN_1, std=IMAGENET_STD_1))
    transform = transforms.Compose(transform_list)
    img = transform(img).to(device)
    img = img.repeat(batch_size, 1, 1, 1)
    return img


def post_process_image(dump_img):
    mean = IMAGENET_MEAN_1.reshape(-1, 1, 1)
    std = IMAGENET_STD_1.reshape(-1, 1, 1)
    dump_img = (dump_img * std) + mean  # de-normalize
    dump_img = (np.clip(dump_img, 0., 1.) * 255).astype(np.uint8)
    dump_img = np.moveaxis(dump_img, 0, 2)
    return dump_img

def save_image(inference_config, dump_img):

    dump_img = post_process_image(dump_img)
    dump_dir = inference_config['output_images_path']
    dump_img_name = os.path.basename(inference_config['content']).split('.')[0] + '_model_' + inference_config['checkpoint'].split('.')[0] + '.jpg'
    cv.imwrite(os.path.join(dump_dir, dump_img_name), dump_img[:, :, ::-1]) 
    return str(os.path.join(dump_dir, dump_img_name))


def gram_matrix(x):
    (b, ch, h, w) = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)
    gram /= ch * h * w
    return gram


def get_training_config(training_config):
    num_of_datapoints = training_config['subset_size'] * training_config['num_of_epochs']
    training_metadata = {
        "content_weight": training_config['content_weight'],
        "style_weight": training_config['style_weight'],
        "num_of_datapoints": num_of_datapoints
    }
    return training_metadata
