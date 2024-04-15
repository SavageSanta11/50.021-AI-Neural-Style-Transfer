import os
import argparse
import torch
import utils.utils as utils
from models.transformer_net import ImageTransfomer

def stylize_static_image(inference_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nst_model = None
    if ".pth" not in inference_config["checkpoint"]:
        # load from hugging face
        nst_model = ImageTransfomer.from_pretrained(inference_config["checkpoint"]).to(device)
        nst_model.eval()
    else:
        nst_model = ImageTransfomer().to(device)
        training_state = torch.load(os.path.join(inference_config["checkpoint_dir"], inference_config["checkpoint"]))
        state_dict = training_state["state_dict"]
        nst_model.load_state_dict(state_dict, strict=True)
        nst_model.eval()

    with torch.no_grad():
        content_img_path = os.path.join(inference_config['content_images_path'], inference_config['content'])
        content_image = utils.prepare_img(content_img_path, inference_config['img_width'], device)
        stylized_img = nst_model(content_image).to('cpu').numpy()[0]
        dump = utils.save_image(inference_config, stylized_img)
        print(dump)

if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    content_images_path = os.path.join(parent_dir, 'data', 'content-images')
    output_images_path = os.path.join(parent_dir, 'data', 'output-images', 'stage2')
    checkpoint_dir = os.path.join(parent_dir, 'checkpoints', 'stage2')

    os.makedirs(output_images_path, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--content", type=str, default='tubingen.png')
    parser.add_argument("--img_width", type=int, default=500)
    parser.add_argument("--checkpoint", type=str, default='mosaic.pth')

    args = parser.parse_args()

    inference_config = dict()
    for arg in vars(args):
        inference_config[arg] = getattr(args, arg)
    inference_config['content_images_path'] = content_images_path
    inference_config['output_images_path'] = output_images_path
    inference_config['checkpoint_dir'] = checkpoint_dir

    stylize_static_image(inference_config)
