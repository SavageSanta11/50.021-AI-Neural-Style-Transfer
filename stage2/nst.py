import os
import argparse
import torch
import utils.utils as utils
from models.transformer_net import ImageTransfomer


def stylize_static_image(inference_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load trained model
    nst_model = ImageTransfomer().to(device)
    print(os.path.join(inference_config["model_binaries_path"], inference_config["checkpoint_name"]))
    training_state = torch.load(os.path.join(inference_config["model_binaries_path"], inference_config["checkpoint_name"]))
    print(training_state.keys())
    state_dict = training_state["state_dict"]
    nst_model.load_state_dict(state_dict, strict=True)
    nst_model.eval()

    #stylize using trained model
    with torch.no_grad():
        content_img_path = os.path.join(inference_config['content_images_path'], inference_config['content'])
        content_image = utils.prepare_img(content_img_path, inference_config['img_width'], device)
        stylized_img = nst_model(content_image).to('cpu').numpy()[0]
        utils.save_and_maybe_display_image(inference_config, stylized_img, should_display=inference_config['should_not_display'])


if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    content_images_path = os.path.join(parent_dir, 'data', 'content-images')
    output_images_path = os.path.join(parent_dir, 'data', 'output-images', 'stage2')
    model_binaries_path = os.path.join(parent_dir, 'checkpoints', 'stage2')

    os.makedirs(output_images_path, exist_ok=True)

    parser = argparse.ArgumentParser()
    # Put image name or directory containing images (if you'd like to do a batch stylization on all those images)
    parser.add_argument("--content", type=str, help="Content image(s) to stylize", default='taj_mahal.jpg')
    parser.add_argument("--img_width", type=int, help="Resize content image to this width", default=500)
    parser.add_argument("--checkpoint_name", type=str, help="Model binary to use for stylization", default='giger_crop.pth')

    parser.add_argument("--should_not_display", action='store_false', help="Should display the stylized result")
    parser.add_argument("--redirected_output", type=str, help="Overwrite default output dir. Useful when this project is used as a submodule", default=None)
    args = parser.parse_args()

    if os.path.isdir(args.content) and args.redirected_output is None:
        args.redirected_output = output_images_path

    # Wrapping inference configuration into a dictionary
    inference_config = dict()
    for arg in vars(args):
        inference_config[arg] = getattr(args, arg)
    inference_config['content_images_path'] = content_images_path
    inference_config['output_images_path'] = output_images_path
    inference_config['model_binaries_path'] = model_binaries_path

    stylize_static_image(inference_config)
