import os
import argparse
import torch
import stage2.utils.utils as utils
from stage2.models.transformer_net import ImageTransfomer

def stylize_static_image(inference_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nst_model = ImageTransfomer.from_pretrained(inference_config["checkpoint"]).to(device)
    nst_model.eval()
    print("Stage 2 Model successfully loaded ...")

    # Stylize using trained model
    with torch.no_grad():
        content_img_path = os.path.join(inference_config['content'])
        content_image = utils.prepare_img(content_img_path, inference_config['img_width'], device)
        stylized_img = nst_model(content_image).to('cpu').numpy()[0]
        output_image_path = utils.save_and_maybe_display_image(inference_config, stylized_img)

    return output_image_path