import os
import argparse
import torch
import stage2.utils.utils as utils
from stage2.models.transformer_net import ImageTransfomer

# def stylize_static_image(inference_config):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     nst_model = ImageTransfomer.from_pretrained(inference_config["checkpoint"]).to(device)
#     nst_model.eval()
#     print("Stage 2 Model successfully loaded ...")

#     # Stylize using trained model
#     with torch.no_grad():
#         content_img_path = os.path.join(inference_config['content'])
#         content_image = utils.prepare_img(content_img_path, inference_config['img_width'], device)
#         stylized_img = nst_model(content_image).to('cpu').numpy()[0]
    #     output_image_path = utils.save_image(inference_config, stylized_img)
    #     print(output_image_path)
    # return output_image_path

def stylize_static_image(inference_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nst_model = None
    if ".pth" not in inference_config["checkpoint"]:
        # load from hugging face
        nst_model = ImageTransfomer.from_pretrained(inference_config["checkpoint"]).to(device)
        inference_config["checkpoint"] = inference_config["checkpoint"].split("/")[1]
        nst_model.eval()
    else:
        nst_model = ImageTransfomer().to(device)
        training_state = torch.load(os.path.join(inference_config["checkpoint_dir"], inference_config["checkpoint"]))
        state_dict = training_state["state_dict"]
        nst_model.load_state_dict(state_dict, strict=True)
        nst_model.eval()

    with torch.no_grad():
        content_img_path = os.path.join(inference_config['content'])
        content_image = utils.prepare_img(content_img_path, inference_config['img_width'], device)
        stylized_img = nst_model(content_image).to('cpu').numpy()[0]
        output_image_path = utils.save_image(inference_config, stylized_img)
        print(output_image_path)
    return output_image_path