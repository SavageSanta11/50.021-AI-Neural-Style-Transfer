import os
import argparse
import time
import torch
from torch.optim import Adam
from models.perceptual_loss_net import PerceptualLossNet
from models.transformer_net import ImageTransfomer
import utils.utils as utils
import utils.dataloader as dataloader_utils


def train(training_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # prepare data loader
    train_loader = dataloader_utils.load_data(training_config)

    perceptual_loss_net = PerceptualLossNet(requires_grad=False).to(device)
    # Calculate style image's Gram matrices (style representation)
    style_img_path = os.path.join(training_config['style_images_path'], training_config['style_img_name'])
    style_img = utils.prepare_img(style_img_path, target_shape=None, device=device, batch_size=training_config['batch_size'])
    style_img_set_of_feature_maps = perceptual_loss_net(style_img)
    target_style_representation = [utils.gram_matrix(x) for x in style_img_set_of_feature_maps]

    transformer_net = ImageTransfomer().train().to(device)
    optimizer = Adam(transformer_net.parameters())

    acc_content_loss, acc_style_loss = [0., 0.]
    for epoch in range(training_config['num_of_epochs']):
        for batch_id, (content_batch, _) in enumerate(train_loader):
            # step1: Feed content batch through transformer net
            content_batch = content_batch.to(device)
            stylized_batch = transformer_net(content_batch)

            # step2: Feed content and stylized batch through perceptual net (VGG16)
            content_batch_set_of_feature_maps = perceptual_loss_net(content_batch)
            stylized_batch_set_of_feature_maps = perceptual_loss_net(stylized_batch)

            # step3: Calculate content representations and content loss
            target_content_representation = content_batch_set_of_feature_maps.relu2_2
            current_content_representation = stylized_batch_set_of_feature_maps.relu2_2
            content_loss = training_config['content_weight'] * torch.nn.MSELoss(reduction='mean')(target_content_representation, current_content_representation)

            # step4: Calculate style representation and style loss
            style_loss = 0.0
            current_style_representation = [utils.gram_matrix(x) for x in stylized_batch_set_of_feature_maps]
            for gram_gt, gram_hat in zip(target_style_representation, current_style_representation):
                style_loss += torch.nn.MSELoss(reduction='mean')(gram_gt, gram_hat)
            style_loss /= len(target_style_representation)
            style_loss *= training_config['style_weight']

            total_loss = content_loss + style_loss 
            total_loss.backward()
            optimizer.step()

            optimizer.zero_grad()  # clear gradients for the next round

            acc_content_loss += content_loss.item()
            acc_style_loss += style_loss.item()

            print(f'epoch={epoch}|batch=[{batch_id}/{len(train_loader)}]|content-loss={acc_content_loss}|style-loss={acc_style_loss}')
            acc_content_loss, acc_style_loss = [0., 0.]

    training_state = utils.get_training_config(training_config)
    training_state["state_dict"] = transformer_net.state_dict()
    training_state["optimizer_state"] = optimizer.state_dict()
    style_name = training_config['style_img_name'].split('.')[0]
    content_wt = training_config["content_weight"]
    style_wt = training_config["style_weight"]
    model_name = f"style_{style_name}_cw_{str(content_wt)}_sw_{str(style_wt)}.pth"
    torch.save(training_state, os.path.join(training_config['checkpoint_dir'], model_name))


if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    dataset_path = os.path.join(current_dir, 'data', 'mscoco')
    style_images_path = os.path.join(parent_dir, 'data', 'style-images')
    checkpoint_dir = os.path.join(parent_dir, 'checkpoints', 'stage2')
    checkpoints_root_path = os.path.join(os.path.dirname(__file__), 'models', 'checkpoints')
    
    image_size = 256 
    batch_size = 4

    os.makedirs(checkpoint_dir, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--style_img_name", type=str, default='edtaonisl.jpg')
    parser.add_argument("--subset_size", type=int, default=None)
    args = parser.parse_args()

    checkpoints_path = os.path.join(checkpoints_root_path, args.style_img_name.split('.')[0])
    training_config = dict()
    training_config = {
        "content_weight": 1e0,
        "style_weight": 4e5,
        "num_of_epochs": 2,
        "console_log_freq": 2,
    }

    for arg in vars(args):
        training_config[arg] = getattr(args, arg)
    training_config['dataset_path'] = dataset_path
    training_config['style_images_path'] = style_images_path
    training_config['checkpoint_dir'] = checkpoint_dir
    training_config['checkpoints_path'] = checkpoints_path
    training_config['image_size'] = image_size
    training_config['batch_size'] = batch_size

    train(training_config)

