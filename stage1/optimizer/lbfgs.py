
from torch.optim import LBFGS
import torch
from loss.nstloss import build_loss
import utils.image_utils as image_utils

def lbfgs_optimize(neural_net, optimizing_img, target_representations, content_feature_maps_index_name, style_feature_maps_indices_names, config, dump_path, num_of_iterations):
    optimizer = LBFGS((optimizing_img,), max_iter=num_of_iterations, line_search_fn='strong_wolfe')
    cnt = 0

    def closure():
        nonlocal cnt
        if torch.is_grad_enabled():
            optimizer.zero_grad()
        total_loss, content_loss, style_loss, tv_loss = build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], config)
        if total_loss.requires_grad:
            total_loss.backward()
        with torch.no_grad():
            print(f'iteration: {cnt:03}, total loss={total_loss}, content_loss={config["content_weight"] * content_loss}, style loss={config["style_weight"] * style_loss}, tv loss={config["tv_weight"] * tv_loss}')

        cnt += 1
        return total_loss

    optimizer.step(closure)
