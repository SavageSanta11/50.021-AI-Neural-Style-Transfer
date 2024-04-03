import torch
import utils.image_utils as image_utils

def build_loss(model, image_input, target_representations, content_feature_maps_index, style_feature_maps_indices, config):
    alpha = config['content_weight']
    beta = config['style_weight']
    gamma = config['tv_weight']

    target_content_representation = target_representations[0]
    target_style_representation = target_representations[1]

    generated_feature_maps = model(image_input)

    # content loss calculation
    current_content_representation = generated_feature_maps[content_feature_maps_index].squeeze(axis=0)
    content_loss = torch.nn.MSELoss(reduction='mean')(target_content_representation, current_content_representation)
    
    # style loss calculation
    style_loss = 0.0
    current_style_representation = []
    for cnt, x in enumerate(generated_feature_maps):
        if cnt in style_feature_maps_indices: ## only takes gram matrices from the layers we requested in the config
            required_gram_matrix = image_utils.generate_gram_matrix(x)
            current_style_representation.append(required_gram_matrix)

    for target_gram_matrix, current_gram_matrix in zip(target_style_representation, current_style_representation):
        style_loss += torch.nn.MSELoss(reduction='sum')(target_gram_matrix[0], current_gram_matrix[0])
    average_style_loss = style_loss/len(target_style_representation)

    # regulatization
    tv_loss = image_utils.total_variation(image_input)

    # nst loss calculation
    total_loss = alpha * content_loss + beta * average_style_loss + gamma * tv_loss

    return total_loss, content_loss, style_loss, tv_loss
