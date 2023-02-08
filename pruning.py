import torch
import torch.nn as nn
import copy
from sklearn.cluster import KMeans


def cluster_conv_net(model, clust_dict, device):
    model_copy = copy.deepcopy(model)
    k = 0
    for layer in model_copy.named_modules():
        if isinstance(layer[1], nn.Conv2d):
            k += 1
            weights = layer[1].weight.detach().cpu()
            size = weights.shape
            weights = weights.flatten(1)
            kmeans = KMeans(n_clusters=clust_dict[k], random_state=42).fit(weights)
            centers = kmeans.cluster_centers_
            new_weights = torch.tensor(centers[kmeans.fit_predict(weights)])
            new_weights = torch.reshape(new_weights, size)
            new_weights.to(device)
            layer[1].weight = nn.Parameter(new_weights.float())
    # Return updated model
    return model_copy