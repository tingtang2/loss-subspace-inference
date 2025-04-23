# code from https://github.com/wjmaddox/drbayes/blob/master/subspace_inference/utils.py

import numpy as np


def unflatten_like(vector, likeTensorList):
    # Takes a flat torch.tensor and unflattens it to a list of torch.tensors
    #    shaped like likeTensorList
    outList = []
    i = 0
    for tensor in likeTensorList:
        #n = module._parameters[name].numel()
        n = tensor.numel()
        outList.append(vector[:, i:i + n].view(tensor.shape))
        i += n
    return outList


def extract_parameters(model):
    params = []
    for module in model.modules():
        for name in list(module._parameters.keys()):
            if module._parameters[name] is None:
                continue
            param = module._parameters[name]
            params.append((module, name, param.size()))
            module._parameters.pop(name)
    return params


def set_weights_old(params, w, device):
    offset = 0
    for module, name, shape in params:
        size = np.prod(shape)
        value = w[offset:offset + size]
        setattr(module, name, value.view(shape).to(device))
        offset += size
