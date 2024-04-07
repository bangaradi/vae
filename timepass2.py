import torch
hyperparam_k = 0.1
layer_name = 'fc1'

def find_indices_to_drop(layer_name, hyperparam_k = 0.1):
    model = torch.load('./run0/mnist/initial/ckpts/ckpt_modified.pt', map_location='cpu')

    # find the L2 norm of each neuron and store it
    l2_norm = []
    for i in range(model['model'][layer_name + '.weight'].size()[0]):
        l2_norm.append(torch.norm(model['model'][layer_name + '.weight'][i], 2))
        # Add the contribution from layer_name.bias
        l2_norm[-1] += torch.norm(model['model'][layer_name + ".bias"][i], 2)

    # find the threshold i.e. the value of the bottom k% of L2 norm
    threshold = torch.kthvalue(torch.tensor(l2_norm), int(hyperparam_k * len(l2_norm)))[0]
    # print("The max L2 norm is: ", max(l2_norm))
    # print("The min L2 norm is: ", min(l2_norm))
    # print("The threshold is: ", threshold)

    # find the indices of neurons with L2 norm less than threshold
    indices = []
    for i in range(len(l2_norm)):
        if l2_norm[i] < threshold:
            indices.append(i)
    return indices

find_indices_to_drop(layer_name, hyperparam_k)

def add_weight_regularization(model):
    # calculate the L1 norm of all the weights of the model
    l1_norm_sum = 0
    for layer_name in model['model']:
        if 'weight' in layer_name or 'bias' in layer_name:
            for i in range(model['model'][layer_name].size()[0]):
                l1_norm_sum += torch.norm(model['model'][layer_name][i], 1)
    return l1_norm_sum

print(add_weight_regularization(torch.load('./run0/mnist/initial/ckpts/ckpt_modified.pt', map_location='cpu')))
