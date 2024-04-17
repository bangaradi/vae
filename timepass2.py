# import torch
# hyperparam_k = 0.1
# layer_name = 'fc1'

# def find_indices_to_drop(layer_name, hyperparam_k = 0.1):
#     model = torch.load('./run0/mnist/initial/ckpts/ckpt_modified.pt', map_location='cpu')

#     # find the L2 norm of each neuron and store it
#     l2_norm = []
#     for i in range(model['model'][layer_name + '.weight'].size()[0]):
#         l2_norm.append(torch.norm(model['model'][layer_name + '.weight'][i], 2))
#         # Add the contribution from layer_name.bias
#         l2_norm[-1] += torch.norm(model['model'][layer_name + ".bias"][i], 2)

#     # find the threshold i.e. the value of the bottom k% of L2 norm
#     threshold = torch.kthvalue(torch.tensor(l2_norm), int(hyperparam_k * len(l2_norm)))[0]
#     # print("The max L2 norm is: ", max(l2_norm))
#     # print("The min L2 norm is: ", min(l2_norm))
#     # print("The threshold is: ", threshold)

#     # find the indices of neurons with L2 norm less than threshold
#     indices = []
#     for i in range(len(l2_norm)):
#         if l2_norm[i] < threshold:
#             indices.append(i)
#     return indices

# find_indices_to_drop(layer_name, hyperparam_k)

# def add_weight_regularization(model):
#     # calculate the L1 norm of all the weights of the model
#     l1_norm_sum = 0
#     for layer_name in model['model']:
#         if 'weight' in layer_name or 'bias' in layer_name:
#             for i in range(model['model'][layer_name].size()[0]):
#                 l1_norm_sum += torch.norm(model['model'][layer_name][i], 1)
#     return l1_norm_sum

# print(add_weight_regularization(torch.load('./run0/mnist/initial/ckpts/ckpt_modified.pt', map_location='cpu')))

import torch
from utils import evaluate_with_classifier
from model import OneHotCVAE
from model import Classifier
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

# ckpt_path = "/home/stud-1/aditya/vae/results/mnist/2024_04_09_165401/ckpts"
ckpt_path = "/home/stud-1/aditya/vae/run0/mnist/initial/ckpts"
classifier_paths = ["/home/stud-1/aditya/vae/classifier_ckpts/model1.pt", "/home/stud-1/aditya/vae/classifier_ckpts/model2.pt", "/home/stud-1/aditya/vae/classifier_ckpts/model3.pt","/home/stud-1/aditya/vae/classifier_ckpts/model4.pt", "/home/stud-1/aditya/vae/classifier_ckpts/model5.pt"]

# model = torch.load(ckpt_path + "/ckpt_modified.pt", map_location='cuda')

# model = Classifier(output_dim=10)
# ckpt = torch.load(classifier_paths[0], map_location='cpu')
# model.load_state_dict(ckpt)
# model.eval()
models = []
for path in classifier_paths:
    model = Classifier(output_dim=10)
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt)
    model.eval()
    models.append(model)

majority_count = 0
total = 0

# load the MNIST dataset
test_dataset = datasets.MNIST("./dataset", train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

for i in tqdm(range(10000)):
    # noise_input = torch.randn(2, 1, 28, 28)
    image, label = next(iter(test_loader))
    total += 1
    preds = []
    for model in models:
        with torch.no_grad():
            output = model(image)
            preds.append(output.argmax().item())

    # check if all the predictions are the same
    if all(pred == preds[0] for pred in preds):
        majority_count += 1

print(majority_count, total)

    



# rem, forgot, ent = evaluate_with_classifier(ckpt_path, classifier_path, [1,2,3,4], [])

# print(rem, forgot, ent)

