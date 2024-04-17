import torch
import torch.nn as nn
import torch.nn.functional as F
    

class OneHotCVAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim, class_size=10):
        super(OneHotCVAE, self).__init__()
        
        # encoder part
        self.fc1 = nn.Linear(x_dim + class_size, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim + class_size, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)
        
    def encoder(self, x, c):
        inputs = torch.cat([x,c], dim=1)
        h = F.relu(self.fc1(inputs))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h) # mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z, c):
        inputs = torch.cat([z,c], dim=1)
        h = F.relu(self.fc4(inputs))
        h = F.relu(self.fc5(h))
        return torch.sigmoid(self.fc6(h)) 
    
    def forward(self, x, c):
        mu, log_var = self.encoder(x.view(-1, 784), c)
        z = self.sampling(mu, log_var)
        return self.decoder(z, c), mu, log_var
    

class Classifier(nn.Module):
    def __init__(self, output_dim=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(20)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, output_dim) # changed from 10 to 11 for additional noise label

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.bn1(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.bn2(self.conv2(x))), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
class SelectiveDropout(nn.Module):
    def __init__(self, dropout_rate, neuron_indices):
        super(SelectiveDropout, self).__init__()
        self.dropout_rate = dropout_rate
        self.neuron_indices = neuron_indices

    def forward(self, inputs):
        if self.training and len(self.neuron_indices) != 0 and self.dropout_rate != 0:
            # print("inside training mode ")
            # print("inputs ", inputs)
            mask = torch.ones_like(inputs, dtype=torch.float32).to(inputs.device)
            neuron_indices = torch.ones(inputs.shape[1], dtype=torch.float32).to(inputs.device)
            # make the neuron_indices at positions self.neuron_indices to 0 with probability self.dropout_rate
            for i in self.neuron_indices:
                # set neuron_indices[i] to 0 with probability self.dropout_rate
                if torch.rand(1) < self.dropout_rate:
                    neuron_indices[i] = 0
                else :
                    neuron_indices[i] = 1 * 1/(1 - self.dropout_rate)
            # multiply the mask which is an array of arrays with the neuron_indices
            # print("neuron_indices ", neuron_indices)
            mask = mask * neuron_indices
            # print("mask ", mask)
            # print("returning ", inputs * mask)
            return inputs * mask
        else:
            return inputs

def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

def add_weight_regularization(model_state_dict):
    l1_norm_sum = 0
    for layer_name in model_state_dict:
        if 'weight' in layer_name or 'bias' in layer_name:
            for i in range(model_state_dict[layer_name].size()[0]):
                l1_norm_sum += torch.norm(model_state_dict[layer_name][i], 1)
    return l1_norm_sum
