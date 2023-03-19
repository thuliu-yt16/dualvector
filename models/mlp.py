import torch.nn as nn
import models

@models.register('mlp')
class MLP(nn.Module):
    def __init__(self, layers, bias=True, activate='leaky_relu', slope = 0.01, activate_last=False, **kwargs):
        super().__init__()

        self.layers = layers
        assert len(layers) > 1
        assert activate in ['leaky_relu', 'relu', 'sigmoid', 'tanh', 'gelu']
        self.in_dim = layers[0]
        self.out_dim = layers[-1]

        model = []
        for i in range(len(layers) - 1):
            model.append(nn.Linear(layers[i], layers[i + 1], bias=bias))
            if not activate_last and i < len(layers) - 2:
                if activate == 'leaky_relu':
                    model.append(nn.LeakyReLU(negative_slope=slope, inplace=True))
                elif activate == 'relu':
                    model.append(nn.ReLU(inplace=True))
                elif activate == 'sigmoid':
                    model.append(nn.Sigmoid())
                elif activate == 'tanh':
                    model.append(nn.Tanh())
                elif activate == 'gelu':
                    model.append(nn.GELU())
                else:
                    raise NotImplementedError('not implemented activation')
        
        self.model = nn.Sequential(*model)

        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        self.model.apply(init_weights)
    
    def forward(self, x):
        shape = x.shape[:-1]
        x = self.model(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)

    