from torch import nn


class MLPEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, num_hiddens):
        super(MLPEmbedding, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_hiddens = num_hiddens
        self.layers = []
        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, num_hiddens))
            elif i == num_layers - 1:
                self.layers.append(nn.Linear(num_hiddens, output_dim))
            else:
                self.layers.append(nn.Linear(num_hiddens, num_hiddens))
        self.layers = nn.ModuleList(self.layers)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x