import torch
import torch.nn as nn

# roughly based on https://github.com/sharathadavanne/seld-net/blob/master/keras_model.py

def sample_params():
    return {
        'pool_sizes': [9, 8, 2],
        'conv_channels': 64,
        'dropout_rate': 0.0,
        'nb_cnn2d_filt': 64,
        'rnn_sizes': [128, 128],
        'fnn_sizes': [128],
    }
    
class SELDNetModel(nn.Module):
    def __init__(
        self, 
        input_size = (4, 355, 128),
        output_size = (355, 5),
        params = {},
        ):
        super().__init__()

        conv = []
        for pool_size in params['pool_sizes']:
            conv.append(nn.LazyConv2d(
                out_channels=params['nb_cnn2d_filt'],
                kernel_size=(3,3), stride=1, padding=1,
                ))
            conv.append(nn.BatchNorm2d(params['conv_channels']))
            conv.append(nn.ReLU(inplace=True))
            conv.append(nn.MaxPool2d(
                kernel_size=(1, pool_size), 
                stride=1,
                ceil_mode=True,
                ))
            conv.append(nn.Dropout(p=params['dropout_rate']))
        self.conv = nn.Sequential(*conv)

        rnn_sizes = []
        rnn_sizes.append(
            (input_size[-1] - sum(params['pool_sizes']) + len(params['pool_sizes'])
            ) * params['nb_cnn2d_filt'])
        rnn_sizes.extend(params['rnn_sizes'])
        
        rnn = []
        for i in range(len(rnn_sizes) - 1):
            rnn.append(torch.nn.GRU(
                input_size=rnn_sizes[i], hidden_size=rnn_sizes[i+1],
                batch_first=True, dropout=params['dropout_rate'], 
                bidirectional=True
                ))
        self.rnn = nn.ModuleList(rnn)
        self.tanh = nn.Tanh()
        
        fnn = []
        for i in range(len(params['fnn_sizes']) - 1):
            fnn.append(nn.Linear(params['fnn_sizes'][i], params['fnn_sizes'][i+1], bias=True))
            fnn.append(nn.Dropout(p=params['dropout_rate']))
        fnn.append(nn.Linear(params['fnn_sizes'][-1], output_size[1], bias=True))
        self.fnn = nn.Sequential(*fnn)

    def forward(self, x, vid_feat=None):
        B, C, T, F = x.shape
        x = self.conv(x)

        B, C, W, H = x.shape
        x = x.view(B, W, C*H)
        for rnn in self.rnn:
            (x, _) = rnn(x)
            x = self.tanh(x)
            x = x[:, :, x.shape[-1]//2:] * x[:, :, :x.shape[-1]//2]
            
        x = self.fnn(x)

        return x
    
if __name__ == "__main__":
    # uv run python -m aemodes.models.detection.seldnet
    import torchinfo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    input_size = (4, 355, 128)
    params = sample_params()
    model = SELDNetModel(params=params).to(device)
    torchinfo.summary(model, input_size=input_size)
    input_tensor = torch.randn(2, *input_size).to(device)
    with torch.no_grad(): output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)