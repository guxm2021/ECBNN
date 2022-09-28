import torch
import torch.nn as nn
import torch.nn.functional as F


def return_non_linearity(s):
    if s == "rectify":
        return nn.ReLU()
    return 0

class LinearEncoder(nn.Module):
    """
    Linear module is adopted to process the feature maps after CNN
    """
    def __init__(self, input_size, shapes, nonlinearities, residual=False):
        super(LinearEncoder, self).__init__()
        self.residual = residual
        self.layers_list = nn.ModuleList([])
        self.layers_list.append(nn.Linear(input_size, shapes[0]))
        temp_activation = return_non_linearity(nonlinearities[0])
        if temp_activation != 0:
            self.layers_list.append(temp_activation)

        for i in range(1, len(shapes)):
            self.layers_list.append(nn.Linear(shapes[i-1], shapes[i]))
            temp_activation = return_non_linearity(nonlinearities[i])
            if temp_activation != 0:
                self.layers_list.append(temp_activation)

    def forward(self, x):
        # x: (batch, seq_len, cnn_output_size)
        res_flag = 0
        for l in self.layers_list:
            if self.residual and type(l) == torch.nn.modules.linear.Linear:
                if l.__dict__['_parameters']['weight'].shape[0] == l.__dict__['_parameters']['weight'].shape[1]:
                    res_flag = 1
                    res = x
            x = l(x)
            if self.residual:
                if res_flag == 1:
                    x = x + res
                    res_flag = 0
        return x


class Conv3dEncoder(nn.Module):
    """
    Conv3d module is adopted to process video
    """
    def __init__(self, channels, dropout_p=0.5):
        super(Conv3dEncoder, self).__init__()
        self.conv1 = nn.Conv3d(1, channels[0], (3, 3, 3), (1, 2, 2), (1, 2, 2))
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv2 = nn.Conv3d(channels[0], channels[1], (3, 3, 3), (1, 1, 1), (1, 2, 2))
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv3 = nn.Conv3d(channels[1], channels[2], (3, 3, 3), (1, 1, 1), (1, 1, 1))
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        self.dropout_p = dropout_p

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.dropout_p)
        self.dropout3d = nn.Dropout3d(self.dropout_p)

    def forward(self, x):
        # x: (batch, channel, seqlen, h, w)
        x = self.conv1(x)           # (44, 50) -> (23, 26)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool1(x)           # (23, 26) -> (11, 13)

        x = self.conv2(x)           # (11, 13) -> (13, 14)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool2(x)           # (13, 14) -> (6, 7)

        x = self.conv3(x)           # (6, 7) -> (6, 7)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool3(x)           # (6, 7) -> (3, 3)

        # (B, C, T, H, W)->(T, B, C, H, W)
        x = x.permute(2, 0, 1, 3, 4).contiguous()
        # (B, C, T, H, W)->(T, B, C*H*W)
        x = x.view(x.size(0), x.size(1), -1)
        x = x.permute(1, 0, 2).contiguous()
        return x


class DeltaEncoder(nn.Module):
    """
    Adapted from Petridis et al. End-to-End Multi-View Lipreading. (BMCV)
    """
    def __init__(self, device, window):
        super(DeltaEncoder, self).__init__()
        self.device = device
        self.window = window

    def forward(self, inp):
        # inp (shape): (batch, seq_len, features_size)

        x = inp
        # utils.signals.append_delta_coeff
        y = torch.cat((x[:, 0, :].reshape(x.shape[0], 1, -1).repeat(1, self.window, 1),
                       x, x[:, -1, :].reshape(x.shape[0], 1, -1).repeat(1, self.window, 1)), 1)
        z = torch.zeros(x.size()).to(self.device)

        for i in range(0, x.shape[-2]):
            for j in range(1, self.window+1):
                z[:, i, :] += (y[:, i+self.window+j, :] -
                               y[:, i+self.window-j, :])/(2*j)

        delta = z

        x = delta
        y = torch.cat((x[:, 0, :].reshape(x.shape[0], 1, -1).repeat(1, self.window, 1),
                       x, x[:, -1, :].reshape(x.shape[0], 1, -1).repeat(1, self.window, 1)), 1)
        z = torch.zeros(x.size()).to(self.device)

        for i in range(0, x.shape[-2]):
            for j in range(1, self.window+1):
                z[:, i, :] += (y[:, i+self.window+j, :] -
                               y[:, i+self.window-j, :])/(2*j)

        double_delta = z
        # return shape (batch, seq_len, features_size*3)
        return torch.cat((inp, delta, double_delta), 2)


class LipNet(nn.Module):
    """
    Conv3D + Linear + Delta + BLSTM as encoder
    Linear as classifier
    """
    def __init__(self, opt):
        super(LipNet, self).__init__()
        self.device = opt.device
        # model architecture for Conv3D
        self.cnn_channels = opt.cnn_channels
        self.cnn_encoder = Conv3dEncoder(self.cnn_channels)
         
        # model architecture for Linear
        self.cnn_output_size = opt.cnn_output_size
        self.shapes = opt.dnn_shapes
        self.nonlinearities = opt.nonlinearities
        self.linear_encoder = LinearEncoder(self.cnn_output_size, self.shapes, self.nonlinearities, residual=False)
        
        # model architecture for Delta
        self.window = opt.window
        self.delta_encoder = DeltaEncoder(self.device, self.window)
        
        # model architecture for LSTM
        self.nz = opt.nz                       # size for hidden feature (domain adaptation)
        self.lstm_input_size = self.shapes[-1]
        self.nc = opt.nc

        self.blstm_encoder = nn.LSTM(
            input_size=self.lstm_input_size*3,
            hidden_size=self.nz,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.nz, self.nc), 
            nn.Softmax(dim=-1))

    def init_hidden(self, batch_size, num_layers=1, directions=1):  # blstm so 2 direction
        # h_0=(num_layers * num_directions, batch, hidden_size)
        # c_0 of shape (num_layers * num_directions, batch, hidden_size)

        h_0 = torch.randn(num_layers*directions, batch_size, self.nz)
        c_0 = torch.randn(num_layers*directions, batch_size, self.nz)

        h_0 = h_0.to(self.device)
        c_0 = c_0.to(self.device)

        return h_0, c_0

    def forward(self, x, hidden):
        # x: (batch, 1, seq_len, h, w)
        batch = x.shape[0]
        
        # initialize the hidden state for LSTM
        if hidden == None:
            h_0, c_0 = self.init_hidden(batch)
        else:
            h_0, c_0 = hidden

        # cnn encode 
        x = self.cnn_encoder(x)      # (batch, seq_len, cnn_output_size)

        # linear encode
        x = self.linear_encoder(x)   # (batch, seq_len, lstm_input_size)

        # delta encode
        x = self.delta_encoder(x)    # (batch, seq_len, lstm_input_size * 3)
        
        # z is the feature for domain adaptation
        z, (h_n, c_n) = self.blstm_encoder(x, (h_0, c_0))      # (batch, seq_len, nz)

        x = self.classifier(z)       # output: softmax
        
        return x, z, (h_n, c_n)


if __name__ == "__main__":
    model=Conv3dEncoder([32, 64, 96])
    a = torch.randn(3, 1, 10, 44, 50)
    b = model(a)
    print(b.shape)