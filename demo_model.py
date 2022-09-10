import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CNN_LSTM(nn.Module):
    def __init__(self, seq_length, img_size, kernel_s_layer_1, kernel_s, max1, max2, max3, out_channels_1, out_channels_2, out_channels_final, LSTM_hidden, LSTM_layers, LSTM2_hidden, LSTM2_layers, p=0.0):
        super(CNN_LSTM, self).__init__()
        self.seq_length = seq_length
        self.max_1 = max1
        self.max_2 = max2
        self.max_3 = max3
        self.lstm_size = LSTM_hidden
        self.lstm2_size = LSTM2_hidden
        self.lstm_lay = LSTM_layers
        self.lstm2_lay = LSTM2_layers
        self.conv1 = nn.Conv2d(1, out_channels_1, kernel_size=kernel_s_layer_1)
        self.cnn_bn1 = nn.BatchNorm2d(out_channels_1)
        self.conv2 = nn.Conv2d(out_channels_1, out_channels_2, kernel_size=kernel_s)
        self.cnn_bn2 = nn.BatchNorm2d(out_channels_2)
        self.conv3 = nn.Conv2d(out_channels_2, out_channels_final, kernel_size=kernel_s)
        self.cnn_bn3 = nn.BatchNorm2d(out_channels_final)
        self.img_size = img_size
        size_after_cnn = int((int((int((img_size - kernel_s_layer_1 + 1) / self.max_1) - kernel_s + 1) / self.max_2) - kernel_s + 1) / self.max_3)
        self.out_size_cnn = out_channels_final*size_after_cnn**2
        self.LSTM = nn.LSTM(input_size=self.out_size_cnn, hidden_size=LSTM_hidden, num_layers=self.lstm_lay, dropout=p, batch_first=True)
        self.LSTM2 = nn.LSTM(input_size=1, hidden_size=self.lstm2_size, num_layers=self.lstm2_lay, dropout=p, batch_first=True)
        self.fc1 = nn.Linear(in_features=self.seq_length * (self.lstm_size + self.lstm2_size), out_features=2000)
        self.bn1 = nn.BatchNorm1d(num_features=2000)
        self.fc2 = nn.Linear(in_features=2000, out_features=2000)
        self.bn2 = nn.BatchNorm1d(num_features=2000)
        self.fc3 = nn.Linear(in_features=2000, out_features=750)
        self.bn3 = nn.BatchNorm1d(num_features=750)
        self.fc4 = nn.Linear(in_features=750, out_features=seq_length)

    def forward(self, t, cnts, seq_lens):

        cnts = cnts.unsqueeze(2)
        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        cnts = pack_padded_sequence(cnts, seq_lens, batch_first=True, enforce_sorted=False)
        cnts, (h_n, h_c) = self.LSTM2(cnts)
        # # Unpack
        cnts, unpacked_lens = pad_packed_sequence(cnts, batch_first=True)
        cnts = F.pad(cnts, (0, 0, 0, (self.seq_length - max(unpacked_lens)), 0, 0))

        batch_s, timesteps, C, H, W = t.size()
        t = t.view(batch_s * timesteps, 1, H, W)
        t = F.max_pool2d(F.leaky_relu(self.cnn_bn1(self.conv1(t))), kernel_size=self.max_1, stride=self.max_1)
        t = F.max_pool2d(F.leaky_relu(self.cnn_bn2(self.conv2(t))), kernel_size=self.max_2, stride=self.max_2)
        t = F.max_pool2d(F.leaky_relu(self.cnn_bn3(self.conv3(t))), kernel_size=self.max_3, stride=self.max_3)
        t = t.reshape(batch_s, self.seq_length, self.out_size_cnn)

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        t = pack_padded_sequence(t, seq_lens, batch_first=True, enforce_sorted=False)
        t, (h_n, h_c) = self.LSTM(t)
        t, unpacked_lens = pad_packed_sequence(t, batch_first=True)
        t = F.pad(t, (0, 0, 0, (self.seq_length - max(unpacked_lens)), 0, 0))
        t = t.reshape(batch_s, self.lstm_size*self.seq_length)
        cnts = cnts.reshape(batch_s, self.lstm2_size * self.seq_length)

        x = torch.cat((t, cnts), dim=1)
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = F.leaky_relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x
