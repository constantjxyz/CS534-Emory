import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim=25, sequence_length=20, hidden_size=128, num_layers=2, num_labels=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.3).double()
        self.fc = nn.Linear(hidden_size, num_labels).double()
        self.sequence_length = sequence_length
    def forward(self, x):
        # x = torch.DoubleTensor(x)
        lstm_out, _ = self.lstm(x[:, :self.sequence_length, :])
        output = self.fc(lstm_out[:, -1, :])
        return output
    
class RNNModel(nn.Module):
    def __init__(self, input_dim=25, sequence_length=20, hidden_size=128, num_layers=2, num_labels=2):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.3).double()
        self.fc = nn.Linear(hidden_size, num_labels).double()
        self.sequence_length = sequence_length
    def forward(self, x):
        # x = torch.DoubleTensor(x)
        rnn_out, _ = self.rnn(x[:, :self.sequence_length, :])
        output = self.fc(rnn_out[:, -1, :])
        return output