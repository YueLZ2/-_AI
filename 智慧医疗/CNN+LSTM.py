import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# 模型定义
class CNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2, dropout_rate=0.5):
        super(CNN_LSTM, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout_rate)

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # 添加一个通道维度
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.permute(0, 2, 1)  # 调整形状为 (batch_size, sequence_length, input_size)

        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)

        lstm_out = lstm_out[:, -1, :]  # 只取最后一个时间步的输出

        output = self.fc(lstm_out)

        return output