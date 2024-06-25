import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, attention_size, num_layer, dropout_rate):
        super(BiLSTMWithAttention, self).__init__()

        # 双向LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layer,
                            bidirectional=True, batch_first=True,
                            dropout=dropout_rate)

        # 注意力机制的参数
        self.attention_W = nn.Linear(hidden_size * 2, attention_size)
        self.attention_U = nn.Linear(attention_size, 1, bias=False)

        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)

        # 全连接层
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # 添加一个通道维度
        # LSTM层
        lstm_out, _ = self.lstm(x)

        # Dropout
        lstm_out = self.dropout(lstm_out)

        # 注意力机制
        attention_score = torch.tanh(self.attention_W(lstm_out))
        attention_weight = F.softmax(self.attention_U(attention_score), dim=1)
        attention_output = torch.sum(attention_weight * lstm_out, dim=1)

        # 全连接层
        output = self.fc(attention_output)

        return output


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

if __name__ == "__main__":
    # 使用示例
    input_size = 100  # 输入数据维度，根据你的数据而定
    hidden_size = 64  # LSTM隐藏层维度
    num_classes = 2  # 类别数量，根据你的数据而定
    attention_size = 32  # 注意力机制的维度


    # 实例化模型
    model = BiLSTMWithAttention(input_size, hidden_size, num_classes, attention_size, 2, 0.5)

    # 打印模型结构
    print(model)
