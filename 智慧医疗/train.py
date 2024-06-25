import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from model import BiLSTMWithAttention, CNN_LSTM



class Trainer:
    def __init__(self, data_path, log_path, moder_w_path, labels_path='./DATA/smartmedicine/processed/labels.npy',
                 learning_rate=0.001, num_epochs=50, batch_size=32, patience=5):
        self.data_path = data_path
        self.moder_w_path = moder_w_path
        self.log_path = log_path
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.labels_path = labels_path

        self._load_data()
        self._prepare_data_loaders()
        self._initialize_model()
        self._prepare_log_file()

    def _load_data(self):
        self.datas = np.load(self.data_path)
        self.labels = np.load(self.labels_path)

        # Assuming labels are categorical and need to be encoded
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels)

    def _prepare_data_loaders(self):
        X_train, X_test, y_train, y_test = train_test_split(self.datas, self.labels, test_size=0.2, random_state=42)

        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                      torch.tensor(y_train, dtype=torch.long))
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # X_train, X_test, y_train, y_test = train_test_split(self.datas, self.labels, test_size=0.2, random_state=42)
        #
        # # Reshape data to (batch_size, sequence_length, input_size)
        # X_train = X_train[:, :, np.newaxis]
        # X_test = X_test[:, :, np.newaxis]
        #
        # train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
        #                               torch.tensor(y_train, dtype=torch.long))
        # test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
        #                              torch.tensor(y_test, dtype=torch.long))
        #
        # self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        # self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

    def _initialize_model(self):
        input_size = 100
        hidden_size = 32
        num_classes = 2
        attention_size = 16
        num_layers = 2
        dropout_rate = 0.5

        self.model = CNN_LSTM(input_size, hidden_size, num_classes)

        # self.model = BiLSTMWithAttention(input_size, hidden_size, num_classes, attention_size, num_layers, dropout_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _prepare_log_file(self):
        log_dir = os.path.dirname(self.log_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def train(self):
        best_accuracy = 0.0
        patience_counter = 0

        with open(self.log_path, 'w') as log_file:
            for epoch in range(self.num_epochs):
                self.model.train()
                for X_batch, y_batch in self.train_loader:
                    outputs = self.model(X_batch)
                    loss = self.criterion(outputs, y_batch)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # 验证模型
                self.model.eval()
                all_preds = []
                all_labels = []
                with torch.no_grad():
                    for X_batch, y_batch in self.test_loader:
                        outputs = self.model(X_batch)
                        _, predicted = torch.max(outputs, 1)
                        all_preds.extend(predicted.cpu().numpy())
                        all_labels.extend(y_batch.cpu().numpy())

                accuracy = accuracy_score(all_labels, all_preds)
                log_file.write(
                    f'Epoch {epoch + 1}/{self.num_epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}\n')
                print(f'Epoch {epoch + 1}/{self.num_epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

                # 早停机制
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    patience_counter = 0
                    torch.save(self.model.state_dict(), self.moder_w_path)  # 保存最佳模型
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    print("Early stopping triggered")
                    break

    def test(self):
        self.model.load_state_dict(torch.load(self.moder_w_path))
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                outputs = self.model(X_batch)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        print(f'Final Test Accuracy: {accuracy:.4f}')
        with open(self.log_path, 'a') as log_file:
            log_file.write(f'Final Test Accuracy: {accuracy:.4f}\n')
