import os
import torch
import torch.nn as nn  # pacote redes neurais
import torch.optim as optim # pacote otimização
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader # pacote de carregamento dos datasets

#Dataset Audio
class NoiseDataset(Dataset):
    def __init__(self, noisy_files, clean_files, max_len=400):
        self.noisy_files = noisy_files
        self.clean_files = clean_files
        self.max_len = max_len

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        noisy, sr = librosa.load(self.noisy_files[idx], sr=16000)
        clean, _ = librosa.load(self.clean_files[idx], sr=16000)

        noisy_mfcc = librosa.feature.mfcc(y=noisy, sr=sr, n_mfcc=40)
        clean_mfcc = librosa.feature.mfcc(y=clean, sr=sr, n_mfcc=40)

        # Truncar ou preencher os tensores para o comprimento máximo
        noisy_mfcc = self._pad_or_truncate(noisy_mfcc, self.max_len)
        clean_mfcc = self._pad_or_truncate(clean_mfcc, self.max_len)

        return torch.tensor(noisy_mfcc).unsqueeze(0), torch.tensor(clean_mfcc).unsqueeze(0)

    def _pad_or_truncate(self, mfcc, max_len):
        if mfcc.shape[1] > max_len:
            return mfcc[:, :max_len]
        elif mfcc.shape[1] < max_len:
            pad_width = max_len - mfcc.shape[1]
            return np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            return mfcc

# Definindo o Modelo
class NoiseReductionCNN(nn.Module):
    def __init__(self):
        super(NoiseReductionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.conv6(x)  # Saída com ruído reduzido
        return x

# Treinamento do modelo
def train_model(model, train_loader, epochs=30):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Taxa de aprendizado ajustada

    for epoch in range(epochs):
        running_loss = 0.0
        for i, (noisy_audio, clean_audio) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(noisy_audio)
            loss = criterion(output, clean_audio)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Época {epoch+1} finalizada, Perda: {loss.item()}")

    torch.save(model.state_dict(), "denoise_model.pth")
    print("Modelo salvo como 'denoise_model.pth'")

# Executando treinamento
if __name__ == "__main__":
    noisy_files = [os.path.join("datasets/noisy", f) for f in os.listdir("datasets/noisy")]
    clean_files = [os.path.join("datasets/clean", f) for f in os.listdir("datasets/clean")]

    dataset = NoiseDataset(noisy_files, clean_files)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = NoiseReductionCNN()
    train_model(model, train_loader, epochs=30)