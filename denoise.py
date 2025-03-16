import torch
import librosa
import soundfile as sf
import numpy as np
from train import NoiseReductionCNN

# Pré-processamento do áudio
def preprocess_audio(audio_path):
    # Converte áudio para MFCC - representação espectral do áudio
    audio, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return mfcc

# Reconstrução do áudio
def reconstruct_audio(mfcc, output_path):
    audio = librosa.feature.inverse.mfcc_to_audio(mfcc.squeeze())
    sf.write(output_path, audio, 16000)

# Remoção de ruído
def remove_noise(input_audio, output_audio):
    # Carregar modelo
    model = NoiseReductionCNN()
    model.load_state_dict(torch.load("denoise_model.pth"))
    model.eval()

    # Processa áudio
    mfcc = preprocess_audio(input_audio)
    mfcc_tensor = torch.tensor(mfcc).unsqueeze(0).unsqueeze(0)
    denoised_mfcc = model(mfcc_tensor).detach().numpy()
    denoised_mfcc = denoised_mfcc.squeeze()

    # Reconstrói áudio
    reconstruct_audio(denoised_mfcc, output_audio)
    print(f"Áudio processado e salvo em {output_audio}")

if __name__ == "__main__":
    remove_noise("test_noisy.wav", "test_clean.wav")