import torch
import torchaudio
from torch.utils.data import Dataset
import torch.nn.functional as F

class BirdClsDataset(Dataset):
    def __init__(self, df, target_sample_rate, duration, masking_rate):
        self.audio_paths = df['filename'].values
        self.labels = df['label_encoded'].values
        self.target_sample_rate = target_sample_rate
        self.num_samples = target_sample_rate*duration
        self.masking_rate = masking_rate
        
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, index):
        audio_path = f'/kaggle/input/birdclef-2022/train_audio/{self.audio_paths[index]}'
        signal, sr = torchaudio.load(audio_path) # loaded the audio
        
        # Now we first checked if the sample rate is same as TARGET_SAMPLE_RATE and if it not equal we perform resampling
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        
        # Next we check the number of channels of the signal
        #signal -> (num_channels, num_samples) - Eg.-(2, 14000) -> (1, 14000)
        if signal.shape[0]>1:
            signal = torch.mean(signal, axis=0, keepdim=True)
        
        # Lastly we check the number of samples of the signal
        #signal -> (num_channels, num_samples) - Eg.-(1, 14000) -> (1, self.num_samples)
        # If it is more than the required number of samples, we truncate the signal
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        
        # If it is less than the required number of samples, we pad the signal
        if signal.shape[1]<self.num_samples:
            num_missing_samples = self.num_samples - signal.shape[1]
            last_dim_padding = (0, num_missing_samples)
            signal = F.pad(signal, last_dim_padding)

        signal = signal.squeeze(0)
        padding_mask = torch.rand(10000) < self.masking_rate
        label = torch.tensor(self.labels[index]).squeeze(dim=0).type(torch.LongTensor)
        
        return (signal, padding_mask), label