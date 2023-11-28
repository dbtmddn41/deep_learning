import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import pytorch_lightning as L
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import os
import pickle
import pandas as pd
import numpy as np
import audiomentations
import time
import librosa
from tqdm import tqdm
from IPython.display import Audio, display
from matplotlib import pyplot as plt

class GeneralConfig:
    """
    현장에서 수정해야할 것
    
    filename -> csv file 오디오 파일 이름 column
    target_label -> csv file 분류하려는 오디오의 라벨 혹은 텍스트 column(인코딩 되어야 함)
    audio_folder -> 오디오 파일들 들어있는 폴더
    
    """
    filename = 'filename'
    target_label = 'label_encoded'
    audio_folder = '/kaggle/input/birdclef-2022/train_audio/'
    return_type = 'mspec' #raw_audio or mspec
    
class AudioDatasetConfig:
    """
    현장에서 수정해야할 것
    
    return_type -> audio파일 리턴할지, mel spectrogram 리턴할지. raw_audio or mspec
    
    """


    
class FeatureConfig:
    output_file = '/kaggle/working/feature_extracted_audio/'
    return_type = 'raw_audio' #raw_audio or mspec
    n_fft = 1024
    hop_length = 512
    n_mels = 64
    top_db = 80

def get_df(csv_file, label_column): 
    df = pd.read_csv(csv_file)
    X = df.drop(columns=[label_column])
    y = df[label_column]
    encoder = LabelEncoder()
    df['label_encoded'] = encoder.fit_transform(df[label_column])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    
    return df.iloc[X_train.index], df.iloc[X_test.index]


class AudioDataset(Dataset):
    def __init__(self, df, target_sample_rate, duration, transforms=None):
        self.audio_paths = df[GeneralConfig.filename].values
        self.labels = df['label_encoded'].values
        self.target_sample_rate = target_sample_rate
        self.num_samples = target_sample_rate*duration
        self.return_type = GeneralConfig.return_type
        self.transforms = transforms
        
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, index):
        
        """
            나중에 적당히 수정할 부분
        """
        ######################################################################
        ap = self.audio_paths[index]
        output_file = f'{FeatureConfig.output_file}{ap[:-4]}.pkl'
        data = FeatureLoader(output_file)#######################################################################

        
        
        
        if self.return_type == 'raw_audio':
            if self.transforms:
                audio = data[0].numpy()
                augmented_audio = self.transforms(audio, self.target_sample_rate)
                data[0] = augmented_audio
                
            return (data[0], data[1]), data[2]
        
        elif self.return_type == 'mspec':
            if self.transforms:
                spec = data[0].numpy()
                augmented_spec = self.transforms(spec)
                data[0] = augmented_spec
                
            return data[0], data[1]


def FeatureExtractor(df, target_sample_rate, duration, skip=False):
    
    if skip:
        print("Features are already extracted")
        return
    if not os.path.isdir(f'{FeatureConfig.output_file}'):
            os.mkdir(f'{FeatureConfig.output_file}')
    
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=target_sample_rate, 
                                                              n_fft=FeatureConfig.n_fft,
                                                              hop_length=FeatureConfig.hop_length,
                                                              n_mels=FeatureConfig.n_mels
                                                             )
    amtodb = torchaudio.transforms.AmplitudeToDB(top_db=FeatureConfig.top_db)
    df = df[[GeneralConfig.filename, GeneralConfig.target_label]]
    audio_paths = df[GeneralConfig.filename].values
    labels = df[GeneralConfig.target_label].values
    new_df = pd.DataFrame(columns=range(3))
    
    for index in tqdm(range(len(df))):
        
        """
            나중에 적당히 수정할 부분
        """
        #################################################################################
        ap = audio_paths[index]
        idx = ap.find('/')
        label = ap[:idx+1]
        wav_file = ap[idx+1:-4]
        if not os.path.isdir(f'{FeatureConfig.output_file}{label}'):
            os.mkdir(f'{FeatureConfig.output_file}{label}')
        output_file = f'{FeatureConfig.output_file}{label}{wav_file}.pkl'
        ###################################################################################
        
        audio_path = f'{GeneralConfig.audio_folder}{audio_paths[index]}'
        signal, sr = torchaudio.load(audio_path) # loaded the audio

        # Now we first checked if the sample rate is same as TARGET_SAMPLE_RATE and if it not equal we perform resampling
        if sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
            signal = resampler(signal)

        # Next we check the number of channels of the signal
        #signal -> (num_channels, num_samples) - Eg.-(2, 14000) -> (1, 14000)
        if signal.shape[0]>1:
            signal = torch.mean(signal, axis=0, keepdim=True)

        # Lastly we check the number of samples of the signal
        #signal -> (num_channels, num_samples) - Eg.-(1, 14000) -> (1, self.num_samples)
        # If it is more than the required number of samples, we truncate the signal
        num_samples = target_sample_rate * duration
        if signal.shape[1] > num_samples:
            signal = signal[:, :num_samples]
            padding_mask = [False for _ in range(num_samples)]

        # If it is less than the required number of samples, we pad the signal
        if signal.shape[1]<num_samples:
            padding_mask = [False for i in range(signal.shape[1])]
            num_missing_samples = num_samples - signal.shape[1]
            last_dim_padding = (0, num_missing_samples)
            signal = F.pad(signal, last_dim_padding)
            padding_mask += [True for _ in range(num_missing_samples)]

        signal = signal.squeeze(0)
        padding_mask = torch.tensor(padding_mask)
        label = torch.tensor(labels[index]).squeeze(dim=0).type(torch.LongTensor)
        
        with open(output_file, 'wb') as f:
            if GeneralConfig.return_type == 'raw_audio':
                pickle.dump([signal, padding_mask, label], f)
                #new_df.loc[index] = [signal, padding_mask, label]

            elif GeneralConfig.return_type == 'mspec':
                mel = mel_spectrogram(signal)
                mel = amtodb(mel)
                image = torch.cat([mel, mel, mel])
                max_val = torch.abs(image).max()
                image = image / max_val
                label = torch.tensor(labels[index])
                pickle.dump([image, label], f)
                #new_df.loc[index] = [image, label]

    #new_df.to_hdf(FeatureConfig.output_file, key='df', mode='w', complevel=3)
    #new_df.to_feather(FeatureConfig.output_file)

def FeatureLoader(pkl_file):
    df = pd.read_pickle(pkl_file, compression='infer')
    #df = pd.read_hdf(FeatureConfig.output_file, key='df', mode='r')
    #df = pd.read_feather(FeatureConfig.output_file)
    
    return df

def spec_augment(spec, T=40, F=15, time_mask_num=1, freq_mask_num=1):
    feat_size = spec.shape[0]
    seq_len = spec.shape[1]
   
    for _ in range(time_mask_num):
        t = np.random.uniform(low=0.0, high=T)
        t = int(t)
        t0 = np.random.randint(0, seq_len - t)
        spec[t0 : t0 + t] = 0
   
    for _ in range(freq_mask_num):
        f = np.random.uniform(low=0.0, high=F)
        f = int(f)
        f0 = np.random.randint(0, feat_size - f)
        spec[:, f0 : f0 + f] = 0
    return spec

class cfg:
    train_csv = '/kaggle/input/birdclef-2022/train_metadata.csv'
    ckpt = 'qwer.pt'
    num_classes = 152
    batch_size = 32
    sample_rate = 16000
    duration = 10

def main():
    
    """
        나중에 df 뒤에 슬라이싱 조절해야함
    """
    
    wav_transforms = audiomentations.OneOf(
    [
        # 랜덤한 노이즈 추가(치지직 거리는 소리)
        audiomentations.AddGaussianNoise(p=1),
        # 소음 비율 제어 가능한 랜덤 노이즈 추가. 
        #SNR은 소음 대 잡음 비율로, dB는 로그 스케일이니 아래 예시는 실제 음성-소음 차(dB)가 최소 5데시벨~40데시벨을 의미.
#         audiomentations.AddGaussianSNR(min_snr_db=5.0, max_snr_db=40.0, p=1.0),
        #다른 소리랑 섞는거. 이미지에서 믹스업 하던거랑 똑같은 기법. 단순 classfication쪽이면 이거 꽤 유용할 것 같다고 판단됨. 경로설정 필요. 아래 인버젼은 해당 파형을 거꾸로 해서 섞는다는 듯. optional. 
#         #audiomentations.AddBackgroundNoise(sounds_path="[/path/folder_with_sound_files]", min_snr_in_db=3.0 max_snr_in_db=30.0,noise_transform=PolarityInversion(),p=1.0)
#         #공기로 흡수되는 정도. 고주파일수록 많이 흡수됨. humid 높을수록 흡수율은 떨어짐. 들으면 소리가 좀 먹먹해짐.
#         audiomentations.AirAbsorption(min_distance=10.0,max_distance=50.0,p=1.0),
#         #디폴트 -4~4. 피치 조정
#         audiomentations.PitchShift(p=1),
        
#         #42-95 Hz,91-204 Hz,196-441 Hz,421-948 Hz,909-2045 Hz,1957-4404 Hz,4216-9486 Hz 7개의 밴드에서 min/max_gain_db 범위 안에서 주파수 뽑아다가 데시벨을 min/max_gain_db만큼 조정시키는 역할 하는 것으로 보임.
#         audiomentations.SevenBandParametricEQ(p=1),
#         #고주파 저주파 모두 감쇠 일으킴.(전화통화 소리처럼)
#         audiomentations.BandPassFilter(p=0.75),
#         #mel scale 상의 특정 대역폭 정지. min_center_freq :디폴트 200 max_center_freq : 디폴트 4000. min,max_rolloff:데시벨 얼마나 깎을건지. 6의 배수로 설정. 디폴트 12
#         audiomentations.BandStopFilter(p=0.75),
#         #일정 분위수 밖의 점들을 clipping. default : 0~40
#         audiomentations.ClippingDistortion(min_percentile_threshold=0, max_percentile_threshold=50,p=0.75),
#         #고역 주파수 필터링시킴.
#         audiomentations.HighPassFilter(p=0.75),
#         #일정 주파수 이상을 증폭시키거나 절단
#         audiomentations.HighShelfFilter(p=0.75),
#         #데시벨 리미트 거는 것 같음. min/max_threshold_db:디폴트 -24,-2 
#         audiomentations.Limiter(p=0.75),
#         #150~7500 사이 대역폭 소리 줄이는 필터인듯?
#         audiomentations.LowPassFilter(p=0.75),
        
#         audiomentations.LowShelfFilter(p=0.75),
        
    ])
    spec_transforms = spec_augment 
    
    ##############################################################################
    train_df, val_df = get_df(cfg.train_csv, 'primary_label')
    FeatureExtractor(train_df[:1000], cfg.sample_rate, cfg.duration, True)
    #FeatureExtractor(val_df[:1000], cfg.sample_rate, cfg.duration, True)
    train_dataset = AudioDataset(train_df[:1000], cfg.sample_rate, cfg.duration, spec_transforms)
    #valid_dataset = AudioDataset(val_df[:1000], cfg.sample_rate, cfg.duration)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    #valid_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    ###############################################################################
    
    print(train_dataset[0][0])
    plt.figure(figsize=(36, 12))
    librosa.display.specshow(train_dataset[0][0], sr=cfg.sample_rate, x_axis='time', y_axis='mel')
    plt.title('Mel power spectrogram ')
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout()
    
#     display(Audio(data=train_dataset[0][0][0], rate=cfg.sample_rate))
    

#     model = CustomBEATs(cfg.ckpt, cfg.num_classes)

#     trainer = L.Trainer(accelerator="auto", max_epochs=10, profiler="advanced")
#     trainer.fit(model, train_loader, valid_loader)
#     for batch in train_loader:
#         print(batch)
#         break
    
    
if __name__ == '__main__':
    main()
