from datasets import Audio
from datasets import load_dataset
import torch

def get_dataset():
    dataset = load_dataset("audiofolder", data_dir="/kaggle/working")
    dataset_ = dataset['train'].train_test_split(test_size=0.05)
    dataset_ = dataset_.cast_column("audio", Audio(sampling_rate=16_000))

    labels = dataset_['train'].features['label'].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label
    return dataset_, label2id, id2label



