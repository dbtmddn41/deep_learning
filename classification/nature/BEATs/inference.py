import pandas as pd
from sklearn.model_selection import train_test_split
from model import CustomBEATs
from dataset import BirdClsDataset
from torch.utils.data import DataLoader
import pytorch_lightning as L

class cfg:
    test_csv = 'asd.csv'
    ckpt = 'qwer.pt'
    num_classes = 152
    batch_size = 32


def preprocess_df(df):
    pass

def get_df(csv_file):
    df = pd.read_csv(csv_file)
    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)
    return df.iloc[X_train.index], df.iloc[X_test.index]

def main():
    test_df = pd.read_csv(cfg.test_csv)
    test_dataset = BirdClsDataset(test_df, cfg.sample_rate, cfg.duration)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = CustomBEATs(cfg.ckpt, cfg.num_classes)

    trainer = L.Trainer(accelerator="auto", profiler="advanced")
    trainer.predict(model, test_loader)

if __name__ == '__main__':
    main()