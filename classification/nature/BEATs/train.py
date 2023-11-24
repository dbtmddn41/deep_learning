import pandas as pd
from sklearn.model_selection import train_test_split
from model import CustomBEATs
from dataset import BirdClsDataset
from torch.utils.data import DataLoader
import pytorch_lightning as L

class cfg:
    train_csv = 'asd.csv'
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
    train_df, val_df = get_df(cfg.train_csv)
    train_dataset = BirdClsDataset(train_df, cfg.sample_rate, cfg.duration)
    valid_dataset = BirdClsDataset(val_df, cfg.sample_rate, cfg.duration)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = CustomBEATs(cfg.ckpt, cfg.num_classes)

    trainer = L.Trainer(accelerator="auto", max_epochs=10, profiler="advanced")
    trainer.fit(model, train_loader, valid_loader)

if __name__ == '__main__':
    main()