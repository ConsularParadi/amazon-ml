import pandas as pd
from datasets import load_dataset
from transformers import T5Tokenizer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

from utils import batch_iterator
from preprocess import combine_cols, preprocess_data
from model import FlanT5FineTuner

train = pd.read_csv('train.csv')
train = train[:20000]

for batch_df in batch_iterator(train, batch_size=512):
    batch_df['model_input'] = batch_df.apply(combine_cols, axis=1)

train.to_csv('train_preprocessed.csv', index=False)

# Load your data
train_dataset = load_dataset('csv', data_files='train_preprocessed.csv')
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

# Apply preprocessing
train_dataset = train_dataset.map(lambda x: preprocess_data(x, tokenizer), batched=True)

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

train_size = 0.8  # 80% for training, 20% for validation
train_data, val_data = train_test_split(train_dataset, train_size=train_size)


# Convert the train and validation datasets to PyTorch Dataset if not already
class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

# Create DataLoader for both train and validation sets
train_dataloader = DataLoader(CustomDataset(train_data), batch_size=8, shuffle=True)
val_dataloader = DataLoader(CustomDataset(val_data), batch_size=8)

# Initialize your LightningModule
model = FlanT5FineTuner()

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",  # Save based on validation loss
    dirpath="./checkpoints",  # Directory to save the model checkpoints
    filename="flan-t5-{epoch:02d}-{val_loss:.2f}",  # Filename format
    save_top_k=3,  # Save top 3 models with lowest validation loss
    mode="min",  # Minimize validation loss
)

# Trainer
trainer = Trainer(
    max_epochs=4,
    gpus=1,  # Or use multiple GPUs with "gpus=-1"
    callbacks=[checkpoint_callback]
)

# Train
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

best_model_path = checkpoint_callback.best_model_path
model = FlanT5FineTuner.load_from_checkpoint(best_model_path)
