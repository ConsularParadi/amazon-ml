from pytorch_lightning import LightningModule
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import torchmetrics


class FlanT5FineTuner(LightningModule):
    def __init__(self, model_name="google/flan-t5-base", learning_rate=1e-2):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.learning_rate = learning_rate

        self.num_classes = self.model.config.vocab_size
        self.metric = torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes, ignore_index=self.tokenizer.pad_token_id)

    def forward(self, input_ids, attention_mask, labels=None):        
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        outputs = self(input_ids, attention_mask, labels=labels)
        loss = outputs.loss
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], batch["attention_mask"], labels=batch["labels"])
        val_loss = outputs.loss

        # Log validation loss
        self.log("val_loss", val_loss, prog_bar=True)

        # For tracking accuracy or other metrics (optional)
        preds = torch.argmax(outputs.logits, dim=-1)
        labels = batch["labels"]
        self.metric.update(preds, labels)

        return val_loss

    def on_validation_epoch_end(self):
        # Calculate the accumulated accuracy
        accuracy = self.metric.compute()
        self.log("val_accuracy", accuracy)

        self.metric.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)