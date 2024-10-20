from pytorch_lightning import LightningModule
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch


class FlanT5FineTuner(LightningModule):
    def __init__(self, model_name="google/flan-t5-base", learning_rate=3e-5):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.learning_rate = learning_rate

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
        self.metric.add_batch(predictions=preds, references=labels)

        return val_loss

    def validation_epoch_end(self, outputs):
        # Calculate the accumulated accuracy
        accuracy = self.metric.compute()
        self.log("val_accuracy", accuracy)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)