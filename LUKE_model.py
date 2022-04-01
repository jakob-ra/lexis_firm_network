from transformers import LukeTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import LukeForEntityPairClassification, AdamW
import pytorch_lightning as pl
from sklearn import metrics
import pandas as pd

class RelationExtractionDataset(Dataset):
    """Relation extraction dataset."""

    def __init__(self, data: pd.DataFrame, has_labels=True,
                 tokenizer={'name': 'studio-ousia/luke-base', 'task': 'entity_pair_classification'},
                 max_len=128):
        self.data = data
        self.has_labels = has_labels
        self.tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base", task="entity_pair_classification")
        self.max_len= max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]

        text = item.document

        encoding = self.tokenizer(text, entity_spans=item.spans, padding="max_length", truncation=True, return_tensors="pt",
                            max_length=self.max_len)

        for k,v in encoding.items():
          encoding[k] = encoding[k].squeeze()

        if self.has_labels:
            encoding["label"] = torch.tensor(item.rel_one_hot)

        return encoding

class LUKE(pl.LightningModule):

    def __init__(self, num_labels=None, lr=1e-5, batch_size=128, class_weights="equal",
                 thresholds=0.5, weight_decay=0,
                 datasets={'train_dataset': None, 'val_dataset': None, 'test_dataset': None}
                 ):
        super().__init__()
        self.model = LukeForEntityPairClassification.from_pretrained("studio-ousia/luke-base",
                                                                     num_labels=num_labels)
        if class_weights == "equal":
            class_weights = torch.tensor([1 for i in range(num_labels)])
        self.save_hyperparameters()
        print(self.hparams)

    def forward(self, input_ids, entity_ids, entity_position_ids, attention_mask, entity_attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, entity_ids=entity_ids,
                             entity_attention_mask=entity_attention_mask,
                             entity_position_ids=entity_position_ids)
        return outputs

    def common_step(self, batch, batch_idx):
        labels = batch['label'].float()
        del batch['label']
        outputs = self(**batch)
        logits = outputs.logits

        criterion = torch.nn.BCEWithLogitsLoss(
            weight=self.hparams.class_weights)  # multi-label classification with weighted classes
        loss = criterion(logits, labels)
        preds = (torch.sigmoid(logits) > self.hparams.thresholds).float()

        return {'loss': loss, 'preds': preds, 'labels': labels}

    def training_step(self, batch, batch_idx):
        output = self.common_step(batch, batch_idx)
        loss = output['loss']
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        output = self.common_step(batch, batch_idx)
        loss = output['loss']
        self.log("val_loss", loss, on_epoch=True)

        preds = output['preds']
        labels = output['labels']

        return {"loss": loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):
        for i, output in enumerate(outputs):
            preds = output["preds"].detach().cpu().numpy()
            labels = output["labels"].detach().cpu().numpy()
            loss = output["loss"].mean()

            f1_scores = []
            for idx, label_name in enumerate(rel_label_names):
                label = [label[idx] for label in labels]
                pred = [pred[idx] for pred in preds]

                precision = metrics.precision_score(label, pred, zero_division=0)
                recall = metrics.recall_score(label, pred, zero_division=0)
                f1 = metrics.f1_score(label, pred, zero_division=0)
                self.log(f'val_precision_{label_name}', precision)
                self.log(f'val_recall_{label_name}', recall)
                self.log(f'val_f1_{label_name}', f1, prog_bar=True)
                f1_scores.append(f1)
            self.log(f'val_f1_macro_avg', sum(f1_scores) / len(f1_scores), prog_bar=True)

    def test_step(self, batch, batch_idx):
        output = self.common_step(batch, batch_idx)
        loss = output['loss']
        self.log("test_loss", loss, on_epoch=True)

        preds = output['preds']
        labels = output['labels']

        return {"loss": loss, "preds": preds, "labels": labels}

    def test_epoch_end(self, outputs):
        for i, output in enumerate(outputs):
            preds = output["preds"].detach().cpu().numpy()
            labels = output["labels"].detach().cpu().numpy()
            loss = output["loss"].mean()

            f1_scores = []
            for idx, label_name in enumerate(rel_label_names):
                label = [label[idx] for label in labels]
                pred = [pred[idx] for pred in preds]
                precision = metrics.precision_score(label, pred, zero_division=0)
                recall = metrics.recall_score(label, pred, zero_division=0)
                f1 = metrics.f1_score(label, pred, zero_division=0)
                self.log(f'test_precision_{label_name}', precision)
                self.log(f'test_recall_{label_name}', recall)
                self.log(f'test_f1_{label_name}', f1, prog_bar=True)
                f1_scores.append(f1)
            self.log(f'test_f1_macro_avg', sum(f1_scores) / len(f1_scores), prog_bar=True)

    def predict_step(self, batch, batch_idx):
        output = self.common_step(batch, batch_idx)
        loss = output['loss']

        preds = output['preds']
        labels = output['labels']

        return {"loss": loss, "preds": preds, "labels": labels}

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay,
                          no_deprecation_warning=True)

        return optimizer

    def train_dataloader(self):
        return DataLoader(train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=16)

    def val_dataloader(self):
        return DataLoader(val_dataset, batch_size=self.hparams.batch_size, num_workers=16)

    def test_dataloader(self):
        return DataLoader(test_dataset, batch_size=self.hparams.batch_size, num_workers=16)


