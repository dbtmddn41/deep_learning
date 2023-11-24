import torch
from BEATs import BEATs, BEATsConfig
import torchmetrics
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as L

class CustomBEATs(L.LightningModule):
    def __init__(self, ckpt, num_classes):
        super(CustomBEATs, self).__init__()
        checkpoint = torch.load(ckpt)

        cfg = BEATsConfig(checkpoint['cfg'])
        BEATs_model = BEATs(cfg)
        BEATs_model.load_state_dict(checkpoint['model'])
        self.BEATs_model = BEATs_model
        for name, m in self.BEATs_model.named_parameters():
            if not (name.startswith("encoder.layers.10") or name.startswith("encoder.layers.11")):
                m.requires_grad = False
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(checkpoint['cfg']['encoder_embed_dim'], num_classes)
        
        self.f1_scorer = torchmetrics.classification.MulticlassF1Score(num_classes)
    
    def forward(self, inputs, padding_mask=None):
        x, padding_mask = self.BEATs_model.extract_features(inputs)
        x = self.dropout(x)
        logits = self.classifier(x)
        if padding_mask is not None and padding_mask.any():
                logits[padding_mask] = 0
                logits = logits.sum(dim=1)
                logits = logits / (~padding_mask).sum(dim=1).unsqueeze(-1).expand_as(logits)
        else:
            logits = logits.mean(dim=1)
            
        return logits, padding_mask
    
    def training_step(self, batch, batch_idx):
        self.train()
        (data, mask), label = batch
        outputs, mask  = self(data, mask)
        loss = F.cross_entropy(outputs, label)
        self.log('loss', loss, prog_bar=True, on_step=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        self.eval()
        (data, mask), label = batch
        outputs, mask = self(data, mask)
        _, preds = torch.max(outputs, 1)
        loss = F.cross_entropy(outputs, label)
        valid_f1 = self.f1_scorer(preds, label)
#         valid_f1 = f1_score(label.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro')
        metrics = {'val_loss': loss, 'val_f1': valid_f1}
        self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, 
                                                                 T_mult=1, eta_min=0.00001)
        return [optimizer], [scheduler]