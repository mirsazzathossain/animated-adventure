import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torchmetrics.functional import accuracy
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import OneCycleLR


def create_model():
    """
    Modify the pre-existing Resnet architecture for CIFAR-10 images (32x32).

    Returns:
        model: Torchvision model
    """
    model = models.resnet18(pretrained=False, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    return model



class LitResnet(LightningModule):
    """
    Lightning Module to train the Resnet model.

    Functions:
        forward: Forward pass of the model.
        training_step: Training step of the model.
        validation_step: Validation step of the model.
        test_step: Test step of the model.
        configure_optimizers: Configure the optimizers for the model.
        evaluate: Evaluate the model on the test set.
    """
    def __init__(self, lr = 0.05, **kwargs):
        """
        Initialize the Lightning Module.
            
        Args:
            lr: Learning rate for the model. Default: 0.05
        """
        super().__init__()

        self.save_hyperparameters()
        self.model = create_model()
        self.momentum = kwargs['momentum']
        self.weight_decay = kwargs['weight_decay']
        self.batch_size = kwargs['batch_size']
        self.max_lr=kwargs['max_lr']

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x: Input data.
        
        Returns:
            x: Output data.
        """
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        """
        Training step of the model.

        Args:
            batch: Input data.
            batch_idx: Batch index.

        Returns:
            loss: Loss of the model.
        """
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def evaluate(self, batch, stage=None):
        """
        Evaluate the model on the test set.

        Args:
            batch: Input data.
            stage: Stage of the model.
        """
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f'{stage}_loss', loss, prog_bar=True)
            self.log(f'{stage}_acc', acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        """
        Validation step of the model.

        Args:
            batch: Input data.
            batch_idx: Batch index.
        """
        self.evaluate(batch, stage='val')

    def test_step(self, batch, batch_idx):
        """
        Test step of the model.

        Args:
            batch: Input data.
            batch_idx: Batch index.
        """
        self.evaluate(batch, stage='test')

    def configure_optimizers(self):
        """
        Configure the optimizers for the model.

        Args:
            **kwargs: Keyword arguments for the optimizer.

        Returns:
            optimizers: Dictionary of optimizers.
        """
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        step_per_epoch = self.batch_size
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                max_lr=self.max_lr,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=step_per_epoch,
            ),
            "interval": "step"
        }

        return {'optimizer': optimizer, 'scheduler': scheduler_dict}
        
