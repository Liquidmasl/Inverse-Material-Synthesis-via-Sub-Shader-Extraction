import os

import pytorch_lightning as pl
import torch
import wandb
import yaml
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint


from src.data_loading import BlenderShaderDataModule

torch.set_float32_matmul_precision('medium')

class MyModel(pl.LightningModule):
    def __init__(self, input_size, num_classes, lr):
        super().__init__()
        self.lr = lr

        self.fc1 = nn.Linear(input_size, 2000)
        self.fc2 = nn.Linear(2000, 1500)
        self.fc3 = nn.Linear(1500, 1000)
        self.fc4 = nn.Linear(1000, 1000)
        self.fc5 = nn.Linear(1000, 1000)
        self.fc6 = nn.Linear(1000, 500)
        self.fc7 = nn.Linear(500, 100)
        self.fc8 = nn.Linear(100, 100)
        self.fc9 = nn.Linear(100, num_classes)
        self.apply(self.init_weights)

        self.validation_abs_errors = []

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            # He initialization for ReLU activation
            init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)

    def forward(self, x):
        """
        x here is the mnist images and we run it through fc1, fc2 that we created above.
        we also add a ReLU activation function in between and for that (since it has no parameters)
        I recommend using nn.functional (F)
        """
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = F.relu(x)
        x = self.fc6(x)
        x = F.relu(x)
        x = self.fc7(x)
        x = F.relu(x)
        x = self.fc8(x)
        x = F.relu(x)
        x = self.fc9(x)
        x = torch.sigmoid(x)
        return x

    def training_step(self, batch, batch_idx):
        data, targets = batch
        prediction = self(data)
        loss = F.mse_loss(prediction, targets)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, targets = batch
        prediction = self(data)
        loss = F.mse_loss(prediction, targets)
        abs_error = torch.abs(prediction - targets)
        self.validation_abs_errors.append(abs_error)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)


    def on_validation_epoch_end(self):

        all_abs_errors = torch.cat(self.validation_abs_errors, dim=0)
        # Calculate the mean across all batches
        mean_abs_error = all_abs_errors.mean()
        self.log('val_mean_accuracy', 1 - mean_abs_error, on_epoch=True, prog_bar=True, logger=True)

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self.validation_abs_errors = []


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer



def start_training_run():

    config_path = r'configs.yaml'
    configs = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    wandb_logger = WandbLogger(project='2024Lightning', save_code=True, log_model='all')
    wandb_logger.experiment.config.update(configs)
    wandb_logger.experiment.log_code(".")

    model = MyModel(input_size=configs['input_size'], num_classes=41, lr=configs['initial_lr'])

    run_folder = wandb_logger.experiment.dir

    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=configs['max_epochs'],
        accelerator='gpu',
        callbacks=[ModelCheckpoint(dirpath=os.path.join(run_folder, 'Models'), save_top_k=1, monitor='val_loss', mode='min')]
    )

    training_run_path = run_folder

    data_module = BlenderShaderDataModule(
        batch_size=configs['batch_size'],
        configs=configs,
        training_run_path=training_run_path,
        transform=None,
    )

    wandb.save(config_path)

    trainer.fit(model, data_module)



if __name__ == '__main__':

    start_training_run()

