import pytorch_lightning as pl
import torch
from torch import nn


class WeatherLSTM(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        # Saves the Hydra config to the Lightning module for easy access
        self.save_hyperparameters(logger=False)
        self.cfg = cfg

        # Core LSTM Architecture
        self.lstm = nn.LSTM(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            batch_first=True
        )
        # Regressor to map the hidden state back to the 3 target variables
        self.regressor = nn.Linear(cfg.hidden_size, cfg.input_size)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        # We only care about the prediction at the final time step of the sequence
        predictions = self.regressor(lstm_out[:, -1, :])
        return predictions

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        # Log the loss directly to MLflow via Lightning's logger
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.cfg.learning_rate)
