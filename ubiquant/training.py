import torch
import numpy as np
import os
import pytorch_lightning as pl

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from data import UbiquantDataModule
from models import UbiquantMultiTask


def fit_model(split):

    torch.manual_seed(0)
    np.random.seed(0)

    data = UbiquantDataModule(split=split)

    model = UbiquantMultiTask(input_width=data.train_ds.features.shape[1])

    monitor='val_corr_resp'
    mode='max'
    test_name = 'final'

    early_stop_callback = EarlyStopping(
        monitor=monitor,
        mode=mode,
        patience=2,
        min_delta=0.0001,
        verbose=True
    )

    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        mode=mode,
        dirpath='./weights/'+test_name+'/',
        filename='multitask-split='+split+'-{epoch}-{'+monitor+':.3f}',
        save_top_k=1,
        verbose=True
    )

    trainer = pl.Trainer(   logger=pl_loggers.TensorBoardLogger('./logs/'+test_name+'/',name=split),
                            gpus=1,
                            max_epochs=18,
                            log_every_n_steps=10,
                            checkpoint_callback=True,
                            callbacks=[early_stop_callback,checkpoint_callback] )

    trainer.fit(model, data)


if __name__ == '__main__':

    fit_model('EVAL')

    # for i in range(5):
    #     fit_model('CV5'+str(i))



