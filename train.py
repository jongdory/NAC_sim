import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from ldm.data.breast import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint

import warnings
warnings.filterwarnings("ignore")

def load_state_dict_with_shape(model, checkpoint_path):
    saved_state_dict = load_state_dict(resume_path, location='cpu')
    model_state_dict = model.state_dict()

    for name, param in saved_state_dict.items():
        if name in model_state_dict:
            if param.shape == model_state_dict[name].shape:
                model_state_dict[name].copy_(param)

    model.load_state_dict(model_state_dict, strict=False)

# Configs
resume_path = './models/control_v11p_sd21_canny.ckpt'
batch_size = 8
logger_freq = 3000 #300
save_freq = 100
log_freq = 100
learning_rate = 1e-6
sd_locked = False #True
only_mid_control = False

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v21.yaml').cpu()
load_state_dict_with_shape(model, resume_path)
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

checkpoint_callback = ModelCheckpoint(
    save_top_k=-1,
    every_n_epochs=save_freq
)

# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=12, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq, log_freq=log_freq)
trainer = pl.Trainer(accelerator="gpu", gpus=1, precision=32, callbacks=[logger, checkpoint_callback], max_epochs=5000)
# trainer = pl.Trainer(strategy="ddp", accelerator="gpu", gpus=2, precision=32, callbacks=[logger, checkpoint_callback], max_epochs=5000) # strategy="ddp", 

# Train!
trainer.fit(model, dataloader)
