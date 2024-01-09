
import os
import io

from PIL import Image
import matplotlib.pyplot as plt

import torch.multiprocessing
from torchmetrics import Metric
from torchvision import transforms as T
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def add_plot(writer, name, step):
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg', dpi=100)
    buf.seek(0)
    image = Image.open(buf)
    image = T.ToTensor()(image)
    writer.add_image(name, image, step)
    plt.clf()
    plt.close()

def _remove_axes(ax):
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.set_xticks([])
    ax.set_yticks([])

def remove_axes(axes):
    if len(axes.shape) == 2:
        for ax1 in axes:
            for ax in ax1[:]:
                _remove_axes(ax)
    else:
        for ax in axes:
            _remove_axes(ax)

def prep_args():
    import sys

    old_args = sys.argv
    new_args = [old_args.pop(0)]
    while len(old_args) > 0:
        arg = old_args.pop(0)
        if len(arg.split("=")) == 2:
            new_args.append(arg)
        elif arg.startswith("--"):
            new_args.append(arg[2:] + "=" + old_args.pop(0))
        else:
            raise ValueError("Unexpected arg style {}".format(arg))
    sys.argv = new_args

def get_ckpt_callback(save_path, exp_name):
    ckpt_dir = os.path.join(save_path, exp_name)
    return ModelCheckpoint(dirpath=ckpt_dir,
                           save_top_k=1,
                           every_n_epochs=1,
                           verbose=True,
                           monitor='validation/acc',
                           mode='max')


def get_early_stop_callback(patience=100):
    return EarlyStopping(monitor='validation/acc',
                         patience=patience,
                         verbose=True,
                         mode='max')