import os
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import SNN1
from datasets import MNISTSequencesDataset, NormaliseToZeroOneRange

from SNNCustomConfig import SNNCustomConfig

def run():

    device = torch.device(config.params["torch_device"])

    tao_mem = config.params["membrane_time_constant_tao"] # 10ms membrane time constant
    timestep = 1/config.params["dataset_sampling_frequency"]
    beta = np.exp(-timestep / tao_mem)

    # Data ingest and network initialisation
    normalise_transform = NormaliseToZeroOneRange(dtype=dtype)
    mnist_dataset = MNISTSequencesDataset(config.dirs["training"], config.params["LIF_linear_features"], transform=normalise_transform)
    train_loader = DataLoader(mnist_dataset, batch_size=config.params["batch_size"], shuffle=True, num_workers=config.params["dataloader_num_workers"])

    config.data_shape = next(iter(train_loader)).shape

    model = SNN1(num_steps=config.data_shape[1], beta=beta).to(device).to(dtype)

    # Initialisation of weights
    if config.params["init_type"] == 'pretrained' and config.params["LIF_linear_features"] == 28*28:
        model.lif1.recurrent.weight.data = torch.Tensor(np.load(config.pretrained_weights_path)).to(device).to(dtype)
    elif config.params["init_type"] == 'pretrained' and config.params["LIF_linear_features"] == 1024:
        class_weights_784 = torch.Tensor(np.load(config.pretrained_weights_path)).to(device).to(dtype)
        model.lif1.recurrent.weight.data[:class_weights_784.shape[0], :class_weights_784.shape[1]] = class_weights_784

    loss = nn.L1Loss()
    optimizer = config.get_optimizer(model)
    scheduler = config.get_scheduler(optimizer)
        
    max_grad_norm = config.params["grad_clipping_max_norm"]  # Define the maximum gradient norm threshold


    # TRAINING LOOP
    model, optimizer = config.iteration_begin_step(model, optimizer)

    loss_hist = []
    batch_loss = 0

    for epoch in range(config.start_epoch, config.params["epochs"]):
        config.epoch_begin_step()
        model.train(True)

        # Minibatch training loop
        for batch_i, data in enumerate(tqdm(train_loader, f'Epoch {epoch} Loss {batch_loss}')):

            data = data.to(device)

            optimizer.zero_grad(set_to_none=True)

            # Forward pass
            spk_rec, mem_rec = model(data)

            if config.params["loss_subtype"] == "preactivation": batch_loss = loss(mem_rec, torch.zeros_like(mem_rec)) # preactivation loss
            elif config.params["loss_subtype"] == "postactivation": batch_loss = loss(spk_rec, torch.zeros_like(mem_rec)) # postactivation loss

            # gradient calculation and weight update
            batch_loss.backward()
            if config.params["grad_clipping"] == "True": nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            loss_hist.append(batch_loss.item())
            config.batch_end_step(epoch, batch_i, batch_loss, optimizer, scheduler)
        config.epoch_end_step(epoch, batch_loss, optimizer, scheduler, model, loss_hist)
    config.iteration_end_step()

if __name__ == '__main__':
    
    # Setting random seeds
    torch.manual_seed(0)
    np.random.seed(0)
    
    ingest_torch_dtype = torch.uint8
    ingest_numpy_dtype = np.uint8

    dtype = torch.float32
    
    config = SNNCustomConfig(cli_args=sys.argv)
    # config = SNNCustomConfig(model_name="SNN1", dataset_name="mnist_sequences_10hz", configuration_name="config1", continue_training=False)

    run()