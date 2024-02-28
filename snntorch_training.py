import os
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms

import snntorch as snn

from SNNCustomConfig import SNNCustomConfig
import Metrics

def mkdir(path):
    if not os.path.exists(path): os.makedirs(path)
    return path

# def save_state(path, epoch, model, optimizer, loss_hist):
#     weights_save_file = f'snntorch_model_state_epoch_{epoch}_{datetime.datetime.now().strftime("%d-%m-%YT%H-%M")}.pth'
#     state = {
#         'epochs': num_epochs,
#         'model_state': model.state_dict(),
#         'optimizer_state': optimizer.state_dict(),
#         'loss_hist': loss_hist
#     }
#     torch.save(state, mkdir(path)/weights_save_file)

class NormaliseToZeroOneRange():
    def __init__(self, divide_by=255, dtype=torch.float16):
        self.divide_by = divide_by
        self.dtype = dtype
    def __call__(self, tensor):
        return (tensor / self.divide_by).to(self.dtype)

class MNISTSequencesDataset(Dataset):

    def __init__(self, file_path, transform=None):
        """
        Arguments:
            file_path (string): Path to the npy file with sequences.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = np.load(file_path, mmap_mode='r')
        self.transform = transform

    def perform_conversion(self, data):
        if config.params["LIF_linear_features"] == 1024: # Unsure how well this will work with passive memmap loading
            # Zero padding data to 32x32
            data_aug = np.zeros((data.shape[0], data.shape[1], 32, 32), dtype=ingest_numpy_dtype)
            data_aug[:,:,:28,:28] = data
            data = data_aug.copy()
            del data_aug
        # data = data.reshape((data.shape[0], data.shape[1], data.shape[2]*data.shape[3]))
        data = data.reshape((-1, data.shape[-2]*data.shape[-1]))
        data = torch.tensor(data, dtype=ingest_torch_dtype)
        return data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]
        sample = self.perform_conversion(sample)

        if self.transform:
            sample = self.transform(sample)

        return sample

class Net(nn.Module):
    def __init__(self, num_steps, beta):
        super().__init__()

        self.num_steps = num_steps

        # self.fc1 = nn.Linear(28*28, 28*28)
        self.lif1 = snn.RLeaky(beta=beta, linear_features=config.params["LIF_linear_features"], reset_mechanism=config.params["reset_mechanism"]) # also experiment with all_to_all and V (weights) parameters

    def forward(self, x):
        spk1, mem1 = self.lif1.init_rleaky()
        spk1, mem1 = spk1.to(dtype), mem1.to(dtype)

        spk1_rec = []
        mem1_rec = []

        for step in range(self.num_steps):
            # x = self.fc1(x)
            spk1, mem1 = self.lif1(x[:,step,:], spk1, mem1)

            spk1_rec.append(spk1)
            mem1_rec.append(mem1)

        # convert lists to tensors
        spk1_rec = torch.stack(spk1_rec)
        mem1_rec = torch.stack(mem1_rec)
        spk1_rec = torch.swapaxes(spk1_rec, 0, 1)
        mem1_rec = torch.swapaxes(mem1_rec, 0, 1)

        return spk1_rec, mem1_rec

def run():

    tao_mem = config.params["membrane_time_constant_tao"] # 10ms membrane time constant
    timestep = 1/config.params["dataset_sampling_frequency"]
    beta = np.exp(-timestep / tao_mem)

    # Data ingest and network initialisation
    normalise_transform = NormaliseToZeroOneRange(dtype=dtype)
    mnist_dataset = MNISTSequencesDataset(config.dirs["training"], transform=normalise_transform)
    train_loader = DataLoader(mnist_dataset, batch_size=config.params["batch_size"], shuffle=True, num_workers=config.params["dataloader_num_workers"])

    batch_data_shape = next(iter(train_loader)).shape

    model = Net(num_steps=batch_data_shape[1], beta=beta).to(device).to(dtype)

    # Initialisation of weights
    if config.params["init_type"] == 'pretrained' and config.params["LIF_linear_features"] == 28*28:
        model.lif1.recurrent.weight.data = torch.Tensor(np.load(config.pretrained_weights_path)).to(device).to(dtype)
    elif config.params["init_type"] == 'pretrained' and config.params["LIF_linear_features"] == 1024:
        class_weights_784 = torch.Tensor(np.load(config.pretrained_weights_path)).to(device).to(dtype)
        model.lif1.recurrent.weight.data[:class_weights_784.shape[0], :class_weights_784.shape[1]] = class_weights_784

    loss = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.params["optimiser_learning_rate"], betas=(0.9, 0.999))
    if config.params["scheduler_type"] == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.params["scheduler_StepLR_step_size"], gamma=config.params["scheduler_StepLR_gamma"])
        
    max_grad_norm = config.params["grad_clipping_max_norm"]  # Define the maximum gradient norm threshold

    config.check_weights_dir()
    metrics = Metrics()

    # TRAINING LOOP

    start_epoch = 0
    if config.continue_training:
        weights_file, start_epoch = config.get_last_weights_file_path()
        print("Continuing training from file:", weights_file, "| starting epoch:", start_epoch + 1)
        dict = torch.load(weights_file)
        model.load_state_dict(dict["model_state"])
        optimizer.load_state_dict(dict['optimizer_state'])
        config.previous_weights_file,_ = config.get_last_weights_file_path()

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} trainable parameters.')

    loss_hist = []
    # counter = 0
    batch_loss = 0

    metrics.init_iteration()
    if config.continue_training: metrics.load_interrupted_iteration(config)


    for epoch in range(start_epoch, config.params["epochs"]):
        model.train(True)
        metrics.init_epoch()
        # iter_counter = 0
        # train_batch = iter(train_loader)

        # Minibatch training loop
        # for data in tqdm(train_batch, f'Epoch {epoch} Loss {batch_loss}'):
        for i, data in enumerate(tqdm(train_loader, f'Epoch {epoch} Loss {batch_loss}')):

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
            metrics.batch_update(batch_loss.item())
            if config.params["step_scheduler_per"] == "batch": metrics.save_lr_metrics(config,epoch,i,config.get_lr(optimizer),batch_loss.item()); scheduler.step() # per batch lr + loss logging + per batch scheduler step
            # if config.training_parameters["step_scheduler_per"] == "batch": scheduler.step() # per batch scheduler step
        # if config.params["scheduler_type"] != 'none': scheduler.step()
        if config.params["step_scheduler_per"] == "epoch": metrics.save_lr_metrics(config,epoch,0,config.get_lr(optimizer),batch_loss.item()); scheduler.step() # per epoch lr + loss logging + per epoch scheduler step
        # if config.training_parameters["step_scheduler_per"] == "epoch": scheduler.step() # per epoch scheduler step
        metrics.epoch_update()
        config.save_at_checkpoint(epoch,model.state_dict(),optimizer.state_dict(),scheduler.state_dict(),loss_hist,config.get_weights_file_dir(metrics))
        metrics.epoch_end_print()
        metrics.save_metrics(config)
    metrics.iteration_update()
    print('Training completed.')

if __name__ == '__main__':
    
    # Setting random seeds
    torch.manual_seed(0)
    np.random.seed(0)
    
    ingest_torch_dtype = torch.uint8
    ingest_numpy_dtype = np.uint8

    dtype = torch.float32
    device = torch.device('cuda')
    
    config = SNNCustomConfig(cli_args=sys.argv)
    # config = SNNCustomConfig(model_name="SNN1", dataset_name="mnist_sequences_10hz", configuration_name="config1", continue_training=False)

    run()