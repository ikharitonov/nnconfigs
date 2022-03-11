#!/usr/bin/env python3

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
from torch.cuda.amp import autocast

import Metrics
import Config


config = Config.ImageNet_Local_Config("ResNet34", "config1", continue_training=False)

def run():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        config.dirs["training"],
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.training_parameters["batch_size"], shuffle= True,
        num_workers=config.training_parameters["dataloader_num_workers"], pin_memory=config.training_parameters["dataloader_pin_memory"])

    testloader = torch.utils.data.DataLoader(
        datasets.ImageFolder(config.dirs["testing"], transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=config.training_parameters["batch_size"], shuffle=True,
        num_workers=config.training_parameters["dataloader_num_workers"], pin_memory=config.training_parameters["dataloader_pin_memory"])

    config.check_weights_dir()

    metrics = Metrics()

    for iteration in range(config.training_parameters["iterations"]):
        model = config.get_model().to("cuda")
        criterion = nn.CrossEntropyLoss()
        optimizer = config.get_optimizer(model)
        
        max_steps = 0
        if config.training_parameters["step_scheduler_per"] == "epoch": max_steps = config.training_parameters["epochs"] # for stepping scheduler every epoch
        elif config.training_parameters["step_scheduler_per"] == "batch": max_steps = config.training_parameters["epochs"] * len(trainloader) # for stepping scheduler every batch
        schedule = config.get_scheduler(optimizer,max_steps)

        scaler = torch.cuda.amp.GradScaler()

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
        
        metrics.init_iteration()
        if config.continue_training: metrics.load_interrupted_iteration(config)

        for epoch in range(start_epoch, config.training_parameters["epochs"]):
            model.train(True)
            metrics.init_epoch()
            for i, data in enumerate(tqdm(trainloader)):
                inputs, labels = data
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                optimizer.zero_grad(set_to_none=True)
                with autocast():
                    preds = model(Variable(inputs.cuda()))
                    loss = criterion(preds.cuda(), labels.cuda())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                _,predicted = preds.max(1)

                metrics.batch_update(predicted.eq(labels).sum().item(), labels.size(0), loss.item())
                if config.training_parameters["step_scheduler_per"] == "batch": metrics.save_lr_metrics(config,epoch,i,config.get_lr(optimizer),loss.item()) # per batch lr + loss logging
                if config.training_parameters["step_scheduler_per"] == "batch": schedule.step() # per batch scheduler step
            if config.training_parameters["step_scheduler_per"] == "epoch": metrics.save_lr_metrics(config,epoch,0,config.get_lr(optimizer),loss.item()) # per epoch lr + loss logging
            if config.training_parameters["step_scheduler_per"] == "epoch": schedule.step() # per epoch scheduler step
            metrics.epoch_update(config.test(model, testloader))
            state = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'training_accuracy': metrics.per_epoch_training_accuracies[-1],
                'testing_accuracy':  metrics.per_epoch_testing_accuracies[-1]
            }
            config.save_at_checkpoint(epoch,state,config.get_weights_file_dir(metrics))
            metrics.epoch_end_print()
            metrics.save_metrics(config)
        metrics.iteration_update()

if __name__ == '__main__':
    run()