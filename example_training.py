#!/usr/bin/env python3

import sys
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
from tqdm import tqdm

from ExampleConfig import ExampleConfig

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
        train_dataset, batch_size=config.params["batch_size"], shuffle=True,
        num_workers=config.params["dataloader_num_workers"], pin_memory=config.params["dataloader_pin_memory"])

    testloader = torch.utils.data.DataLoader(
        datasets.ImageFolder(config.dirs["testing"], transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=config.params["batch_size"], shuffle=True,
        num_workers=config.params["dataloader_num_workers"], pin_memory=config.params["dataloader_pin_memory"])

    config.data_shape = next(iter(trainloader)).shape

    for iteration in range(config.params["iterations"]):
        model = Model1()
        criterion = nn.CrossEntropyLoss()
        optimizer = config.get_optimizer(model)
        scheduler = config.get_scheduler(optimizer)

        model, optimizer = config.iteration_begin_step(model, optimizer)

        for epoch in range(config.start_epoch, config.params["epochs"]):
            config.epoch_begin_step()
            model.train(True)
            for batch_i, data in enumerate(tqdm(trainloader)):
                inputs, labels = data

                optimizer.zero_grad(set_to_none=True)

                predictions = model(inputs.cuda())
                loss = criterion(predictions.cuda(), labels.cuda())

                loss.backward()
                optimizer.step()

                config.batch_end_step(epoch, batch_i, loss, optimizer, scheduler)
            config.epoch_end_step(epoch=epoch, batch_loss=loss, optimizer=optimizer, scheduler=scheduler, model=model)
        config.iteration_end_step()

if __name__ == '__main__':
    config = ExampleConfig(cli_args=sys.argv)
    # config = ExampleConfig(model_name="Model1", dataset_name="Dataset1", configuration_name="config1", configuration_file="/path/to/config1.txt" continue_training=False)
    run()