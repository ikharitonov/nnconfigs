#!/usr/bin/env python3

import os
import socket
import torch
from .Optimizers import *
from .ConfigParser import ConfigParser
from .Metrics import Metrics

def get_cli_args(args):

    out_dict = {}

    out_dict['model_name'] = None
    out_dict['dataset_name'] = None
    out_dict['configuration_name'] = None
    out_dict['configuration_file'] = None
    out_dict['continue_training'] = False

    if len(args) > 1:
        if args[1]=='--help':
            print("PARAMETERS:")
            print("--model_name : the name for the model to be trained")
            print("--dataset_name : the name for the dataset to train on")
            print("--configuration_name : training config, assumed to be the same as filename provided in configuration_file in e.g. \"config1\" (default)")
            print("--configuration_file : full path to the file to load training configuration from e.g. \"/full/path/to/config1.txt\" (default) or \"/full/path/to/config_baseline_monday.txt\"")
            print("--continue_training : option to continue training \"false\" (default) or \"true\"")
            print("EXAMPLE: python training_script.py --model_name VGG19 --dataset_name imagenet --configuration_name 1 --configuration_file baseline_c --continue_training false")
            exit()
        elif args[1]=='--create_config_file':
            """
            TODO: create txt file with one of the default configs
            """
            raise NotImplementedError("not implemented")

    # for key in out_dict.keys():
    for key in ['model_name', 'dataset_name', 'configuration_name']: # Must-have information
        if key not in args and f"--{key}" not in args and f"-{key}" not in args:
            print(f"Error: {key} missing from the command line arguments. Exiting.")
            exit()
    for key in ['configuration_file', 'continue_training']:
        if key not in args and f"--{key}" not in args and f"-{key}" not in args:
            print(f"Warning: {key} missing from the command line arguments. Using default value/s.")

    for i in range(1,len(args),2):
        if args[i]=='--model_name':
            model_name = args[i+1]
            out_dict['model_name'] = model_name
        elif args[i]=='--dataset_name':
            dataset_name = args[i+1]
            out_dict['dataset_name'] = dataset_name
        elif args[i]=='--configuration_name':
            configuration_name = args[i+1]
            out_dict['configuration_name'] = configuration_name
        elif args[i]=='--configuration_file':
            configuration_file = args[i+1]
            out_dict['configuration_file'] = configuration_file
        elif args[i]=='--continue_training':
            continue_training = args[i+1].lower()
            if continue_training=='false': continue_training = False
            elif continue_training=='true': continue_training = True
            out_dict['continue_training'] = continue_training
    
    return out_dict
class BaseConfig:

    def __init__(self, cli_args=None, model_name=None, dataset_name=None, configuration_name=None, configuration_file=None, continue_training=False):
        if cli_args:
            parsed_args = get_cli_args(cli_args)
            model_name = parsed_args["model_name"]
            dataset_name = parsed_args["dataset_name"]
            configuration_name = parsed_args["configuration_name"]
            configuration_file = parsed_args["configuration_file"]
            continue_training = parsed_args["continue_training"]

        self.model_name = model_name
        self.dataset_name = dataset_name
        self.training_name = f"{model_name}_{dataset_name}"
        self.configuration_name = configuration_name
        self.continue_training = continue_training
        self.running_machine = socket.gethostname()
        self.weights_save_dir = self.base_path / self.training_name
        self.previous_weights_file = ''
        self.parser = ConfigParser(configuration_file, self)
        self.params = self.parser.training_parameters
        self.save_checkpoints = [x for x in range(self.params["epochs"])]
        self.data_shape = None
        self.start_epoch = 0
        self.metrics = Metrics()

    def update_weights_save_dir(self):
        self.weights_save_dir = self.base_path + self.slash + self.training_name + self.weights_dataset_name + self.slash
    
    def update_training_parameters(self):
        self.parser = ConfigParser('void', self)
        self.params = self.parser.training_parameters
                    
    def get_weights_file_dir(self):
        # returns path + file name for a new weights file to be saved, naming it according to latest epoch metrics
        new_filename = f"{self.training_name}_i_{len(self.metrics.per_iteration_training_losses)}_epoch_{len(self.metrics.per_epoch_training_losses)-1}_loss_{self.metrics.per_epoch_training_losses[-1]:.3f}.pth"
        return self.weights_save_dir / self.configuration_name / new_filename
    
    def check_weights_dir(self):
        # checks if the folder for storing the weights already exists, if not, creates it
        if not os.path.exists(self.weights_save_dir):
            os.makedirs(self.weights_save_dir)
            os.makedirs(self.weights_save_dir / self.configuration_name)
        if not os.path.exists(self.weights_save_dir / self.configuration_name):
            os.makedirs(self.weights_save_dir / self.configuration_name)
        # transferring loaded config file into the training directory
        if not self.continue_training: self.parser.transfer_config_file(self.weights_save_dir / self.configuration_name)
    
    def get_last_weights_file_path(self):
        # called if we want to continue training the model from last epoch/weights file
        # returns the path + file name of last weights file at which training was interrupted to be loaded in training script + the last epoch
        p = self.weights_save_dir / self.configuration_name
        file_list = [f for f in os.listdir(p) if os.path.isfile(os.path.join(p, f))]
        if os.path.isfile(p / "metrics.csv"):
            file_list.remove("metrics.csv")
        if os.path.isfile(p / "lr_metrics.csv"):
            file_list.remove("lr_metrics.csv")
        if os.path.isfile(p / f"{self.configuration_name}.txt"):
            file_list.remove(f"{self.configuration_name}.txt")
        epoch_list = []
        for file in file_list:
            string_split = file.split('_')
            epoch_ind = -1
            for e in string_split:
                if e == "epoch":
                    epoch_ind = string_split.index(e)+1
            epoch_list.append(int(string_split[epoch_ind]))
        if len(epoch_list) == 0:
            print("Error: no previous model weights found in the directory " + p.name)
            print("Set continue_training to False")
            exit()
        return p / file_list[epoch_list.index(max(epoch_list))], max(epoch_list)
    
    def save_at_checkpoint(self,current_weights_file=None,epoch=None,model_state_dict=None,optimizer_state_dict=None,scheduler_state_dict=None,loss_history=None,training_accuracy=None,testing_accuracy=None):
        # This function saves the model weights at every epoch and deletes the previous epoch weights, unless it's epoch 1 or a checkpoint
        state = {
                'epoch': epoch,
                'model_state': model_state_dict,
                'optimizer_state': optimizer_state_dict,
                'scheduler_state': scheduler_state_dict,
                'loss_history': loss_history,
                'training_accuracy': training_accuracy,
                'testing_accuracy': testing_accuracy
        }
        torch.save(state, current_weights_file)
        if epoch==0: self.previous_weights_file = current_weights_file
        elif epoch in self.save_checkpoints: self.previous_weights_file = current_weights_file
        else:
            os.remove(self.previous_weights_file)
            self.previous_weights_file = current_weights_file
    
    def get_optimizer(self, model):
        if self.params["optimiser_type"] == "SGD":
            optimizer = torch.optim.SGD(model.parameters(),lr = self.params["optimiser_learning_rate"],momentum = self.params["optimiser_momentum"],weight_decay = self.params["optimiser_weight_decay"])
        elif self.params["optimiser_type"] == "Adam":
            optimizer = torch.optim.Adam(model.parameters(),lr = self.params["optimiser_learning_rate"],weight_decay = self.params["optimiser_weight_decay"], betas=(0.9, 0.999))
        return optimizer
    
    def get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()
    
    def get_scheduler(self, optimizer):
        if self.params["scheduler_type"].lower() == "steplr":
            return torch.optim.lr_scheduler.StepLR(optimizer,step_size=self.params["scheduler_StepLR_step_size"],gamma=self.params["scheduler_StepLR_gamma"])
        elif self.params["scheduler_type"].lower() == "warmupcosinelr":
            # max_steps for stepping scheduler every epoch
            if self.params["step_scheduler_per"] == "epoch": max_steps = self.params["epochs"]
            # max_steps for stepping scheduler every batch
            elif self.params["step_scheduler_per"] == "batch": max_steps = int(self.params["epochs"] * self.data_shape[0]/self.params["batch_size"])
            return WarmupCosineLR(optimizer,max_iters=max_steps,warmup_factor=self.params["scheduler_WarmupCosineLR_warmup_factor"],warmup_iters=self.params["scheduler_WarmupCosineLR_warmup_iterations"],warmup_method="linear",last_epoch=-1,)
        elif self.params["scheduler_type"].lower() == "cosineannealingwarmrestarts":
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.params["scheduler_CosineAnnealingWarmRestarts_restart_period"], T_mult=self.params["scheduler_CosineAnnealingWarmRestarts_period_multiplier"], eta_min=self.params["scheduler_CosineAnnealingWarmRestarts_min_lr"])
        elif self.params["scheduler_type"].lower() == "reducelronplateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.params["scheduler_ReduceLROnPlateau_factor"], patience=self.params["scheduler_ReduceLROnPlateau_patience"], threshold=self.params["scheduler_ReduceLROnPlateau_threshold"])
            

    def get_lr(self,optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']
        
    def print_model_parameters(self, model):
        total_params = sum(p.numel() for p in model.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} trainable parameters.')

    def check_for_training_continuation(self, model, optimizer, weights_file):
        dict = None
        if self.continue_training:
            if weights_file:
                print("Continuing training from file:", weights_file)
            else:
                weights_file, latest_epoch = self.get_last_weights_file_path()
                self.start_epoch = latest_epoch + 1
                print("Continuing training from file:", weights_file, "| starting epoch:", self.start_epoch)
            dict = torch.load(weights_file)
            model.load_state_dict(dict["model_state"])
            optimizer.load_state_dict(dict['optimizer_state'])
            self.previous_weights_file,_ = self.get_last_weights_file_path()

            self.metrics.load_interrupted_iteration(self)
        return model, optimizer, dict
    
    def iteration_begin_step(self, model, optimizer, specify_weights_file=None):
        self.check_weights_dir()
        self.metrics.init_iteration()
        model, optimizer, dict = self.check_for_training_continuation(model, optimizer, specify_weights_file)
        self.print_model_parameters(model)
        return model, optimizer, dict
    
    def epoch_begin_step(self):
        self.metrics.init_epoch()
    
    def batch_end_step(self, epoch, batch_i, batch_loss, optimizer, scheduler):
        self.metrics.batch_update(batch_loss.item())
        if self.params["step_scheduler_per"] == "batch":
            self.metrics.save_lr_metrics(self,epoch,batch_i,self.get_lr(optimizer),batch_loss.item()) # per batch lr + loss logging
            scheduler.step() # per batch scheduler step
    
    def epoch_end_step(self, epoch=None, batch_loss=None, optimizer=None, scheduler=None, model=None, loss_history=None):
        if self.params["step_scheduler_per"] == "epoch":
            self.metrics.save_lr_metrics(self,epoch,0,self.get_lr(optimizer),batch_loss.item()) # per epoch lr + loss logging
            scheduler.step() # per epoch scheduler step
        self.metrics.epoch_update()
        self.save_at_checkpoint(current_weights_file=self.get_weights_file_dir(),epoch=epoch,model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),scheduler_state_dict=scheduler.state_dict(),loss_history=loss_history)
        self.metrics.epoch_end_print()
        self.metrics.save_metrics(self)

    def iteration_end_step(self):
        self.metrics.iteration_update()
        print('Training completed.')