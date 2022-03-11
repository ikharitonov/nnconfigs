#!/usr/bin/env python3

import os
import torch
from torch.autograd import Variable
from tqdm import tqdm
from Optimizers import WarmupCosineLR
from models.ResNet import resnet34

class Config:
    def __init__(self, model_type, parameter_config, continue_training=False):
        self.model_type = model_type
        self.parameter_config = parameter_config
        self.continue_training = continue_training
        self.previous_weights_file = ''
        self.save_checkpoints = [1,25,50,75,100,125,150,175,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000,3100,3200,3300,3400,3500,3600,3700,3800,3900,4000]
        self.training_parameters = {"image_dim": 0} # initialized in child classes
    def get_model(self):
        name = self.model_type.split('_')[0]
        if name == 'ResNet34':
            return resnet34(num_classes=self.training_parameters["model_out_classes"])
    def get_weights_file_dir(self, m):
        # returns path + file name for a new weights file to be saved, naming it according to latest epoch metrics
        return self.weights_save_dir + self.parameter_config + self.slash + self.model_type + "_i_" + str(len(m.per_iteration_training_accuracies)) + "_epoch_" + str(len(m.per_epoch_training_accuracies)) + '_trainaccuracy_' + str(int(m.per_epoch_training_accuracies[-1])) + "_testaccuracy_" + str(int(m.per_epoch_testing_accuracies[-1])) +'.pth'
    def check_weights_dir(self):
        # checks if the folder for storing the weights already exists, if not, creates it
        if not os.path.exists(self.weights_save_dir):
            os.makedirs(self.weights_save_dir)
            os.makedirs(self.weights_save_dir + self.slash + self.parameter_config)
        if not os.path.exists(self.weights_save_dir + self.slash + self.parameter_config):
            os.makedirs(self.weights_save_dir + self.slash + self.parameter_config)
    def get_last_weights_file_path(self):
        # called if we want to continue training the model from last epoch/weights file
        # returns the path + file name of last weights file at which training was interrupted to be loaded in training script + the last epoch
        p = self.weights_save_dir + self.parameter_config + self.slash
        file_list = [f for f in os.listdir(p) if os.path.isfile(os.path.join(p, f))]
        if os.path.isfile(p + "metrics.csv"):
            file_list.remove("metrics.csv")
        if os.path.isfile(p + "lr_metrics.csv"):
            file_list.remove("lr_metrics.csv")
        epoch_list = []
        for file in file_list:
            string_split = file.split('_')
            epoch_ind = -1
            for e in string_split:
                if e == "epoch":
                    epoch_ind = string_split.index(e)+1
            epoch_list.append(int(string_split[epoch_ind]))
        if len(epoch_list) == 0:
            print("Error: no previous model weights found in the directory " + p)
            print("Set continue_training to False")
            exit()
        return p + file_list[epoch_list.index(max(epoch_list))], max(epoch_list)
    def save_at_checkpoint(self,epoch,state,current_weights_file):
        # This function saves the model weights at every epoch and deletes the previous epoch weights, unless it's epoch 1 or a checkpoint
        torch.save(state, current_weights_file)
        if epoch==0: self.previous_weights_file = current_weights_file
        elif epoch in self.save_checkpoints: self.previous_weights_file = current_weights_file
        else:
            os.remove(self.previous_weights_file)
            self.previous_weights_file = current_weights_file

    def test(self, model, testloader):
        # Fuction performing testing and returning testing accuracy
        correct = 0
        total = 0
        accuracy = 0
        model.train(False)
        with torch.no_grad():
            for i,(images,labels)in enumerate(tqdm(testloader)):
                if torch.cuda.is_available():
                    images = images.cuda()
                    labels = labels.cuda()
                outputs = model(Variable(images.cuda()))
                labels = Variable(labels.cuda())

                _,predicted = outputs.max(1)
                correct = predicted.eq(labels).sum().item()
                total = labels.size(0)
                accuracy+=100*(correct/total)
        return accuracy/len(testloader)
    def get_optimizer(self, model):
        ml = list()
        count =0 
        for name, param in model.named_parameters():
            if count == 0: #save threshold layer with different weight decay 
                ml.append({'params': param, 'weight_decay': 0})
                #print(name, "   decay 0")
            else:
                ml.append({'params': param})
                #print(name, "   decay 0.006")
            count +=1
        #print("number of parameters in model", count)
        if self.training_parameters["optimiser_type"] == "SGD":
            optimizer = torch.optim.SGD(ml,lr = self.training_parameters["optimiser_learning_rate"],momentum = self.training_parameters["optimiser_momentum"],weight_decay = self.training_parameters["optimiser_weight_decay"])
        elif self.training_parameters["optimiser_type"] == "Adam":
            optimizer = torch.optim.Adam(ml,lr = self.training_parameters["optimiser_learning_rate"],weight_decay = self.training_parameters["optimiser_weight_decay"])
        return optimizer
    def get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()
    def get_scheduler(self, optimizer, max_steps):
        if self.training_parameters["scheduler_type"] == "StepLR":
            return torch.optim.lr_scheduler.StepLR(optimizer,step_size=self.training_parameters["scheduler_StepLR_step_size"],gamma=self.training_parameters["scheduler_StepLR_gamma"])
        elif self.training_parameters["scheduler_type"] == "WarmupCosineLR":
            return WarmupCosineLR(optimizer,max_iters=max_steps,warmup_factor=self.training_parameters["scheduler_WarmupCosineLR_warmup_factor"],warmup_iters=self.training_parameters["scheduler_WarmupCosineLR_warmup_iterations"],warmup_method="linear",last_epoch=-1,)
    def get_lr(self,optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

class Local_Config(Config):
    def __init__(self, model_type, parameter_config, continue_training=False):
        super(Local_Config,self).__init__(model_type, parameter_config, continue_training)
        self.running_machine = "Local"
        self.base_path = "base\\path\\"
        self.base_data_path = "base\\data\\path"
        self.slash = "\\"

class RCS_Home_Config(Config):
    def __init__(self, model_type, parameter_config, continue_training=False):
        super(RCS_Home_Config,self).__init__(model_type, parameter_config, continue_training)
        self.running_machine = "RCS_home"
        self.base_path = "/base/path"
        self.base_data_path = "/base/data/path"
        self.slash = "/"

class RCS_Ephemeral_Config(Config):
    def __init__(self, model_type, parameter_config, continue_training=False):
        super(RCS_Ephemeral_Config,self).__init__(model_type, parameter_config, continue_training)
        self.running_machine = "RCS_ephemeral"
        self.base_path = "/base/path"
        self.base_data_path = "/bast/data/path"
        self.slash = "/"

class ImageNet_Local_Config(Local_Config):
    def __init__(self, model_type, parameter_config, continue_training=False):
        super(ImageNet_Local_Config,self).__init__(model_type, parameter_config, continue_training)
        self.training_parameters = {"image_dim": 224,
                                        "batch_size": 40,
                                        "iterations": 1,
                                        "epochs": 100,
                                        "model_out_classes": 2,
                                        "distribured_mode": "DP", # DP for DataParallel and DDP for DistributedDataParallel
                                        "ddp_world_size": 1,
                                        "ddp_backend_type": "nccl", # nccl or gloo
                                        "optimiser_type": "Adam", # SGD or Adam
                                        "optimiser_learning_rate": 0.0002,
                                        "optimiser_momentum": 0.9,
                                        "optimiser_weight_decay": 0.006,
                                        "dataloader_num_workers": 2,
                                        "dataloader_pin_memory": True,
                                        "scheduler_type": "WarmupCosineLR", # StepLR or WarmupCosineLR
                                        "scheduler_StepLR_step_size": 5,
                                        "scheduler_StepLR_gamma": 0.7,
                                        "scheduler_WarmupCosineLR_warmup_factor": 0.001,
                                        "scheduler_WarmupCosineLR_warmup_iterations": 3,
                                        "step_scheduler_per": "epoch"} # batch or epoch
        self.dirs = {"training": self.base_data_path + "\\data\\imagenet\\test_train",
                    "testing": self.base_data_path + "\\data\\imagenet\\test_val"}
        self.weights_save_dir = self.base_path + self.slash + self.model_type + "-ImageNet_Weights" + self.slash
        
class CIFAR10_Local_Config(Local_Config):
    def __init__(self, model_type, parameter_config, continue_training=False):
        super(CIFAR10_Local_Config,self).__init__(model_type, parameter_config, continue_training)
        self.training_parameters = {"image_dim": 32,
                                        "batch_size": 500,
                                        "iterations": 1,
                                        "epochs": 100,
                                        "model_out_classes": 10,
                                        "distribured_mode": "DP",
                                        "ddp_world_size": 4,
                                        "ddp_backend_type": "nccl",
                                        "optimiser_type": "SGD",
                                        "optimiser_learning_rate": 0.001,
                                        "optimiser_momentum": 0.9,
                                        "optimiser_weight_decay": 0.006,
                                        "dataloader_num_workers": 4,
                                        "dataloader_pin_memory": True,
                                        "scheduler_type": "StepLR",
                                        "scheduler_StepLR_step_size": 20,
                                        "scheduler_StepLR_gamma": 0.7,
                                        "scheduler_WarmupCosineLR_warmup_factor": 0.001,
                                        "scheduler_WarmupCosineLR_warmup_iterations": 40,
                                        "step_scheduler_per": "epoch"}
        self.dirs = {"training": self.base_data_path + "\\data\\cifar\\train",
                    "testing": self.base_data_path + "\\data\\cifar\\test"}
        self.weights_save_dir = self.base_path + self.slash + self.model_type + "-CIFAR10_Weights" + self.slash

class ImageNet_RCS_Config(RCS_Home_Config):
    def __init__(self, model_type, parameter_config, continue_training=False):
        super(ImageNet_RCS_Config,self).__init__(model_type, parameter_config, continue_training)
        self.training_parameters = {"image_dim": 224,
                                        "batch_size": 256,
                                        "iterations": 1,
                                        "epochs": 150,
                                        "model_out_classes": 1000,
                                        "distribured_mode": "DP",
                                        "ddp_world_size": 6,
                                        "ddp_backend_type": "nccl",
                                        "optimiser_type": "Adam",
                                        "optimiser_learning_rate": 0.0004,
                                        "optimiser_momentum": 0.9,
                                        "optimiser_weight_decay": 0.006,
                                        "dataloader_num_workers": 12,
                                        "dataloader_pin_memory": False,
                                        "scheduler_type": "WarmupCosineLR",
                                        "scheduler_StepLR_step_size": 20,
                                        "scheduler_StepLR_gamma": 0.7,
                                        "scheduler_WarmupCosineLR_warmup_factor": 0.001,
                                        "scheduler_WarmupCosineLR_warmup_iterations": 4000,
                                        "step_scheduler_per": "batch"}
        self.dirs = {"training": self.base_data_path + "/imagenet/train",
                    "testing": self.base_data_path + "/imagenet/val"}
        self.weights_save_dir = self.base_path + self.slash + self.model_type + "-ImageNet_Weights" + self.slash
        
class CIFAR10_RCS_Config(RCS_Home_Config):
    def __init__(self, model_type, parameter_config, continue_training=False):
        super(CIFAR10_RCS_Config,self).__init__(model_type, parameter_config, continue_training)
        self.training_parameters = {"image_dim": 32,
                                        "batch_size": 200,
                                        "iterations": 1,
                                        "epochs": 500,
                                        "model_out_classes": 10,
                                        "distribured_mode": "DP",
                                        "ddp_world_size": 4,
                                        "ddp_backend_type": "nccl",
                                        "optimiser_type": "SGD",
                                        "optimiser_learning_rate": 0.01,
                                        "optimiser_momentum": 0.9,
                                        "optimiser_weight_decay": 0.006,
                                        "dataloader_num_workers": 8,
                                        "dataloader_pin_memory": True,
                                        "scheduler_type": "StepLR",
                                        "scheduler_StepLR_step_size": 20,
                                        "scheduler_StepLR_gamma": 0.7,
                                        "scheduler_WarmupCosineLR_warmup_factor": 0.001,
                                        "scheduler_WarmupCosineLR_warmup_iterations": 40,
                                        "step_scheduler_per": "epoch"}
        self.dirs = {"training": self.base_data_path + "/cifar/train",
                    "testing": self.base_data_path + "/cifar/test"}
        self.weights_save_dir = self.base_path + self.slash + self.model_type + "-CIFAR10_Weights" + self.slash

class ImageNette_RCS_Config(RCS_Ephemeral_Config):
    def __init__(self, model_type, parameter_config, continue_training=False):
        super(ImageNette_RCS_Config,self).__init__(model_type, parameter_config, continue_training)
        self.training_parameters = {"image_dim": 224,
                                        "batch_size": 256,
                                        "iterations": 1,
                                        "epochs": 1200,
                                        "model_out_classes": 10,
                                        "distribured_mode": "DP",
                                        "ddp_world_size": 6,
                                        "ddp_backend_type": "nccl",
                                        "optimiser_type": "Adam",
                                        "optimiser_learning_rate": 0.0002,
                                        "optimiser_momentum": 0.9,
                                        "optimiser_weight_decay": 0.006,
                                        "dataloader_num_workers": 12,
                                        "dataloader_pin_memory": False,
                                        "scheduler_type": "WarmupCosineLR",
                                        "scheduler_StepLR_step_size": 20,
                                        "scheduler_StepLR_gamma": 0.7,
                                        "scheduler_WarmupCosineLR_warmup_factor": 0.001,
                                        "scheduler_WarmupCosineLR_warmup_iterations": 125,
                                        "step_scheduler_per": "epoch"}
        self.dirs = {"training": self.base_data_path + "/imagenette2/train",
                    "testing": self.base_data_path + "/imagenette2/val"}
        self.weights_save_dir = self.base_path + self.slash + self.model_type + "-ImageNette_Weights" + self.slash