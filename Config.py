#!/usr/bin/env python3

import os
import torch
import numpy as np
from pathlib import Path
from torch.autograd import Variable
from tqdm import tqdm
import Optimizers as Optimizers
import ConfigParser as ConfigParser
from torch.utils.data import DataLoader

def get_cli_args(args):

    out_dict = {}

    out_dict['model_name'] = None
    out_dict['dataset_name'] = None
    out_dict['configuration_name'] = None
    out_dict['configuration_file'] = None
    out_dict['continue_training'] = False

    for key in out_dict.keys():
        if key not in args or f"--{key}" not in args or f"-{key}" not in args:
            print(f"Error: {key} missing from the command line arguments. Exiting.")
            exit()

    if len(args) > 1:
        if args[1]=='--help':
            print("PARAMERES:")
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

        self.training_name = f"{model_name}_{dataset_name}"
        self.configuration_name = configuration_name
        self.continue_training = continue_training
        self.previous_weights_file = ''
        self.save_checkpoints = [1,25,50,75,100,125,150,175,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000,3100,3200,3300,3400,3500,3600,3700,3800,3900,4000]
        # self.training_parameters = {} # initialized in child classes
        self.parser = ConfigParser.ConfigParser(configuration_file, self)
        self.params = self.parser.training_parameters
    
    def update_weights_save_dir(self):
        self.weights_save_dir = self.base_path + self.slash + self.training_name + self.weights_dataset_name + self.slash
    
    def update_training_parameters(self):
        self.parser = ConfigParser.ConfigParser('void', self)
        self.params = self.parser.training_parameters
    
    # def get_model(self, params=None):
    #     name = self.training_name.split('_')[0]
    #     if name == 'VGG16':
    #         return VGG.VGG16(self)
    #     elif name == 'ResNet34':
    #         temp_training_name = self.training_name
    #         input_shape = (self.training_parameters["input_channels"],self.training_parameters["image_dim"],self.training_parameters["image_dim"])
    #         first_conv_flex = False
    #         if temp_training_name.split('_')[1][0] == 'c':
    #             first_conv_flex = True
    #             temp_training_name = temp_training_name[:9] + temp_training_name[10:]
    #         flex_layers = []
    #         for char in temp_training_name.split('_')[1][:4]:
    #             if int(char): flex_layers.append(True)
    #             else: flex_layers.append(False)
    #         if temp_training_name.split('_')[1][:4]=='0000':
    #             return ResNet.resnet34(input_shape, flex_layers = flex_layers, num_classes=self.training_parameters["model_out_classes"],first_conv_flex=first_conv_flex)
    #         else:
    #             return ResNet.resnet34(input_shape, flex_layers = flex_layers, flex_mode=temp_training_name.split('-')[-1], flex_block_mode=temp_training_name.split('_')[1][4:].split('-')[0], num_classes=self.training_parameters["model_out_classes"],first_conv_flex=first_conv_flex)
    #     elif name == 'UNet':
    #         model = UNet(params['n_channels'], params['n_classes'], use_DP=params['use_DP'])
    #         m_type = self.training_name.split('-')[0]
    #         if m_type == 'UNet_Baseline':
    #             return model
    #         else:
    #             layer = None
    #             flex_type = self.training_name.split('-')[1]
    #             if flex_type == 'DefaultFlex':
    #                 layer = FlexLayer_DefaultFlex
    #             elif flex_type == 'Random50_50':
    #                 layer = FlexLayer_Random50_50
    #             def get_flex_layer(module):
    #                 return layer(in_channels=module.in_channels,out_channels=module.out_channels,kernel_size=module.kernel_size,stride=module.stride,padding=module.padding,img_dim=params['input_shape'])
                
    #             if m_type == 'UNet_All_Encoder':
    #                 flex_layer = get_flex_layer(model.inc.double_conv[0])
    #                 model.inc.double_conv[0] = flex_layer
    #                 flex_layer = get_flex_layer(model.inc.double_conv[3])
    #                 model.inc.double_conv[3] = flex_layer
    #                 flex_layer = get_flex_layer(model.down1.maxpool_conv[1].double_conv[0])
    #                 model.down1.maxpool_conv[1].double_conv[0] = flex_layer
    #                 flex_layer = get_flex_layer(model.down1.maxpool_conv[1].double_conv[3])
    #                 model.down1.maxpool_conv[1].double_conv[3] = flex_layer
    #                 flex_layer = get_flex_layer(model.down2.maxpool_conv[1].double_conv[0])
    #                 model.down2.maxpool_conv[1].double_conv[0] = flex_layer
    #                 flex_layer = get_flex_layer(model.down2.maxpool_conv[1].double_conv[3])
    #                 model.down2.maxpool_conv[1].double_conv[3] = flex_layer
    #                 flex_layer = get_flex_layer(model.down3.maxpool_conv[1].double_conv[0])
    #                 model.down3.maxpool_conv[1].double_conv[0] = flex_layer
    #                 flex_layer = get_flex_layer(model.down3.maxpool_conv[1].double_conv[3])
    #                 model.down3.maxpool_conv[1].double_conv[3] = flex_layer
    #                 flex_layer = get_flex_layer(model.down4.maxpool_conv[1].double_conv[0])
    #                 model.down4.maxpool_conv[1].double_conv[0] = flex_layer
    #                 flex_layer = get_flex_layer(model.down4.maxpool_conv[1].double_conv[3])
    #                 model.down4.maxpool_conv[1].double_conv[3] = flex_layer

    #                 return model
    #             elif m_type == 'UNet_Single_Layer':
    #                 flex_layer = get_flex_layer(model.inc.double_conv[0])
    #                 model.inc.double_conv[0] = flex_layer
    #                 return model
                    
    def get_weights_file_dir(self, m):
        # returns path + file name for a new weights file to be saved, naming it according to latest epoch metrics
        new_filename = f"{self.training_name}_i_{len(m.per_iteration_training_losses)}_epoch_{len(m.per_epoch_training_losses)}_loss_{m.per_epoch_training_losses[-1]:.3f}.pth"
        return self.weights_save_dir / self.configuration_name / new_filename
    
    def check_weights_dir(self):
        # checks if the folder for storing the weights already exists, if not, creates it
        if not os.path.exists(self.weights_save_dir):
            os.makedirs(self.weights_save_dir)
            os.makedirs(self.weights_save_dir + self.slash + self.configuration_name)
        if not os.path.exists(self.weights_save_dir + self.slash + self.configuration_name):
            os.makedirs(self.weights_save_dir + self.slash + self.configuration_name)
        # transferring loaded config file into the training directory
        if not self.continue_training: self.parser.transfer_config_file(self.weights_save_dir / self.configuration_name)
    
    def get_last_weights_file_path(self):
        # called if we want to continue training the model from last epoch/weights file
        # returns the path + file name of last weights file at which training was interrupted to be loaded in training script + the last epoch
        p = self.weights_save_dir + self.configuration_name + self.slash
        file_list = [f for f in os.listdir(p) if os.path.isfile(os.path.join(p, f))]
        if os.path.isfile(p + "metrics.csv"):
            file_list.remove("metrics.csv")
        if os.path.isfile(p + "lr_metrics.csv"):
            file_list.remove("lr_metrics.csv")
        if os.path.isfile(p + self.configuration_name + ".txt"):
            file_list.remove(self.configuration_name + ".txt")
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
    
    def save_at_checkpoint(self,epoch,model_state_dict,optimizer_state_dict,scheduler_state_dict,loss_history,current_weights_file,training_accuracy=None,testing_accuracy=None):
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
        if self.params["optimiser_type"] == "SGD":
            optimizer = torch.optim.SGD(ml,lr = self.params["optimiser_learning_rate"],momentum = self.params["optimiser_momentum"],weight_decay = self.params["optimiser_weight_decay"])
        elif self.params["optimiser_type"] == "Adam":
            optimizer = torch.optim.Adam(ml,lr = self.params["optimiser_learning_rate"],weight_decay = self.params["optimiser_weight_decay"])
        return optimizer
    
    def get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()
    
    def get_scheduler(self, optimizer, max_steps):
        if self.params["scheduler_type"] == "StepLR":
            return torch.optim.lr_scheduler.StepLR(optimizer,step_size=self.params["scheduler_StepLR_step_size"],gamma=self.params["scheduler_StepLR_gamma"])
        elif self.params["scheduler_type"] == "WarmupCosineLR":
            return Optimizers.WarmupCosineLR(optimizer,max_iters=max_steps,warmup_factor=self.params["scheduler_WarmupCosineLR_warmup_factor"],warmup_iters=self.params["scheduler_WarmupCosineLR_warmup_iterations"],warmup_method="linear",last_epoch=-1,)
        elif self.params["scheduler_type"] == "CosineAnnealingWarmRestarts":
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.params["scheduler_CosineAnnealingWarmRestarts_restart_period"], T_mult=self.params["scheduler_CosineAnnealingWarmRestarts_period_multiplier"], eta_min=self.params["scheduler_CosineAnnealingWarmRestarts_min_lr"])
    
    def get_lr(self,optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']