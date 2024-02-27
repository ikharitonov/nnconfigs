#!/usr/bin/env python3
import os

class ValidParameters:
    """
    TODO: implement checking against a predefined list of accepted parameters e.g. StepLR / WarmupCosineLR / CosineAnnealingWarmRestarts

    reference:
    self.training_parameters = {"image_dim": 224,
                                        "input_channels": 3,
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
                                        "scheduler_type": "WarmupCosineLR", # StepLR or WarmupCosineLR or CosineAnnealingWarmRestarts
                                        "scheduler_StepLR_step_size": 5,
                                        "scheduler_StepLR_gamma": 0.7,
                                        "scheduler_WarmupCosineLR_warmup_factor": 0.001,
                                        "scheduler_WarmupCosineLR_warmup_iterations": 3,
                                        "scheduler_CosineAnnealingWarmRestarts_restart_period": 10,
                                        "scheduler_CosineAnnealingWarmRestarts_period_multiplier": 1,
                                        "scheduler_CosineAnnealingWarmRestarts_min_lr": 0,
                                        "step_scheduler_per": "epoch"} # batch or epoch

    """
    pass

class ConfigParser:
    def __init__(self, configuration_file, config):
        """
        TODO: when continuing training and trying to load from configN.txt file:
            1) if it's not there, try to load them from metrics
            2) if not all config parameters are present, load the rest from default configs
        """
        self.configuration_file = configuration_file
        self.config = config
        self.training_parameters = {}
        self.loading_defaults = False

        if self.config.continue_training:
            self.configuration_file = self.config.weights_save_dir / self.config.configuration_name / f"{self.config.configuration_name}.txt"
            print("ConfigParser: Loading configuration from " + self.configuration_file)

        if not self.configuration_file:
            print("ConfigParser: Configuration file was not provided. Using default configuration for \""+self.config.name+"\".")
            self.loading_defaults = True
            raise NotImplementedError('Default configurations are not generally implemented.')
            # self.load_default_config()
        elif not os.path.isfile(self.configuration_file):
            print("ConfigParser: Configuration file with provided name was not found. Using default configuration for \""+self.config.name+"\".")
            self.loading_defaults = True
            raise NotImplementedError('Default configurations are not generally implemented.')
            # self.load_default_config()
        else:
            self.parse_txt_contents(self.configuration_file)
    
    def load_default_config(self):
        self.parse_txt_contents(os.path.dirname(__file__) + self.config.slash + 'DefaultConfigs' + self.config.slash + self.config.name + '.txt')

    def parse_txt_contents(self,path):
        lines = []
        with open(path) as file:
            lines = [line.rstrip().replace(" ", "") for line in file]
        while '' in lines: lines.remove('')
        for l in lines:
            line_elements = l.split(':')
            value = None
            if line_elements[1][0] == '"':
                value = line_elements[1].split('"')[1]
            elif '.' in line_elements[1]:
                value = float(line_elements[1])
            elif line_elements[1] in ['true','True','"true"','"True"']:
                value = True
            elif line_elements[1] in ['false','False','"false"','"False"']:
                value = False
            else:
                value = int(line_elements[1])
            self.training_parameters[line_elements[0]] = value
    
    def write_to_txt(self,path):
        with open(path, 'a') as file:
            for key, value in self.training_parameters.items():
                if type(value)==type('str'):
                    file.write('%s:"%s"\n' % (key, value))
                else:
                    file.write('%s:%s\n' % (key, value))

    def transfer_config_file(self, new_filepath):
        path = new_filepath / f"{self.config.configuration_name}.txt"
        self.write_to_txt(path)
        if not os.path.isfile(path):
            print("ConfigParser: Could not save configuration file \"" + f"{self.config.configuration_name}.txt" + "\" into folder " + new_filepath)
            exit()
        print("ConfigParser: Saved configuration file \"" + f"{self.config.configuration_name}.txt" + "\" into folder " + new_filepath + " successfully.")
        if not self.loading_defaults:
            os.remove(self.configuration_file)
            print("ConfigParser: Configuration file removed from " + self.config.base_path + " successfully.")