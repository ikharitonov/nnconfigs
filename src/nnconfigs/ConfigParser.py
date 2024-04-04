#!/usr/bin/env python3
import os

class ConfigParser:
    def __init__(self, configuration_file, config):

        self.configuration_file = configuration_file
        self.config = config
        self.training_parameters = {}
        self.loading_defaults = False

        if self.config.continue_training:
            self.configuration_file = self.config.weights_save_dir / self.config.configuration_name / f"{self.config.configuration_name}.txt"
            print("ConfigParser: Loading configuration from " + self.configuration_file.name)

        if not self.configuration_file:
            print("ConfigParser: Configuration file was not provided. Using default configuration for \""+self.config.training_name+"\".")
            self.loading_defaults = True
            self.load_default_config()
        elif not os.path.isfile(self.configuration_file):
            print("ConfigParser: Configuration file with provided name was not found. Please make sure file exists.")
            exit()
        else:
            self.load_default_config() # to ensure no parameters are missing - using default values
            self.parse_txt_contents(self.configuration_file) # updating training_parameters dictionary with the relevant parameters
    
    def load_default_config(self):
        self.parse_txt_contents(self.config.default_config_path)

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
            elif '.' in line_elements[1] or 'e-' in line_elements[1]:
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
        print("ConfigParser: Saved configuration file \"" + f"{self.config.configuration_name}.txt" + "\" into folder " + new_filepath.name+ " successfully.")
        if not self.loading_defaults:
            os.remove(self.configuration_file)
            print("ConfigParser: Configuration file removed from " + self.config.base_path.name + " successfully.")