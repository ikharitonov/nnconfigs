from Config import BaseConfig
from pathlib import Path

# class CustomConfig(BaseConfig):
#     def __init__(self, model_type, configuration_name, configuration_filename='', continue_training=False):

#         super(BaseConfig,self).__init__(model_type, configuration_name, continue_training)

#         # self.name = 'ImageNet_Local_Config'
#         # self.running_machine = "Local"
#         self.base_path = "C:\\Users\\sarfi\\Desktop"
#         self.base_data_path = "C:\\Users\\sarfi\\Desktop\\data"
#         self.slash = "\\"
#         self.dirs = {"training": self.base_data_path + "\\imagenet\\test_train",
#                     "testing": self.base_data_path + "\\imagenet\\test_val"}
#         self.weights_dataset_name = "-ImageNet_Weights"
#         self.weights_save_dir = self.base_path + self.slash + self.model_type + self.weights_dataset_name + self.slash
#         self.parser = ConfigParser.ConfigParser(configuration_filename, self)
#         self.training_parameters = self.parser.training_parameters

class SNNCustomConfig(BaseConfig):
    def __init__(self, model_name, dataset_name, configuration_name, configuration_file=None, continue_training=False):

        training_name = f"{model_name}_{dataset_name}"
        super(BaseConfig,self).__init__(training_name, configuration_name, configuration_file, continue_training)

        self.base_path = Path.home() / 'RANCZLAB-NAS/iakov/produced/mnist_sequence_checkpoints'
        data_path = Path.home() / 'RANCZLAB-NAS/produced/'
        self.dirs = {
            "training": data_path / f'{dataset_name}.npy'
        }
        self.weights_save_dir = self.base_path / training_name