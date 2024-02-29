from nnconfigs.Config import BaseConfig
from pathlib import Path

class ExampleConfig(BaseConfig):
    def __init__(self, cli_args=None, model_name=None, dataset_name=None, configuration_name=None, configuration_file=None, continue_training=False):

        self.base_path = Path('/path/to/model/configurations/and/weights/folder')
        
        super().__init__(cli_args, model_name, dataset_name, configuration_name, configuration_file, continue_training)

        data_path = Path('/path/to/data/folder')
        self.dirs = {
            "training": data_path / 'training_data.npy',
            "testing": data_path / 'testing_data.npy'
        }
        self.weights_save_dir = self.base_path / self.training_name