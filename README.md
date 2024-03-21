### __nnconfigs__

__nnconfigs__ is a small tool for simplifying PyTorch training scripts. It adds support for parametrising training configurations through simple text files, collection of training metrics and continuous saving of checkpoints.

### How to install

Clone repository:
```python
git clone https://github.com/ikharitonov/nnconfigs.git
```
Activate the target conda environment:
```python
conda activate my_env
```
Change directory:
```python
cd nnconfigs
```
Install with pip:
```python
pip install -e .
```

### How to use

The main way to use __nnconfigs__ is through the command line, e.g.:
```python
python example_training.py --model_name Model1 --dataset_name Dataset1 --configuration_name=config1 --configuration_file=/path/to/config1.txt --continue_training False
```
For more info, type ```python example_training.py --help```. For the necessary format of the training script, see the provided _example\_training.py_ file.

Of course, scripts can also be run through an IDE by swapping
```python
config = ExampleConfig(cli_args=sys.argv)
```
with
```python
config = ExampleConfig(model_name="Model1", dataset_name="Dataset1", configuration_name="config1", configuration_file="/path/to/config1.txt" continue_training=False)
```

Lastly, you must create two additional files. First is a _.py_ file with a custom config class, extending the _nnconfigs.Config.BaseConfig_. By default, it should define the directories for data, default configurations and model checkpoints, but any other custom functionality may be included (see Config.py to see what's already implemented). Second is a _.txt_ file with the default values for all parameters used in your training script/s. In case new parameters are added, ensure that the listing in this file is complete.

TL;DR: make sure your training script looks like _example\_training.py_ and create your own custom copies of _ExampleConfig.py_ and _example\_config.txt_.

### Extra: batch execution

This is an additional functionality implemented in two Bash scripts: _batch\_script.sh_ and _start\_and\_monitor.sh_. The main idea behind this is automating long batched training tasks (e.g. exploration of the hyperparameter space). All you need to do is prepare a folder with _.txt_ training configurations (just as _example\_config.txt_), a _.csv_ file with the necessary information to execute each training script and launch the batch script, specifying where to find the _.csv_ file, where to put the temporary logs, which anaconda environment to use and the maximum number of trainings to run concurrently at a given time.

Here is how to do this in practice. 

First, create a folder with all the training configurations you want to run:
```python
ls /path/to/configurations_folder
```
```python
config1.txt
config2.txt
config3.txt
config4.txt
```

Create a _.csv_ file in the following format:
```python
cat /path/to/batch_config_info.csv
```
```python
training_script_full_path,model_name,dataset_name,configuration_name,configuration_file_full_path,continue_training
/path/to/training_script1.py,MyCNNModel,MNIST_dataset,config1,/path/to/configurations_folder/config1.txt,False
/path/to/training_script1.py,ResNet-34,CIFAR10_dataset,config2,/path/to/configurations_folder/config2.txt,False
/path/to/training_script2.py,RNN,dataset2,config3,/path/to/configurations_folder/config3.txt,False
/path/to/training_script2_2.py,LSTM,dataset2,config4,/path/to/configurations_folder/config4.txt,False
```
Keep in mind that the first line of this file (the field names) is for your convenience, but shall not be removed because the batch script reads starting from the second line.

Before running the batch script, make sure that you are inside of __nnconfigs__ directory and that both scripts is executable:
```python
cd /path/to/nnconfigs
chmod +x batch_script.sh
chmod +x start_and_monitor.sh
```

_batch\_script.sh_ requires several arguments:
```python
-f is the full path to the .csv file described above
-l is the full path to directory where temporary logs will be saved
-m is the maximum number of concurrently run jobs
-e is the anaconda environment to use
```

Then, just run the batch script:
```python
./batch_script.sh -f /path/to/batch_config_info.csv -l ~/Desktop -m 2 -e my_env
```

If you want to launch it as a background process:
```python
nohup ./batch_script.sh -f /path/to/batch_config_info.csv -l ~/Desktop -m 2 -e my_env &
```

### Useful features

- Streamlined ingestion and training on many configurations (sets of parameters)
- Flexibility of launching through the command line
- Tidy directory structure of training outputs
- Training metrics are collected into Pandas-friendly files
- Support for continuing training from where you left off (or something crashed)
- Batch running of training scripts
