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

### Useful features

- Streamlined ingestion and training on many configurations (sets of parameters)
- Flexibility of launching through the command line
- Tidy directory structure of training outputs
- Training metrics are collected into Pandas-friendly files
- Support for continuing training from where you left off (or something crashed)
