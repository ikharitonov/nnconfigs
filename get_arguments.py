#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(__file__).split('flexnet')[0])

import flexnet.training.definitions.Config as Config

"""
CONFIG OPTIONS

VGG16 architecture:
    VGG16_Baseline
    VGG16_Single_Layer-DefaultFlex
    VGG16_All_Layers-DefaultFlex
    VGG16_Single_Layer-Random50_50
    VGG16_All_Layers-Random50_50

ResNet34 architecture:
    ResNet34_0000
    ResNet34_####Basic-DefaultFlex
    ResNet34_####Full-DefaultFlex
    ResNet34_####Basic-Random50_50
    ResNet34_####Full-Random50_50
    (insert 1 or 0 instead each # above)
    (0 and 1 will insert a BasicBlock or a Flexible BasicBlock respectively into the model)
    (e.g. ResNet34_0001Full-Random50_50)
    (Basic and Full after #### create blocks with only second or all conv layers replaced with a Flexible Layer respectively)
    (to make the very first conv layer flexible, use e.g. ResNet34_c0000)
"""

def get_config(args):

    args_dict = get_args(args)

    if args_dict['machine'].lower()=='local':
        if args_dict['dataset'].lower()=='imagenet':
            return Config.ImageNet_Local_Config(args_dict['model'], args_dict['config'], config_filename=args_dict['config_filename'], continue_training=args_dict['continue_training'])
        elif args_dict['dataset'].lower()=='cifar10':
            return Config.CIFAR10_Local_Config(args_dict['model'], args_dict['config'], config_filename=args_dict['config_filename'], continue_training=args_dict['continue_training'])
        elif args_dict['dataset'].lower()=='imagenette':
            print("LOCAL VERSION OF IMAGENETTE TRAINING NOT IMPLEMENTED")
            exit()
    elif args_dict['machine'].lower()=='rcs':
        if args_dict['dataset'].lower()=='imagenet':
            return Config.ImageNet_RCS_Config(args_dict['model'], args_dict['config'], config_filename=args_dict['config_filename'], continue_training=args_dict['continue_training'])
        elif args_dict['dataset'].lower()=='cifar10':
            return Config.CIFAR10_RCS_Config(args_dict['model'], args_dict['config'], config_filename=args_dict['config_filename'], continue_training=args_dict['continue_training'])
        elif args_dict['dataset'].lower()=='imagenette':
            return Config.ImageNette_RCS_Config(args_dict['model'], args_dict['config'], config_filename=args_dict['config_filename'], continue_training=args_dict['continue_training'])

def get_args(args):

    out_dict = {}

    out_dict['machine'] = 'local'
    out_dict['dataset'] = 'cifar10'
    out_dict['model'] = 'VGG16_Baseline'
    out_dict['config'] = 'config1'
    out_dict['config_filename'] = ''
    out_dict['continue_training'] = False

    if len(args)==1:
        return out_dict

    if len(args) > 1:
        if args[1]=='--help':
            print("PARAMERES:")
            print("--machine : \"local\" (default) or \"rcs\"")
            print("--dataset : \"imagenet\", \"cifar10\" (default) or \"imagenette\"")
            print("--model : for list of models use option \"--modelhelp\" ; default is \"VGG16_Baseline\"")
            print("--config : training config e.g. \"config1\" (default) or \"1\"")
            print("--config_filename : name of the file to load training configuration from e.g. \"config1\" (default) or \"config_baseline_monday\"")
            print("--continue : option to continue training \"false\" (default) or \"true\"")
            print("EXAMPLE: python imagenet_amp.py --machine local --dataset imagenet --model ResNet34_0000 --config 1 --config_filename baseline_c --continue false")
            exit()
        elif args[1]=='--modelhelp':
            print("CONFIG OPTIONS\n\nVGG16 architecture:\n\tVGG16_Baseline\n\tVGG16_Single_Layer-DefaultFlex\n\tVGG16_All_Layers-DefaultFlex\n\tVGG16_Single_Layer-Random50_50\n\tVGG16_All_Layers-Random50_50\n\nResNet34 architecture:\n\tResNet34_0000\n\tResNet34_####Basic-DefaultFlex\n\tResNet34_####Full-DefaultFlex\n\tResNet34_####Basic-Random50_50\n\tResNet34_####Full-Random50_50\n\t(insert 1 or 0 instead each # above)\n\t(0 and 1 will insert a BasicBlock or a Flexible BasicBlock respectively into the model)\n\t(e.g. ResNet34_0001Full-Random50_50)\n\t(Basic and Full after #### create blocks with only second or all conv layers replaced with a Flexible Layer respectively)\n\t(to make the very first conv layer flexible, use e.g. ResNet34_c0000)")
            exit()
        elif args[1]=='--create_config_file':
            """
            TODO: create txt file with one of the default configs
            """

    for i in range(1,len(args),2):
        if args[i]=='--machine':
            machine = args[i+1]
            out_dict['machine'] = machine
        elif args[i]=='--dataset':
            dataset = args[i+1]
            out_dict['dataset'] = dataset
        elif args[i]=='--model':
            model = args[i+1]
            out_dict['model'] = model
        elif args[i]=='--config':
            if len(args[i+1])==1:
                config = 'config'+args[i+1]
            else:
                config = args[i+1]
            out_dict['config'] = config
        elif args[i]=='--config_filename':
            config_filename = args[i+1]
            out_dict['config_filename'] = config_filename
        elif args[i]=='--continue':
            continue_training = args[i+1].lower()
            if continue_training=='false': continue_training = False
            elif continue_training=='true': continue_training = True
            out_dict['continue_training'] = continue_training
    
    return out_dict