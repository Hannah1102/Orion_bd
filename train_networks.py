# train_networks.py
# For training CNNs and SDNs via IC-only and SDN-training strategies
# It trains and save the resulting models to an output directory specified in the main function

import argparse
import copy
import torch
import time
import os
import random
import numpy as np

import aux_funcs  as af
import network_architectures as arcs

from architectures.CNNs.VGG import VGG
from architectures.CNNs.ResNet import ResNet

def train(models_path, untrained_models, sdn=False, ic_only_sdn=False, device='cpu', clean_ratio=1.0):
    print('Training models...')

    for base_model in untrained_models:
        trained_model, model_params = arcs.load_model(models_path, base_model, 0)
        dataset = af.get_dataset(model_params['task'],add_trigger=True, attack_type='BadNets',clean_ratio=clean_ratio) #backdoor changed

        learning_rate = model_params['learning_rate']
        momentum = model_params['momentum']
        weight_decay = model_params['weight_decay']
        milestones = model_params['milestones']
        gammas = model_params['gammas']
        num_epochs = model_params['epochs']

        model_params['optimizer'] = 'SGD'

        if ic_only_sdn:  # IC-only training, freeze the original weights
            learning_rate = model_params['ic_only']['learning_rate']
            num_epochs = model_params['ic_only']['epochs']
            milestones = model_params['ic_only']['milestones']
            gammas = model_params['ic_only']['gammas']

            model_params['optimizer'] = 'Adam'
            
            trained_model.ic_only = True
        else:
            trained_model.ic_only = False


        optimization_params = (learning_rate, weight_decay, momentum)
        lr_schedule_params = (milestones, gammas)

        if sdn:
            if ic_only_sdn:
                optimizer, scheduler = af.get_sdn_ic_only_optimizer(trained_model, optimization_params, lr_schedule_params)
                trained_model_name = base_model+'_ic_only'

            else:
                optimizer, scheduler = af.get_full_optimizer(trained_model, optimization_params, lr_schedule_params)
                trained_model_name = base_model+'_sdn_training'

        else:
                optimizer, scheduler = af.get_full_optimizer(trained_model, optimization_params, lr_schedule_params)
                trained_model_name = base_model

        print('Training: {}...'.format(trained_model_name))
        trained_model.to(device)
        metrics = trained_model.train_func(trained_model, dataset, num_epochs, optimizer, scheduler, device=device)
        model_params['train_top1_acc'] = metrics['train_top1_acc']
        model_params['test_top1_acc'] = metrics['test_top1_acc']
        model_params['train_top5_acc'] = metrics['train_top5_acc']
        model_params['test_top5_acc'] = metrics['test_top5_acc']
        model_params['test_bd_top1_acc'] = metrics['test_bd_top1_acc'] 
        model_params['test_bd_top5_acc'] = metrics['test_bd_top5_acc'] 
        model_params['epoch_times'] = metrics['epoch_times'] 
        model_params['lrs'] = metrics['lrs']
        total_training_time = sum(model_params['epoch_times'])
        model_params['total_time'] = total_training_time
        print('Training took {} seconds...'.format(total_training_time))
        arcs.save_model(trained_model, model_params, models_path, trained_model_name, epoch=-1)

# for backdoored models, load a backdoored CNN and convert it to an SDN via IC-only strategy
def sdn_ic_only_backdoored(device, args):
    if args.struct == 'vgg':
        params = arcs.create_vgg16bn(None, args.dataset, None, True)
        backdoored_cnn = VGG(params)
    else:
        params = arcs.create_resnet56(None, args.dataset, None, True) # tiny imagenet
        backdoored_cnn = ResNet(params)

    backdoored_cnn.load_state_dict(torch.load('{}/{}'.format(args.bdmodel_path, args.bdmodel_name), map_location='cpu'), strict=False)

    # convert backdoored cnn into a multi-exit network
    backdoored_sdn, sdn_params = af.cnn_to_sdn(None, backdoored_cnn, params, preloaded=backdoored_cnn) # load the CNN and convert it to a sdn
    arcs.save_model(backdoored_sdn, sdn_params, args.bdmodel_path, args.sdmodel_name, epoch=0) # save the resulting sdn

    networks = [args.sdmodel_name]

    train(args.bdmodel_path, networks, sdn=True, ic_only_sdn=True, device=device, clean_ratio=args.c_ratio)

    
def main(args):
    random_seed = af.get_random_seed()
    af.set_random_seeds()
    print('Random Seed: {}'.format(random_seed))
    device = af.get_pytorch_device()
    af.set_logger('outputs/VGG16-bn_CIFAR-10_tact_ic_only_0.1clean'.format(af.get_random_seed()))

    sdn_ic_only_backdoored(device=device, args=args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='S-Net training')
    parser.add_argument('--dataset', default='cifar10')  # gtsrb, tiny-imagenet
    parser.add_argument('--struct', default='vgg')  # resnet56
    parser.add_argument('--bdmodel_path', default=None)
    parser.add_argument('--bdmodel_name', default=None)
    parser.add_argument('--sdmodel_name', default=None)

    parser.add_argument('c_ratio', default=0.1)

    args = parser.parse_args()

    main(args)
