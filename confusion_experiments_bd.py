# confusion_experiments.py
# model_confusion_experiment() runs the experiments in section 5.2
# it measures confusion for the sdn for correct and wrong predictions at the final layer
# for cnn it measure the softmax confidence scores
# it generates histograms and outputs the comparison

import torch
import numpy as np
import random

import aux_funcs  as af
import model_funcs as mf
import network_architectures as arcs
from architectures.CNNs.VGG import VGG

def get_sdn_stats(layer_correct, layer_wrong, instance_confusion):
    layer_keys = sorted(list(layer_correct.keys()))

    correct_confusion = []
    wrong_confusion = []
    confused_list = []

    #print("confusion list: ", instance_confusion)

    for inst in layer_correct[layer_keys[-1]]:   # correctly classified at the last layer
        #if instance_confusion[inst] <= 1.0:
        #    confused_list.append(inst)
        correct_confusion.append(instance_confusion[inst])
        
    for inst in layer_wrong[layer_keys[-1]]:   # wrongly classified at the last layer
        wrong_confusion.append(instance_confusion[inst])

    mean_correct_confusion = np.mean(correct_confusion)
    mean_wrong_confusion = np.mean(wrong_confusion)

    print('Confusion of corrects: {}, Confusion of wrongs: {}'.format(mean_correct_confusion, mean_wrong_confusion))

    return correct_confusion, wrong_confusion  #, confused_list

def get_cnn_stats(correct, wrong, instance_confidence):
    print('get cnn stats')

    correct_confidence = []
    wrong_confidence = []

    for inst in correct:
        correct_confidence.append(instance_confidence[inst])
    for inst in wrong:
        wrong_confidence.append(instance_confidence[inst])

    mean_correct_confidence = np.mean(correct_confidence)
    mean_wrong_confidence = np.mean(wrong_confidence)

    print('Confidence of corrects: {}, Confidence of wrongs: {}'.format(mean_correct_confidence, mean_wrong_confidence))
    return correct_confidence, wrong_confidence


def model_confusion_experiment(models_path, device='cpu', trigger_test=False):

    sdn_name = 'ckpt_epoch_120_sdn_ic_only'
    cnn_name = '120.pth'

    print("sdn_name: ", sdn_name)
    print("cnn_name: ", cnn_name)

    sdn_model, sdn_params = arcs.load_model(models_path, sdn_name, epoch=-1)
    sdn_model.to(device)
    dataset = af.get_dataset(sdn_params['task'], add_trigger=trigger_test, attack_type='feature')

    params = arcs.create_vgg16bn(None, 'cifar10', None, True)
    cnn_model = VGG(params)
    cnn_model.load_state_dict(torch.load('{}/{}'.format(models_path, cnn_name), map_location='cpu'), strict=False) 

    cnn_model.to(device)

    top1_test, top5_test = mf.sdn_test(sdn_model, dataset.test_loader, device)
    top1_test_bd, top5_test_bd = mf.sdn_test(sdn_model, dataset.trigger_test_loader, device)
    print('SDN Top1 Test accuracy: {}'.format(top1_test))
    print('SDN Top5 Test accuracy: {}'.format(top5_test))
    print('SDN bd Top1 Test accuracy: {}'.format(top1_test_bd))
    print('SDN bd Top5 Test accuracy: {}'.format(top5_test_bd))

    top1_test, top5_test = mf.cnn_test(cnn_model, dataset.test_loader, device)
    top1_test_bd, top5_test_bd = mf.cnn_test(cnn_model, dataset.trigger_test_loader, device)
    print('CNN Top1 Test accuracy: {}'.format(top1_test))
    print('CNN Top5 Test accuracy: {}'.format(top5_test))
    print('CNN bd Top1 Test accuracy: {}'.format(top1_test_bd))
    print('CNN bd Top5 Test accuracy: {}'.format(top5_test_bd))
    
    # the the normalization stats from the training set
    confusion_stats = None
    # SETTING 1 - IN DISTRIBUTION
    sdn_layer_correct, sdn_layer_wrong, sdn_layer_changed, sdn_instance_confusion = mf.sdn_get_confusion(sdn_model, loader=dataset.test_loader, confusion_stats=confusion_stats, device=device)
    sdn_layer_correct_bd, sdn_layer_wrong_bd, sdn_layer_changed_bd, sdn_instance_confusion_bd = mf.sdn_get_confusion(sdn_model, loader=dataset.trigger_test_loader, confusion_stats=confusion_stats, device=device)
    
    # sdn_layer_correct[layer]: correct instance id
    # sdn_layer_wrong[layer]: wrong instance id
    # sdn_instance_confusion[instance_id]: confusion score
    sdn_correct_confusion, sdn_wrong_confusion = get_sdn_stats(sdn_layer_correct, sdn_layer_wrong, sdn_instance_confusion) #sdn_instance_confusion 
    sdn_correct_confusion_bd, sdn_wrong_confusion_bd = get_sdn_stats(sdn_layer_correct_bd, sdn_layer_wrong_bd, sdn_instance_confusion_bd) #sdn_instance_confusion_bd
    # sdn_correct/wrong_confusion: correctly/wrongly classified samples' confusion list

    for i in range(3):

        random.shuffle(sdn_correct_confusion)
        random.shuffle(sdn_correct_confusion_bd)

        test_num = 500 #realbd:300 others:500 
        clean_test = sdn_correct_confusion[:test_num]
        clean_test_2 = sdn_correct_confusion[-test_num:]
        bd_test = sdn_correct_confusion_bd[:test_num]

        clean_set_sort = np.sort(clean_test,0)
        percentile_95 = clean_set_sort[int(len(clean_set_sort)*0.95)]
        percentile_99 = clean_set_sort[int(len(clean_set_sort)*0.99)]
        print(f'percentile_95:{percentile_95}')
        print(f'percentile_99:{percentile_99}')

        true_positive_95 =  np.where(bd_test>percentile_95, 1, 0).sum()
        #false_negative = test_num - true_positive
        false_positive_95 = np.where(clean_test_2>percentile_95, 1, 0).sum()
        true_positive_99 =  np.where(bd_test>percentile_99, 1, 0).sum()
        false_positive_99 = np.where(clean_test_2>percentile_99, 1, 0).sum()
        
        REC_95 = true_positive_95 / test_num #(true_positive + false_negative)
        PRE_95 = true_positive_95 / (true_positive_95 + false_positive_95)
        REC_99 = true_positive_99 / test_num #(true_positive + false_negative)
        PRE_99 = true_positive_99 / (true_positive_99 + false_positive_99)
        F1_95 = 2*PRE_95*REC_95/(PRE_95+REC_95)
        F1_99 = 2*PRE_99*REC_99/(PRE_99+REC_99)

        print(f'TP_95:{true_positive_95},FP_95:{false_positive_95}')
        print(f'TP_99:{true_positive_99},FP_99:{false_positive_99}')
        print(f'REC_95:{REC_95},PRE_95:{PRE_95}')
        print(f'REC_99:{REC_99},PRE_99:{PRE_99}')
        print(f'F1_95:{F1_95},F1_99:{F1_99}')

    
def main():
    torch.manual_seed(af.get_random_seed())    # reproducible
    np.random.seed(af.get_random_seed())
    device = af.get_pytorch_device()
    trained_models_path = 'experiments/feature_bd'
    #af.set_logger('outputs/confusion_diff_layers'.format(af.get_random_seed()))
    #af.set_logger('outputs/model_keys')

    trigger_test = True
    model_confusion_experiment(trained_models_path, device, trigger_test)

if __name__ == '__main__':
    main()
