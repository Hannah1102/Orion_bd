# confusion_experiments.py
# model_confusion_experiment() runs the experiments in section 5.2
# it measures confusion for the sdn for correct and wrong predictions at the final layer
# for cnn it measure the softmax confidence scores
# it generates histograms and outputs the comparison

import torch
import numpy as np
import random
from collections import Counter

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

    dataset = af.get_dataset(sdn_params['task'], batch_size=128, add_trigger=trigger_test, attack_type='feature')

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
    
    ### get layer confidence
    layer_correct, _, layer_predictions, _, _ = mf.sdn_get_detailed_results(sdn_model, loader=dataset.test_loader, device=device)
    layer_correct_bd, _, layer_predictions_bd, _, _ = mf.sdn_get_detailed_results(sdn_model, loader=dataset.trigger_test_loader, device=device)
    # layer_correct[output_id]: correctly identified sample_id
    # layer_wrong[output_id]: wrongly identified sample_id
    # layer_predictions[output_id][sample_id]: predicted label
    # layer_confidence[output_id][sample_id]: predicted confidence(max)
    # layer_true_confidence[output_id][sample_id]: predicted confidence(true label)

    # get both true samples for clean&poisoned
    clean_true = layer_correct[6] #set
    bad_true = layer_correct_bd[6] #set
    filter_set = clean_true & bad_true
    print("total test number: ", len(filter_set))

    # cifar10: 10000
    for i in range(3):
        test_sample = np.random.choice(list(filter_set), 500)
        recover = 0
        for idx in test_sample:
            label_set = []
            label_count = {}
            for output_id in range(sdn_model.num_output):
                cur_pre = layer_predictions_bd[output_id][idx][0]
                if cur_pre != layer_predictions_bd[6][idx][0]:
                    label_set.append(cur_pre)
            if label_set:
                # print('label_set: ', label_set)
                # print('ground-truth: ', clean_label[idx], "predict: ", layer_predictions_bd[6][idx])
                recover_label = max(Counter(label_set))
                if recover_label == layer_predictions[6][idx][0]:
                    recover += 1
            # else:
            #     print("Empty label set!")
        # print("recover number: ", recover)
        print("recover rate: ", recover/500)  # add 0.1 to include the target class



        
    
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