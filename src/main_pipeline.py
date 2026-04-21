import os
import sys
import numpy as np
import time

import random
import logging
import argparse
import torch
import tensorflow as tf
import copy
# import torch.nn as nn
# import torchvision.transforms as transforms
# from torchvision import datasets
# import torch.utils
# import torch.backends.cudnn as cudnn
# from tensorboardX import SummaryWriter

from load.LoadConfigs import * #load_configs
from load.LoadParty import load_parties
from evaluates.MainTaskVFL import *
from evaluates.MainTaskVFLwithBackdoor import *
from evaluates.MainTaskVFLwithNoisySample import *
from utils.basic_functions import append_exp_res
import warnings
warnings.filterwarnings("ignore")

TARGETED_BACKDOOR = ['ReplacementBackdoor','ASB'] # main_acc  backdoor_acc
UNTARGETED_BACKDOOR = ['NoisyLabel','MissingFeature','NoisySample','PGD'] # main_acc
LABEL_INFERENCE = ['BatchLabelReconstruction','DirectLabelScoring','NormbasedScoring',\
'DirectionbasedScoring','PassiveModelCompletion','ActiveModelCompletion']
ATTRIBUTE_INFERENCE = ['AttributeInference']
FEATURE_INFERENCE = ['GenerativeRegressionNetwork','ResSFL','CAFE']
ADI_SYNTHESIS = ['GradientBasedADI']
CONTRIBUTION_ANALYSIS = ['LOCO']


def resolve_checkpoint_path(args):
    checkpoint_path = getattr(args, "checkpoint", None)
    if checkpoint_path:
        return checkpoint_path
    return os.path.join(
        args.exp_res_dir,
        "trained_models",
        f"parties{args.k}_epoch{args.main_epochs}_seed{args.current_seed}.pth",
    )


def load_saved_models_into_vfl(vfl, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state_dicts = checkpoint["model_state_dicts"]
    for ik in range(vfl.k):
        vfl.parties[ik].prepare_data_loader(batch_size=vfl.batch_size)
        vfl.parties[ik].local_model.load_state_dict(model_state_dicts[ik])
        vfl.parties[ik].local_model.to(device).eval()
    vfl.parties[vfl.k - 1].global_model.load_state_dict(model_state_dicts[vfl.k])
    vfl.parties[vfl.k - 1].global_model.to(device).eval()
    return checkpoint


def evaluate_main_task_accuracy(vfl, attack_on="test"):
    for ik in range(vfl.k):
        vfl.parties[ik].local_model.eval()
    vfl.parties[vfl.k - 1].global_model.eval()

    correct = 0
    total = 0
    data_loader_list = [vfl.parties[ik].train_loader if attack_on == "train" else vfl.parties[ik].test_loader for ik in range(vfl.k)]
    with torch.no_grad():
        for parties_data in zip(*data_loader_list):
            inputs = [parties_data[ik][0].to(vfl.device) for ik in range(vfl.k)]
            labels = parties_data[vfl.k - 1][1].to(vfl.device)
            logits = vfl.parties[vfl.k - 1].global_model(
                [vfl.parties[ik].local_model(inputs[ik]) for ik in range(vfl.k)]
            )
            pred = torch.argmax(logits, dim=-1)
            if labels.dim() > 1:
                labels = torch.argmax(labels, dim=-1)
            correct += (pred == labels).sum().item()
            total += labels.numel()
    return correct / max(total, 1)


def evaluate_saved_model_test_only(args):
    set_seed(args.current_seed)
    args.need_auxiliary = 0
    args = load_attack_configs(args.configs, args, -1)
    args = load_parties(args)

    checkpoint_path = resolve_checkpoint_path(args)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    vfl = MainTaskVFL(args)
    load_saved_models_into_vfl(vfl, checkpoint_path, args.device)
    print(f"loaded checkpoint from {checkpoint_path}")

    clean_acc = evaluate_main_task_accuracy(vfl, attack_on="test")
    print(f"clean test accuracy from checkpoint: {clean_acc}")

    attack_indices = (
        list(args.untargeted_backdoor_index)
        + list(getattr(args, "adi_synthesis_index", []))
        + list(getattr(args, "contribution_analysis_index", []))
    )
    if attack_indices == []:
        return

    for index in attack_indices:
        attack_args = copy.deepcopy(args)
        attack_args = load_attack_configs(attack_args.configs, attack_args, index)
        print('======= Test Attack',index,': ',attack_args.attack_name,' =======')
        print('attack configs:',attack_args.attack_configs)

        if attack_args.attack_name not in ['PGD', 'GradientBasedADI', 'LOCO']:
            print(f"test-only mode currently supports PGD, GradientBasedADI, and LOCO evaluation only, skipping {attack_args.attack_name}")
            continue

        vfl.args = attack_args
        attack_result = vfl.evaluate_attack()
        if attack_args.attack_name == 'GradientBasedADI':
            attack_metric_name = 'domination_success_rate@95%'
            attack_metric = attack_result['domination_success_rate@95%']
            exp_result = f"K|bs|LR|num_class|Q|top_trainable|epoch|attack_name|{attack_args.attack_param_name}|clean_acc|clean_dom95_count|clean_dom95_rate|attacked_acc|target_success|dom95|dom99|avg_domination,%d|%d|%lf|%d|%d|%d|%d|{attack_args.attack_name}|{attack_args.attack_param}|{attack_result['clean_accuracy']}|{attack_result['clean_input_domination_count@95%']}|{attack_result['clean_input_domination_success_rate@95%']}|{attack_result['attacked_accuracy']}|{attack_result['target_success_rate']}|{attack_result['domination_success_rate@95%']}|{attack_result['domination_success_rate@99%']}|{attack_result['avg_domination_percentage']}" %\
                (attack_args.k,attack_args.batch_size, attack_args.main_lr, attack_args.num_classes, attack_args.Q, attack_args.apply_trainable_layer,attack_args.main_epochs)
        elif attack_args.attack_name == 'LOCO':
            header = f"K|bs|LR|num_class|Q|top_trainable|epoch|attack_name|{attack_args.attack_param_name}|clean_acc|party_id|acc_wo_party|acc_drop|pred_change|logit_shift|true_prob_drop|contribution_score"
            append_exp_res(attack_args.exp_res_path, header)
            for party_metric in attack_result["party_metrics"]:
                exp_result = f"%d|%d|%lf|%d|%d|%d|%d|{attack_args.attack_name}|{attack_args.attack_param}|{attack_result['clean_accuracy']}|{party_metric['party_id']}|{party_metric['accuracy_without_party']}|{party_metric['accuracy_drop']}|{party_metric['prediction_change_rate']}|{party_metric['mean_logit_shift']}|{party_metric['mean_true_class_prob_drop']}|{party_metric['contribution_score']}" %\
                    (attack_args.k,attack_args.batch_size, attack_args.main_lr, attack_args.num_classes, attack_args.Q, attack_args.apply_trainable_layer,attack_args.main_epochs)
                print(exp_result)
                append_exp_res(attack_args.exp_res_path, exp_result)
            continue
        else:
            attack_metric = clean_acc - attack_result
            attack_metric_name = 'adv_acc_loss'
            exp_result = f"K|bs|LR|num_class|Q|top_trainable|epoch|attack_name|{attack_args.attack_param_name}|main_task_acc|{attack_metric_name},%d|%d|%lf|%d|%d|%d|%d|{attack_args.attack_name}|{attack_args.attack_param}|{clean_acc}|{attack_metric}" %\
                (attack_args.k,attack_args.batch_size, attack_args.main_lr, attack_args.num_classes, attack_args.Q, attack_args.apply_trainable_layer,attack_args.main_epochs)
        print(exp_result)
        append_exp_res(attack_args.exp_res_path, exp_result)

def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def evaluate_no_attack(args):
    # No Attack
    set_seed(args.current_seed)

    vfl = MainTaskVFL(args)
    if args.dataset not in ['cora']:
        
        main_acc , stopping_iter, stopping_time, stopping_commu_cost= vfl.train()
    else:
        main_acc, stopping_iter, stopping_time = vfl.train_graph()

    main_acc_noattack = main_acc
    attack_metric = main_acc_noattack - main_acc
    attack_metric_name = 'acc_loss'
    # Save record 
    exp_result = f"K|bs|LR|num_class|Q|top_trainable|epoch|attack_name|{args.attack_param_name}|main_task_acc|{attack_metric_name},%d|%d|%lf|%d|%d|%d|%d|{args.attack_name}|{args.attack_param}|{main_acc}|{attack_metric}" %\
        (args.k,args.batch_size, args.main_lr, args.num_classes, args.Q, args.apply_trainable_layer,args.main_epochs)
    print(exp_result)
    append_exp_res(args.exp_res_path, exp_result)
    append_exp_res(args.exp_res_path, f"==stopping_iter:{stopping_iter}==stopping_time:{stopping_time}==stopping_commu_cost:{stopping_commu_cost}")
    
    return vfl, main_acc_noattack

def evaluate_feature_inference(args):
    for index in args.feature_inference_index:
        torch.cuda.empty_cache()
        vfl = None

        set_seed(args.current_seed)
        args = load_attack_configs(args.configs, args, index)
        print('======= Test Attack',index,': ',args.attack_name,' =======')
        print('attack configs:',args.attack_configs)

        if args.attack_name == 'ResSFL':
            args.need_auxiliary = 1
            args = load_parties(args)

            vfl = MainTaskVFL(args)
            if args.dataset not in ['cora']:
                main_acc, stopping_iter = vfl.train()
            else:
                main_acc = vfl.train_graph()
                main_acc = args.main_acc_noattack_withaux 
                vfl = args.basic_vfl_withaux 
            args.main_acc_noattack_withaux = main_acc
            args.basic_vfl_withaux = vfl
        
        else: # GRN
            args.need_auxiliary = 0
            args = load_parties(args)
            vfl = MainTaskVFL(args)
            if args.dataset not in ['cora']:
                main_acc , stopping_iter, stopping_time, stopping_commu_cost= vfl.train()
            else:
                main_acc = vfl.train_graph()

        rand_mse,mse = vfl.evaluate_attack()
        attack_metric_name = 'mse_reduction'
        
        # Save record for different defense method
        exp_result = f"K|bs|LR|num_class|Q|top_trainable|epoch|attack_name|{args.attack_param_name}|main_task_acc|{attack_metric_name},%d|%d|%lf|%d|%d|%d|%d|{args.attack_name}|{args.attack_param}|{main_acc}|{rand_mse}|{mse}" %\
            (args.k,args.batch_size, args.main_lr, args.num_classes, args.Q, args.apply_trainable_layer,args.main_epochs)
        print(exp_result)
        append_exp_res(args.exp_res_path, exp_result)

def evaluate_label_inference(args):
    # Basic VFL Training Pipeline
    i=0

    for index in args.label_inference_index:
        set_seed(args.current_seed)
        args = load_attack_configs(args.configs, args, index)
        # args = load_parties(args)
        print('======= Test Attack',index,': ',args.attack_name,' =======')
        print('attack configs:',args.attack_configs)
        if args.attack_name == 'PassiveModelCompletion':
            ############### v1: train and auxiliary do not intersect ###############
            # args.need_auxiliary = 1
            # args = load_parties(args) # include load dataset with auxiliary data
            # # actual train = train-aux
            # if args.basic_vfl_withaux == None:
            #     vfl = MainTaskVFL(args)
            #     if args.dataset not in ['cora']:
            #         main_acc = vfl.train()
            #     else:
            #         main_acc = vfl.train_graph()
            # else:
            #     main_acc = args.main_acc_noattack_withaux 
            #     vfl = args.basic_vfl_withaux
            # args.main_acc_noattack_withaux = main_acc
            # args.basic_vfl_withaux = vfl
            ############### v1: train and auxiliary do not intersect ###############

            ############### v2: auxiliary is from train (like original code) ###############
            args.need_auxiliary = 0
            args = load_parties(args) # include load dataset with auxiliary data
            # actual train = train
            vfl = args.basic_vfl
            main_acc = args.main_acc_noattack
            ############### v2: auxiliary is from train (like original code) ###############

            attack_metric = vfl.evaluate_attack()
            attack_metric_name = 'label_recovery_rate'

            # Save record for different defense method
            exp_result = f"K|bs|LR|num_class|Q|top_trainable|epoch|attack_name|{args.attack_param_name}|main_task_acc|{attack_metric_name},%d|%d|%lf|%d|%d|%d|%d|{args.attack_name}|{args.attack_param}|{main_acc}|{attack_metric}" %\
                (args.k,args.batch_size, args.main_lr, args.num_classes, args.Q, args.apply_trainable_layer,args.main_epochs)
            print(exp_result)
            append_exp_res(args.exp_res_path, exp_result)

        elif args.attack_name == 'ActiveModelCompletion':
            ############### v1: train and auxiliary do not intersect ###############
            # args.need_auxiliary = 1
            ############### v1: train and auxiliary do not intersect ###############
            ############### v2: auxiliary is from train (like original code) ###############
            args.need_auxiliary = 0
            ############### v2: auxiliary is from train (like original code) ###############
            
            args = load_parties(args) # include load dataset with auxiliary data
            # actual train = train-aux
            vfl = MainTaskVFL(args)
            if args.dataset not in ['cora']:
                main_acc, stopping_iter = vfl.train()
            else:
                main_acc = vfl.train_graph()

            attack_metric = vfl.evaluate_attack()
            attack_metric_name = 'label_recovery_rate'
            # Save record for different defense method
            exp_result = f"K|bs|LR|num_class|Q|top_trainable|epoch|attack_name|{args.attack_param_name}|main_task_acc|{attack_metric_name},%d|%d|%lf|%d|%d|%d|%d|{args.attack_name}|{args.attack_param}|{main_acc}|{attack_metric}" %\
                (args.k,args.batch_size, args.main_lr, args.num_classes, args.Q, args.apply_trainable_layer,args.main_epochs)
            print(exp_result)
            append_exp_res(args.exp_res_path, exp_result)

        else:  
            args.need_auxiliary = 0
            args = load_parties(args)
            # if i == 0: # Only train once for all label_inference_attack
            #     vfl = MainTaskVFL(args)
            #     if args.dataset not in ['cora']:
            #         main_acc = vfl.train()
            #     else:
            #         main_acc = vfl.train_graph()
            #     i = i + 1
            vfl = args.basic_vfl
            main_acc = args.main_acc_noattack

            if args.attack_name == 'NormbasedScoring' or args.attack_name == 'DirectionbasedScoring':
                attack_acc,attack_auc = vfl.evaluate_attack()
                attack_metric_name = 'label_recovery_rate'
                # Save record for different defense method
                exp_result = f"K|bs|LR|num_class|Q|top_trainable|epoch|attack_name|{args.attack_param_name}|main_task_acc|{attack_metric_name},%d|%d|%lf|%d|%d|%d|%d|{args.attack_name}|{args.attack_param}|{main_acc}|{attack_acc}|{attack_auc}" %\
                    (args.k,args.batch_size, args.main_lr, args.num_classes, args.Q, args.apply_trainable_layer,args.main_epochs)
                print(exp_result)
                append_exp_res(args.exp_res_path, exp_result)
            else:
                attack_metric = vfl.evaluate_attack()
                attack_metric_name = 'label_recovery_rate'
                # Save record for different defense method
                exp_result = f"K|bs|LR|num_class|Q|top_trainable|epoch|attack_name|{args.attack_param_name}|main_task_acc|{attack_metric_name},%d|%d|%lf|%d|%d|%d|%d|{args.attack_name}|{args.attack_param}|{main_acc}|{attack_metric}" %\
                    (args.k,args.batch_size, args.main_lr, args.num_classes, args.Q, args.apply_trainable_layer,args.main_epochs)
                print(exp_result)
                append_exp_res(args.exp_res_path, exp_result)


def evaluate_attribute_inference(args):
    for index in args.attribute_inference_index:
        set_seed(args.current_seed)
        args = load_attack_configs(args.configs, args, index)
        # args = load_parties(args)
        print('======= Test Attack',index,': ',args.attack_name,' =======')
        print('attack configs:',args.attack_configs)
        if args.attack_name == 'AttributeInference':
            args.need_auxiliary = 1
            args = load_parties(args) # include load dataset with auxiliary data
            # actual train = train

            vfl = MainTaskVFL(args)
            if args.dataset not in ['cora']:
                main_acc , stopping_iter, stopping_time, stopping_commu_cost= vfl.train()
            else:
                main_acc = vfl.train_graph()
            # vfl = args.basic_vfl
            # main_acc = args.main_acc_noattack

            attack_metric = vfl.evaluate_attack()
            attack_metric_name = 'attribute_inference_rate'

            # Save record for different defense method
            exp_result = f"K|bs|LR|num_class|Q|top_trainable|epoch|attack_name|{args.attack_param_name}|main_task_acc|{attack_metric_name},%d|%d|%lf|%d|%d|%d|%d|{args.attack_name}|{args.attack_param}|{main_acc}|{attack_metric}" %\
                (args.k,args.batch_size, args.main_lr, args.num_classes, args.Q, args.apply_trainable_layer,args.main_epochs)
            print(exp_result)
            append_exp_res(args.exp_res_path, exp_result)


def evaluate_untargeted_backdoor(args):
    for index in args.untargeted_backdoor_index:
        torch.cuda.empty_cache()
        set_seed(args.current_seed)
        args.train_poison_list = None
        args.test_poison_list = None
        args = load_attack_configs(args.configs, args, index)
        args = load_parties(args)
        
        print('======= Test Attack',index,': ',args.attack_name,' =======')
        print('attack configs:',args.attack_configs)

        if args.attack_name == 'PGD':
            vfl = MainTaskVFL(args)
            if args.dataset not in ['cora']:
                main_acc, _, _, _ = vfl.train()
            else:
                main_acc, _, _ = vfl.train_graph()
            adv_acc = vfl.evaluate_attack()
            attack_metric = main_acc - adv_acc
            attack_metric_name = 'adv_acc_loss'
        else:
            if args.apply_ns:
                vfl = MainTaskVFLwithNoisySample(args)
            else:
                vfl = MainTaskVFL(args)
            if args.dataset not in ['cora']:
                main_acc, noise_main_acc = vfl.train()
            else:
                main_acc,noise_main_acc = vfl.train_graph()

            attack_metric = main_acc - noise_main_acc#args.main_acc_noattack - noise_main_acc
            attack_metric_name = 'acc_loss'
        # Save record for different defense method
        exp_result = f"K|bs|LR|num_class|Q|top_trainable|epoch|attack_name|{args.attack_param_name}|main_task_acc|{attack_metric_name},%d|%d|%lf|%d|%d|%d|%d|{args.attack_name}|{args.attack_param}|{main_acc}|{attack_metric}" %\
            (args.k,args.batch_size, args.main_lr, args.num_classes, args.Q, args.apply_trainable_layer,args.main_epochs)
        print(exp_result)
        append_exp_res(args.exp_res_path, exp_result)


def evaluate_adi_synthesis(args):
    for index in args.adi_synthesis_index:
        torch.cuda.empty_cache()
        set_seed(args.current_seed)
        attack_args = copy.deepcopy(args)
        attack_args = load_attack_configs(attack_args.configs, attack_args, index)
        print('======= Test Attack',index,': ',attack_args.attack_name,' =======')
        print('attack configs:',attack_args.attack_configs)

        vfl = attack_args.basic_vfl
        vfl.args = attack_args
        attack_result = vfl.evaluate_attack()

        exp_result = f"K|bs|LR|num_class|Q|top_trainable|epoch|attack_name|{attack_args.attack_param_name}|clean_acc|clean_dom95_count|clean_dom95_rate|attacked_acc|target_success|dom95|dom99|avg_domination,%d|%d|%lf|%d|%d|%d|%d|{attack_args.attack_name}|{attack_args.attack_param}|{attack_result['clean_accuracy']}|{attack_result['clean_input_domination_count@95%']}|{attack_result['clean_input_domination_success_rate@95%']}|{attack_result['attacked_accuracy']}|{attack_result['target_success_rate']}|{attack_result['domination_success_rate@95%']}|{attack_result['domination_success_rate@99%']}|{attack_result['avg_domination_percentage']}" %\
            (attack_args.k,attack_args.batch_size, attack_args.main_lr, attack_args.num_classes, attack_args.Q, attack_args.apply_trainable_layer,attack_args.main_epochs)
        print(exp_result)
        append_exp_res(attack_args.exp_res_path, exp_result)


def evaluate_contribution_analysis(args):
    for index in args.contribution_analysis_index:
        torch.cuda.empty_cache()
        set_seed(args.current_seed)
        attack_args = copy.deepcopy(args)
        attack_args = load_attack_configs(attack_args.configs, attack_args, index)
        print('======= Test Attack',index,': ',attack_args.attack_name,' =======')
        print('attack configs:',attack_args.attack_configs)

        vfl = attack_args.basic_vfl
        vfl.args = attack_args
        attack_result = vfl.evaluate_attack()

        header = f"K|bs|LR|num_class|Q|top_trainable|epoch|attack_name|{attack_args.attack_param_name}|clean_acc|party_id|acc_wo_party|acc_drop|pred_change|logit_shift|true_prob_drop|contribution_score"
        append_exp_res(attack_args.exp_res_path, header)
        for party_metric in attack_result["party_metrics"]:
            exp_result = f"%d|%d|%lf|%d|%d|%d|%d|{attack_args.attack_name}|{attack_args.attack_param}|{attack_result['clean_accuracy']}|{party_metric['party_id']}|{party_metric['accuracy_without_party']}|{party_metric['accuracy_drop']}|{party_metric['prediction_change_rate']}|{party_metric['mean_logit_shift']}|{party_metric['mean_true_class_prob_drop']}|{party_metric['contribution_score']}" %\
                (attack_args.k,attack_args.batch_size, attack_args.main_lr, attack_args.num_classes, attack_args.Q, attack_args.apply_trainable_layer,attack_args.main_epochs)
            print(exp_result)
            append_exp_res(attack_args.exp_res_path, exp_result)

def evaluate_targeted_backdoor(args):
    if args.defense_configs != None and 'party' in args.defense_configs.keys():
        args.defense_configs['party'] = [1] 
    # mark that backdoor data is never prepared
    args.target_label = None
    args.train_poison_list = None
    args.train_target_list = None
    args.test_poison_list = None
    args.test_target_list = None
    for index in args.targeted_backdoor_index:
        torch.cuda.empty_cache()
        set_seed(args.current_seed)
        args = load_attack_configs(args.configs, args, index)
        args = load_parties(args)
        print('======= Test Attack',index,': ',args.attack_name,' =======')
        print('attack configs:',args.attack_configs)

        if args.attack_name == 'ASB':
            args.need_auxiliary = 1
            args = load_parties(args) # include load dataset with auxiliary data
            
            if args.basic_vfl_withaux == None:
                vfl = MainTaskVFL(args)
                if args.dataset not in ['cora']:
                    main_acc = vfl.train()
                else:
                    main_acc = vfl.train_graph()
            else:
                main_acc = args.main_acc_noattack_withaux 
                vfl = args.basic_vfl_withaux 
            args.main_acc_noattack_withaux = main_acc
            args.basic_vfl_withaux = vfl
            
            attack_metric = vfl.evaluate_attack()
            attack_metric_name = 'attack_acc'

        else:
            # Targeted Backdoor VFL Training pipeline
            if args.apply_backdoor == True:
                vfl = MainTaskVFLwithBackdoor(args)
                main_acc, backdoor_acc = vfl.train()
            else:
                vfl = MainTaskVFL(args)
                if args.dataset not in ['cora']:
                    main_acc = vfl.train()
                else:
                    main_acc = vfl.train_graph()
            
            attack_metric = backdoor_acc
            attack_metric_name = 'backdoor_acc'
        
        # Save record for different defense method
        exp_result = f"K|bs|LR|num_class|Q|top_trainable|epoch|attack_name|{args.attack_param_name}|main_task_acc|{attack_metric_name},%d|%d|%lf|%d|%d|%d|%d|{args.attack_name}|{args.attack_param}|{main_acc}|{attack_metric}" %\
            (args.k,args.batch_size, args.main_lr, args.num_classes, args.Q, args.apply_trainable_layer,args.main_epochs)
        print(exp_result)
        append_exp_res(args.exp_res_path, exp_result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("backdoor")
    parser.add_argument('--device', type=str, default='cuda', help='use gpu or cpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--seed', type=int, default=97, help='random seed')
    parser.add_argument('--configs', type=str, default='test', help='configure json file path')
    parser.add_argument('--save_model', type=bool, default=False, help='whether to save the trained model')
    parser.add_argument('--run_phase', type=str, default='both', choices=['train', 'test', 'both'],
                        help='train only, test only from checkpoint, or train then evaluate attacks')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='optional checkpoint path for test-only mode')
    args = parser.parse_args()

    # for seed in range(97,102): # test 5 times 
    # for seed in [60]:
    # for seed in [97,98,99,100,101]: # test 5 times 
    for seed in [97]: # test 5 times 
        args.current_seed = seed
        set_seed(seed)
        print('================= iter seed ',seed,' =================')
        
        args = load_basic_configs(args.configs, args)
        args.need_auxiliary = 0 # no auxiliary dataset for attackerB

        if args.device == 'cuda':
            cuda_id = args.gpu
            torch.cuda.set_device(cuda_id)
            print(f'running on cuda{torch.cuda.current_device()}')
        else:
            print('running on cpu')

        
        ####### load configs from *.json files #######
        ############ Basic Configs ############
        
        # for mode in [0]:
            
        #     if mode == 0:
        #         args.global_model = 'ClassificationModelHostHead'
        #     else:
        #         args.global_model = 'ClassificationModelHostTrainableHead'
        #     args.apply_trainable_layer = mode

        mode = args.apply_trainable_layer 
        print('============ apply_trainable_layer=',args.apply_trainable_layer,'============')
        #print('================================')
    
        assert args.dataset_split != None, "dataset_split attribute not found config json file"
        assert 'dataset_name' in args.dataset_split, 'dataset not specified, please add the name of the dataset in config json file'
        args.dataset = args.dataset_split['dataset_name']
        # print(args.dataset)

        print('======= Defense ========')
        print('Defense_Name:',args.defense_name)
        print('Defense_Config:',str(args.defense_configs))
        print('===== Total Attack Tested:',args.attack_num,' ======')
        print('targeted_backdoor:',args.targeted_backdoor_list,args.targeted_backdoor_index)
        print('untargeted_backdoor:',args.untargeted_backdoor_list,args.untargeted_backdoor_index)
        print('label_inference:',args.label_inference_list,args.label_inference_index)
        print('attribute_inference:',args.attribute_inference_list,args.attribute_inference_index)
        print('feature_inference:',args.feature_inference_list,args.feature_inference_index)
        print('adi_synthesis:',args.adi_synthesis_list,args.adi_synthesis_index)
        print('contribution_analysis:',args.contribution_analysis_list,args.contribution_analysis_index)
        
        
        # Save record for different defense method
        args.exp_res_dir = f'exp_result/{args.dataset}/Q{str(args.Q)}/{str(mode)}/'
        if not os.path.exists(args.exp_res_dir):
            os.makedirs(args.exp_res_dir)
        filename = f'{args.defense_name}_{args.defense_param},model={args.model_list[str(0)]["type"]}.txt'
        args.exp_res_path = args.exp_res_dir + filename
        print(args.exp_res_path)
        print('=================================\n')

        iterinfo='===== iter '+str(seed)+' ===='
        append_exp_res(args.exp_res_path, iterinfo)

        args.basic_vfl_withaux = None
        args.main_acc_noattack_withaux = None
        args.basic_vfl = None
        args.main_acc_noattack = None

        args = load_attack_configs(args.configs, args, -1)
        args = load_parties(args)

        commuinfo='== commu:'+args.communication_protocol
        append_exp_res(args.exp_res_path, commuinfo)

        if args.run_phase == 'test':
            evaluate_saved_model_test_only(args)
            continue

        args.basic_vfl, args.main_acc_noattack = evaluate_no_attack(args)
        if args.run_phase == 'train':
            continue
        
        if args.label_inference_list != []:
            evaluate_label_inference(args)

        if args.attribute_inference_list != []:
            evaluate_attribute_inference(args)
        
        if args.feature_inference_list != []:
            evaluate_feature_inference(args)

        if args.adi_synthesis_list != []:
            torch.cuda.empty_cache()
            evaluate_adi_synthesis(args)

        if args.contribution_analysis_list != []:
            torch.cuda.empty_cache()
            evaluate_contribution_analysis(args)

        if args.untargeted_backdoor_list != []:
            torch.cuda.empty_cache()
            evaluate_untargeted_backdoor(args)

        if args.targeted_backdoor_list != []:
            torch.cuda.empty_cache()
            evaluate_targeted_backdoor(args)
        
        
