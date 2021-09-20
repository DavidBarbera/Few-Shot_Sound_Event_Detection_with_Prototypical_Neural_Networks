import os
import json
import math
from tqdm import tqdm

import numpy as np

import torch
import torchnet as tnt

from protonets.utils import filter_opt, merge_dict
import protonets.utils.data as data_utils
import protonets.utils.model as model_utils
import protonets.utils.log as log_utils

from protonets.data.sg_base import detect_separator

import seaborn as sns


def get_options_from_json( jsonfile):

    with open(jsonfile) as json_file:
        opt = json.load(json_file)

    return opt



def test(model_path, model_opt, test_episodes=100):
    
    opt = model_opt
    
    model = torch.load(model_path)
    model.eval()

    # Postprocess arguments
    model_opt['model.x_dim'] = map(int, model_opt['model.x_dim'].split(','))
    model_opt['log.fields'] = model_opt['log.fields'].split(',')

    # construct data
    data_opt = { 'data.' + k: v for k,v in filter_opt(model_opt, 'data').items() }

    episode_fields = {
        'data.test_way': 'data.way',
        'data.test_shot': 'data.shot',
        'data.test_query': 'data.query',
        'data.test_episodes': 'data.train_episodes'
    }

    for k,v in episode_fields.items():
        if opt[k] != 0:
            data_opt[k] = opt[k]
        elif model_opt[k] != 0:
            data_opt[k] = model_opt[k]
        else:
            data_opt[k] = model_opt[v]
            
    data_opt['data.test_episodes']=test_episodes
    model_opt['data.test_episodes']=test_episodes
    # print("Evaluating {:d}-way, {:d}-shot with {:d} query examples/class over {:d} episodes".format(
    # data_opt['data.test_way'], data_opt['data.test_shot'],
    # data_opt['data.test_query'], data_opt['data.test_episodes']))

    config_name = "TEST_SG_80_10_10_C{:d}_K{:d}_Q{:d}_over_{:d}_episodes".format(
    data_opt['data.test_way'], data_opt['data.test_shot'],
    data_opt['data.test_query'], data_opt['data.test_episodes'])

    torch.manual_seed(1234)
    if data_opt['data.cuda']:
        torch.cuda.manual_seed(1234)

    data = data_utils.load(model_opt, ['test'])

    if data_opt['data.cuda']:
        model.cuda()

    writer, project = log_utils.get_summary_writer(model_opt, "") #no need of timestamp
    #print(project)


    model.eval()

    test_loss=np.zeros((len(data['test'])))
    test_acc=np.zeros((len(data['test'])))
    for i, sample in enumerate(tqdm(data['test'], desc = "test")):
        #print(f"Evaluation: sample size: {sample.shape}")
        _, output = model.loss(sample)
        test_loss[i]=output['loss']
        test_acc[i]=output['acc']
        writer.add_scalar('Loss/test', output['loss'], i)
        writer.add_scalar('Accuracy/test', output['acc'], i)
        

    print("SHAPE test: ",test_acc.shape)
    print(config_name)
    print(f"Test results over {test_episodes} episodes: ")
    print(f"test loss: {test_loss.mean():.6f} +/- {(1.96*test_loss.std())/np.sqrt(len(test_loss)):.6f},  min loss: {test_loss.min()}, max loss: {test_loss.max()}" )
    print(f"test accuracy: {test_acc.mean():.6f} +/- {(1.96*test_acc.std())/np.sqrt(len(test_acc)):.6f}, min acc: {test_acc.min()}, max acc: {test_acc.max()}" )
    
    return test_loss, test_acc, config_name


def test_all_models(test_episodes=1000):
    results_dir="results"

    configurations = [os.path.join(results_dir,f) for f in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir,f))]

    for config in configurations[:]:
        model_names = [os.path.join(config,f) for f in os.listdir(config) if os.path.join(results_dir,f).endswith(".pt")]
        print(model_names)
        for model_path in model_names:
            print(model_path)
            time_stamp = model_path.split("best_model_")[1].split(".pt")[0]
            print(time_stamp)
            json_file=os.path.join(config,f"opt_{time_stamp}.json")
            print(json_file)
            
            opt = get_options_from_json(json_file)
            test_loss, test_acc, config_name =  test(model_path, opt, test_episodes=test_episodes)
            path_separator = detect_separator(config)
            model_name = model_path.split(path_separator)[-1]
            np.savez(os.path.join("test_results",f"{config_name}__{model_name[:-3]}.npz"), test_loss=test_loss, test_acc=test_acc)


if __name__ == '__main__':

    test_all_models(test_episodes=1000)