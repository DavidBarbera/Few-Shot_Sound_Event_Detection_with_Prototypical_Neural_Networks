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

import matplotlib.pyplot as plt
import seaborn as sns


def load_model_and_options():

    model_type = "protonet_conv_SG_80_10_10_trainC3K1Q16_valC3K0Q16"
    jsonfile = os.path.join("results", model_type, "opt_2021-07-17-09-37-13-103363.json")
    model_path =  os.path.join("results", model_type, "best_model_2021-07-17-09-37-13-103363.pt")

    model_type = "protonet_conv_SG_80_10_10_trainC3K1Q5_valC3K0Q5"
    jsonfile = os.path.join("results", model_type, "opt_2021-07-17-06-46-48-319727.json")
    model_path =  os.path.join("results", model_type, "best_model_2021-07-17-06-46-48-319727.pt")

    model_type = "protonet_conv_SG_80_10_10_trainC3K5Q5_valC3K0Q5"
    jsonfile = os.path.join("results", model_type, "opt_2021-07-17-06-39-32-193869.json")
    model_path =  os.path.join("results", model_type, "best_model_2021-07-17-06-39-32-193869.pt")
    

    print(f" ----  Using config: {jsonfile}")
    with open(jsonfile) as json_file:
        opt = json.load(json_file)
        model_opt = opt
        #print(opt)

    return model_path, opt, model_opt, jsonfile


def main():

    test_episodes = 48
    model_path, opt, model_opt, jsonfile = load_model_and_options()
    
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
    
    

    print("Evaluating {:d}-way, {:d}-shot with {:d} query examples/class over {:d} episodes".format(
    data_opt['data.test_way'], data_opt['data.test_shot'],
    data_opt['data.test_query'], test_episodes))

    torch.manual_seed(1234)
    if data_opt['data.cuda']:
        torch.cuda.manual_seed(1234)

    #print(model_opt)
    #print(model_opt['model.model_name'])
    model_opt['data.test_episodes']=test_episodes
    data = data_utils.load(model_opt, ['test'])

    if data_opt['data.cuda']:
        model.cuda()

    writer, project = log_utils.get_summary_writer(model_opt, "") #no need of timestamp
    print(project)


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
        

    # print(test_acc.shape)
    # print(f"test loss: {test_loss.mean():.6f} +/- {(1.96*test_loss.std())/np.sqrt(len(test_loss)):.6f}" )
    # print(f"test accuracy: {test_acc.mean():.6f} +/- {(1.96*test_acc.std())/np.sqrt(len(test_acc)):.6f}" )

    print(test_acc.shape)
    print(f"Test results over {test_episodes} episodes: ")
    print(f"test loss: {test_loss.mean():.6f} +/- {(1.96*test_loss.std())/np.sqrt(len(test_loss)):.6f},  min loss: {test_loss.min()}, max loss: {test_loss.max()}" )
    print(f"test accuracy: {test_acc.mean():.6f} +/- {(1.96*test_acc.std())/np.sqrt(len(test_acc)):.6f}, min acc: {test_acc.min()}, max acc: {test_acc.max()}" )
    
    ax = sns.boxplot(y=test_acc)
    ax = sns.swarmplot(y=test_acc, color="0.25")
    plt.show()
    

if __name__ == '__main__':
    main()