import os
import json
import math
from tqdm import tqdm

import numpy as np

import torch
import torchnet as tnt
from torch.utils.tensorboard import SummaryWriter

from protonets.utils import filter_opt, merge_dict
import protonets.utils.data as data_utils
import protonets.utils.model as model_utils
import protonets.utils.log as log_utils

from protonets.data.sg_base import logMelSpectro_32, extract_samples_per_class, load_audio, detect_separator
from protonets.data.sg import get_samples, SG_DATA_DIR, data_dir


# SG_DATA_DIR  = '../data/sg/'
# data_dir = '../data/sg/data/'


def get_options_from_json( jsonfile):

    with open(jsonfile) as json_file:
        opt = json.load(json_file)

    return opt

    
def load_all_data(paths, extract_features):
    data_instances=[]
    for path in paths:
        audio = load_audio(path) #here idx is the whole path to the wavfile (for simplicity)
        features = extract_features(audio)
        data_instances.append(features)
    instances_stack=torch.stack(data_instances)
        
    return instances_stack


def obtain_embeddings(model_path, opt):

    #print(opt['model.model_name'], opt['data.split'])
    split_dir = os.path.join(SG_DATA_DIR, 'splits', opt['data.split'])
    samples = get_samples(split_dir,'test')
    class_samples = extract_samples_per_class(samples)

    config_name = "EMBEDDINGS_TEST_SG_80_10_10_C{:d}_K{:d}_Q{:d}_".format(
    opt['data.way'], opt['data.shot'],
    opt['data.query'])
       
    data = load_all_data(samples, logMelSpectro_32)

    model = torch.load(model_path)
    if opt['data.cuda']:
        model.cuda()
    model.eval()

    embeddings = model.encoder.forward(data.cuda())

    return embeddings.cpu().detach().numpy(), config_name


def obtain_all_test_embeddings():
    results_dir="results"

    configurations = [os.path.join(results_dir,f) for f in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir,f))]

    for config in configurations[:]:
        model_names = [os.path.join(config,f) for f in os.listdir(config) if os.path.join(results_dir,f).endswith(".pt")]
        #print(model_names)
        for model_path in model_names:
            #print(model_name)
            time_stamp = model_path.split("best_model_")[1].split(".pt")[0]
            #print(time_stamp)
            json_file=os.path.join(config,f"opt_{time_stamp}.json")
            #print(json_file)
            
            opt = get_options_from_json(json_file)

            embeddings, config_name = obtain_embeddings(model_path, opt)
            path_separator = detect_separator(config)
            model_name = model_path.split(path_separator)[-1]

            print(f"{config_name}__{model_name[:-3]}.npy")
            
            np.save(os.path.join("test_embeddings",f"{config_name}__{model_name[:-3]}.npy"), embeddings)


if __name__ == '__main__':

    obtain_all_test_embeddings()