import os
import sys
import platform
import json
from datetime import datetime

import numpy as np

from torch.utils.tensorboard import SummaryWriter


def create_timestamp():
    t=datetime.now()
    # if sys.platform != 'win32':
    #     b=str(t).split(" ")
    #     timestamp = ".".join(b)

    # else:
    b=str(t).split(" ")

    #print(b[0],b[1])
    c=b[1].split(":")
    #print(c[0],c[1],c[2])
    d=c[2].split(".")
    #print(d[0],d[1])
    day=b[0]
    hour=c[0]
    minute=c[1]
    second=d[0]
    fractionsecond=d[1]
    timestamp_list=[day,hour,minute,second,fractionsecond]
    timestamp = "-".join(timestamp_list)

    print(f"Running on {os.name}, {sys.platform}, {platform.platform()}")
    return timestamp


def get_summary_writer(opt, timestamp):

    c=opt['data.way']
    k=opt['data.shot']
    q=opt['data.query']

    ct=opt['data.test_way']
    kt=opt['data.test_shot']
    qt=opt['data.test_query']

    dataset = opt['data.dataset']

    split = opt['data.split']

    project=f"{opt['model.model_name']}_{dataset}_{split}_trainC{c}K{k}Q{q}_valC{ct}K{kt}Q{qt}"

    results_folder=f"{opt['log.exp_dir']}"

    print(f" ----  Tensorboard logging {project} project on folder {results_folder}/")

    return SummaryWriter(f"{results_folder}/{project}/{timestamp}/", filename_suffix=f"_{timestamp}"), project


def extract_meter_values(meters):
    ret = { }

    for split in meters.keys():
        ret[split] = { }
        for field,meter in meters[split].items():
            ret[split][field] = meter.value()[0]

    return ret

def render_meter_values(meter_values):
    field_info = []
    for split in meter_values.keys():
        for field,val in meter_values[split].items():
            field_info.append("{:s} {:s} = {:0.6f}".format(split, field, val))

    return ', '.join(field_info)

def convert_array(d):
    ret = { }
    for k,v in d.items():
        if isinstance(v, dict):
            ret[k] = { }
            for kk,vv in v.items():
                ret[k][kk] = np.array(vv)
        else:
            ret[k] = np.array(v)

    return ret

def load_trace(trace_file):
    ret = { }

    with open(trace_file, 'r') as f:
        for i,line in enumerate(f):
            vals = json.loads(line.rstrip('\n'))

            if i == 0:
                for k,v in vals.items():
                    if isinstance(v, dict):
                        ret[k] = { }
                        for kk in v.keys():
                            ret[k][kk] = []
                    else:
                        ret[k] = []

            for k,v in vals.items():
                if isinstance(v, dict):
                    for kk,vv in v.items():
                        ret[k][kk].append(vv)
                else:
                    ret[k].append(v)

    return convert_array(ret)
