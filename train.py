import os
import argparse
import json
from functools import partial
from tqdm import tqdm

import numpy as np

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
#from adabelief_pytorch import AdaBelief
#from ranger_adabelief import RangerAdaBelief


import torchvision
import torchnet as tnt
from torchsummary import summary

#from protonets.engine import Engine

import protonets.utils.data as data_utils
import protonets.utils.model as model_utils
import protonets.utils.log as log_utils


import warnings 
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description='Train prototypical networks with different models and datasets.')
parser.add_argument('--gpu', type=int, default=0, help="Option to choose GPU to train on.")


def main(opt):
    #data_utils.generate_dataset_info(opt)
    timestamp = log_utils.create_timestamp()
    writer, project = log_utils.get_summary_writer(opt, timestamp)
    
    
    if not os.path.isdir(opt['log.exp_dir']):
        os.makedirs(opt['log.exp_dir'])

    # save opts
    path_options = os.path.join(opt['log.exp_dir'], project, f'opt_{timestamp}.json')
    print(f"----  Saving options to {path_options}")
    with open( path_options,'w') as f:
        json.dump(opt, f)
        f.write('\n')

    trace_file = os.path.join(opt['log.exp_dir'], 'trace.txt')

    # Postprocess arguments
    if opt['model.model_name'] == "protonet_conv":# and opt['data.split'] == "vinyals":
        opt['model.x_dim'] = list(map(int, opt['model.x_dim'].split(',')))
    opt['log.fields'] = opt['log.fields'].split(',')

    torch.manual_seed(1234)
    if opt['data.cuda']:
        torch.cuda.manual_seed(1234)

    if opt['data.trainval']:
        data = data_utils.load(opt, ['trainval'])
        train_loader = data['trainval']
        val_loader = None
    else:
        data = data_utils.load(opt, ['train', 'val'])
        train_loader = data['train']
        val_loader = data['val']

    model = model_utils.load(opt)

    trainable_params, non_trainable_params = model_utils.count_parameters(model)

    print(f"******************************************************************")
    print(f"****                                                              ")
    print(f"****     model parameters:                                        ")
    print(f"****            trainable: {trainable_params}")
    print(f"****        non-trainable: {non_trainable_params}")
    print(f"****                                                              ")
    print(f"******************************************************************")

    #summary(model, (1,165,26))

    #summary(model, (1,165,26))


    if opt['data.cuda']:
        model.cuda()    

    #Optimizer
    optim_method = getattr(optim, opt['train.optim_method'])
    optim_config = { 'lr': opt['train.learning_rate'], 'weight_decay': opt['train.weight_decay'] }    
    optimizer = optim_method(model.parameters(), **optim_config)
    #optimizer = AdaBelief(model.parameters(), lr=1e-3, eps=1e-16, betas=(0.9,0.999), weight_decouple = True, rectify = False)

    scheduler = lr_scheduler.StepLR(optimizer, opt['train.decay_every'], gamma=0.5)


    best_loss=np.inf
    patience=0
    for epoch in range(opt['train.epochs']):
        if patience > opt['train.patience']:
            print(f"==> patience {opt['train.patience']} exceeded.")  
            break
        

        train_loss=0.
        train_acc=0.
        model.train()
        for i, sample in enumerate(tqdm(train_loader, desc = f"Epoch {epoch+1} train"),0):

            optimizer.zero_grad()
            loss, output = model.loss(sample)
            
            train_loss+=output['loss']/100
            train_acc+=output['acc']/100
            writer.add_scalar('Loss/train', output['loss'], epoch*opt['data.train_episodes']+i)
            writer.add_scalar('Accuracy/train', output['acc'], epoch*opt['data.train_episodes']+i)
            loss.backward()
            optimizer.step()

            #on_update: (log train acc,loss)
        scheduler.step()#"https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)

        #on_end_epoch:  (evaluated validation data / log acc,loss / display metrics / save if best model)

        #model_utils.evaluate(model, val_loader,)
        model.eval()
        with torch.no_grad():
            val_loss=0.
            val_acc=0.
            for j, sample in enumerate(tqdm(val_loader, desc = f"Epoch {epoch+1} valid"),0):
                _, val_output = model.loss(sample)

                val_loss+=val_output['loss']/100
                val_acc+=val_output['acc']/100
                writer.add_scalar('Loss/validation', val_output['loss'], epoch*opt['data.test_episodes']+j)
                writer.add_scalar('Accuracy/validation', val_output['acc'], epoch*opt['data.test_episodes']+j)

            print(f"Epoch {(epoch+1)} -----------  train loss = {train_loss:.6f}, train acc = {train_acc:.6f}, val loss = {val_loss:.6f}, val acc = {val_acc:.6f}                patience: {patience}/{opt['train.patience']}")

            if val_loss < best_loss:        
                best_loss = val_loss
                #save model:
                print(f"==> best model (val loss = {best_loss:.6f}), saving model ...")
                model.cpu()
                model_path = os.path.join(opt['log.exp_dir'], project, f'best_model_{timestamp}.pt')
                print(f"Saving model to {model_path}")
                torch.save(model, model_path )
                if opt['data.cuda']:
                    model.cuda()

                patience=0
            else:
                patience+=1
      



        
    #on_end




if __name__ == '__main__':
#-----------------------------------------------------------
    args = vars(parser.parse_args())
    gpu=args['gpu']
    print(f"device: ", gpu)
    #print(f"Currently on GPU {torch.cuda.current_device()}, {torch.cuda.get_device_name(gpu)} {torch.cuda.get_device_capability(gpu)}")
    
    #jsonfile='opt.json'#convolutions
    #jsonfile='opt_sequences.json' #lstm
    #jsonfile='opt_wang.json' #protonet cnn on swc
    jsonfile='opt_sg.json' #protonet cnn on swc

    with open(jsonfile) as json_file:
        opt = json.load(json_file)
    print(f" ----  Using {jsonfile} options:")
    print(json.dumps(opt, indent=4))

    with torch.cuda.device(gpu):
        print("***************************************************")
        print(f"*****   Currently on GPU {torch.cuda.current_device()}, {torch.cuda.get_device_name(gpu)} {torch.cuda.get_device_capability(gpu)}   ******")
        print("***************************************************")

        main(opt)

#-----------------------------------------------------------