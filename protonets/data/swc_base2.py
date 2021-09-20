import os
import numpy as np
import torch
import torchaudio

from .swcreaders import load_readers_word_info, save_word_infor_per_reader

print("torch version: ", torch.__version__)

SWC_DATA_DIR  = 'data/SWC/english/'
cwd=os.getcwd()
cfilewd=os.path.dirname(os.path.realpath(__file__))

#Audio Processing
mfcc_transform = torchaudio.transforms.MFCC()

sr=16000
n_mfcc = 13
n_mels = 40
n_fft = 512 
hop_length = int(sr*0.01)
win_len = int(sr*0.03)
fmin = 0
fmax = None
sr = 16000  

melkwargs={"n_fft" : n_fft, "n_mels" : n_mels, "hop_length": hop_length, "f_min" : fmin, "f_max" : fmax}

# Torchaudio 'librosa compatible' default dB mel scale 
mfcc_torch_db = torchaudio.transforms.MFCC(sample_rate=sr, n_mfcc=n_mfcc, 
                                           dct_type=2, norm='ortho', log_mels=False, 
                                           melkwargs=melkwargs)
compute_deltas = torchaudio.transforms.ComputeDeltas()

def load_audio(file):
    y, _ = torchaudio.load(file)
    return y


class SWC_Dataset(torch.utils.data.Dataset):
    def __init__(self, readers_list, nc, ns, nq, data_dir):
        self.sr=16000
        self.classes = []
        self.class_name = {}
        self.max_lengths = {}
        self.all_lengths = {}
        self.all_idxs = {}
        self.xs_lengths = {}
        self.xq_lengths = {}
        self.xs_idxs = {}
        self.xq_idxs = {}
        self.class_idxs = {} #class indexes per reader
        
        rinfo, _ = load_readers_word_info()
        #save_word_infor_per_reader( rinfo )
        readers_wordlist, readers_wordinstances = load_word_info_per_reader()
        rcoding, _ = get_readers_coding()

        class_count = 0
        for reader in readers_list:
            reader_class_idxs=[]

            starts = rinfo[reader].starts
            ends = rinfo[reader].ends
            items = rinfo[reader].items
            wordsets = rinfo[reader].wordsets
            
            few_shot_set = np.argwhere(readers_wordinstances[reader]>=(ns+nq))[:,0]
            
            if len(few_shot_set)<nc:
                pass
            
            else:
                wordinstances = readers_wordinstances[reader][few_shot_set]
                wordlist = readers_wordlist[reader][few_shot_set]
                reader_coded = rcoding[reader]

                for i, (word, ninstances) in enumerate(zip(wordlist, wordinstances)):
                    _class_ = f"{reader_coded}/{word}"
                    self.class_name[_class_] = _class_

                    word_data=[]
                    instances_lengths=[]
                    for j,(s,e) in enumerate(zip(starts[wordsets[word]],ends[wordsets[word]])):

                        instances_lengths.append(int(np.ceil((e-s)*self.sr))) #!!!!

                    assert ninstances == len(instances_lengths)

                    idxs = torch.randperm(ninstances)[:ns+nq]
                    sidxs = idxs[:ns]
                    qidxs = idxs[ns:ns+nq]
                    xsinstances=sidxs.data.numpy()
                    xqinstances=qidxs.data.numpy()

                    lengths=[]
                    all_idxs=[]
                    for sidx in sidxs:
                        lengths.append(instances_lengths[sidx])
                        all_idxs.append(sidx)
                    for qidx in qidxs:
                        lengths.append(instances_lengths[qidx])
                        all_idxs.append(qidx)
                        
                    self.max_lengths[_class_] = max(lengths)
                    self.all_lengths[_class_] = lengths
                    self.all_idxs[_class_] = all_idxs
                                              
                    self.xs_idxs[_class_] = xsinstances
                    self.xq_idxs[_class_] = xqinstances
                    self.xs_lengths[_class_] = np.array(instances_lengths)[xsinstances]
                    self.xq_lengths[_class_] = np.array(instances_lengths)[xqinstances]
    
                    self.classes.append(_class_)
                    reader_class_idxs.append(class_count)
                    class_count+=1

                self.class_idxs[rcoding[reader]] = np.array(reader_class_idxs)
                    
        
    def __len__(self):
        return len(self.classes)
    
    
    def __getitem__(self, idx):
        
        _class_=self.classes[idx]

        sample = {
                'class': self.class_name[_class_], 
                'max_lengths': self.max_lengths[_class_],
                'all_lengths': self.all_lengths[_class_],
                'all_idxs': self.all_idxs[_class_],
                'xs_idxs': self.xs_idxs[_class_],
                'xq_idxs': self.xq_idxs[_class_],
                'xs_lengths': self.xs_lengths[_class_],
                'xq_lengths': self.xq_lengths[_class_]
                    }
        
        return sample
    


class FewShotSamplerPerReader(object):
    def __init__(self, class_idxs, n_way, n_episodes):
        self.n_way = n_way
        self.n_episodes = n_episodes
        self.class_idxs = class_idxs
        self.readers = list(self.class_idxs.keys()) #order matters here
        
        #quick check:
        for reader in self.readers:
            assert len(self.class_idxs[reader]) >= self.n_way
            
        
    def __len__(self):
        return self.n_episodes
    
    
    def __iter__(self):
        for i in range(self.n_episodes):
            #random reader (1 per episode)
            n=torch.randperm(len(self.readers))[0]
            #print("--------------",self.readers[n])
            #indexes of the reader randomly chosen
            reader_idxs = self.class_idxs[self.readers[n]]
            #Chose n_way wordclasses randomly from such reader
            yield (torch.randperm(len(reader_idxs))+reader_idxs[0])[:self.n_way]



def pad_audio(audio, padding, margin=0.03, sr=16000):
    audio_padded = torch.zeros((1, padding+int(sr*(2*margin))))
    n=audio.shape[1]
    s = int((margin*sr)+int(padding/2)-int(n/2))
    e = s + n
    audio_padded[:,s:e]=audio
    return audio_padded


def extract_features(audio):
    mfcc = mfcc_torch_db(audio)
    deltas = compute_deltas(mfcc)
    features = torch.cat([mfcc,deltas],1)
    d = torch.transpose(features,1,2)
    return d  


# def load_data(class_names, instances, episodic_length):

#     data = []
#     for i,class_name in enumerate(class_names):
#         data_instances=[]

#         for j,instance in enumerate(instances[i]):
#             audio = pad_audio(instance, padding = episodic_length)
#             features = extract_features(audio)
#             data_instances.append(features)

#         instances_stack=torch.stack(data_instances)
#         data.append(instances_stack)
         
#     stacked_data = torch.stack(data)

#     return stacked_data

def load_data(data_dir, class_names, instances, lengths, episodic_length):
    data = []
    for i,class_name in enumerate(class_names):
        #print(class_name)
        data_instances=[]
        for j,instance in enumerate(instances[i]):
            original_audio = load_audio(f"{data_dir}/data/{class_name}/{instance}.wav")
            #print(i,j,original_audio.size(), lengths[i][j],instance)
            #print(original_audio.size(), lengths[i][j], instance)
            # original_audio = unpad_audio(audio_padded, lengths[i][j])
            audio = pad_audio(original_audio, padding = episodic_length)
            features = extract_features(audio)
            #print(i,j,original_audio.size(), lengths[i][j], instance, audio.size(), features.size())
            #print(features.size())
            data_instances.append(features)
        instances_stack=torch.stack(data_instances)
        #print("instances stack:",instances_stack.size())
        data.append(instances_stack)
    
            
    stacked_data = torch.stack(data)
    #print("stack:",stacked_data.size())
    return stacked_data


class BatchDifferentSizedAudios:
    def __init__(self, batch, data_dir):
        self.class_names = [item['class'] for item in batch]
        # xs_items = [item['xs'] for item in batch]
        # xq_items = [item['xq'] for item in batch]
        class_ns_nq_length = [item['max_lengths'] for item in batch]
        all_lengths = [item['all_lengths'] for item in batch]
        xs_idxs = [item['xs_idxs'] for item in batch]
        xq_idxs = [item['xq_idxs'] for item in batch]
        xs_lengths = [item['xs_lengths'] for item in batch]
        xq_lengths = [item['xq_lengths'] for item in batch]
        all_idxs = [item['all_idxs'] for item in batch]

        #print(self.class_names)

        #print("xs_lengths :",xs_lengths, "xq_lengths: ", xq_lengths)
        #print("xs_idxs:", xs_idxs,"xq_idxs: ", xq_idxs)
        episodic_length = max(class_ns_nq_length)
        #support_lengths = max(xs_lengths)
        #query_lengths = max(xq_lengths)
        #print("all_lengths: ", all_lengths)
        #print("all_idxs: ", all_idxs)
        #torch.cuda.empty_cache()
        #print("max_lengths: ",class_ns_nq_length)
        #print("Episodic length: ", episodic_length)
        #print("xs:") 
        self.xs = load_data(data_dir, self.class_names, xs_idxs, xs_lengths, episodic_length)#.cuda()#this needs refactoring
        #print("xq:")
        self.xq = load_data(data_dir, self.class_names, xq_idxs, xq_lengths, episodic_length)#.cuda()#this needs refactoring
        #print(self.class_names)
        #print("nclasses: ",len(self.class_names), "xs:",self.xs.size(),"xq:",self.xq.size())


    def pin_memory(self):
        self.xs = self.xs.pin_memory()
        self.xq = self.xq.pin_memory()
        return self


def collate_wrapper_with_loader(batch):
    return BatchDifferentSizedAudios(batch, SWC_DATA_DIR)


def load_word_info_per_reader():
    wordlist=np.load(f"{cfilewd}/swc/swc_readers410_wordlist.npz")
    wordinstances=np.load(f"{cfilewd}/swc/swc_readers410_wordinstances.npz")
    
    return wordlist, wordinstances


def get_readers_coding():
    with open(f"{cfilewd}/swc/swc_readers410_coding.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    rcoding={}
    rname={}
    for line in lines:
        coding, name = line.strip("\n").split("\t")
        rcoding[name] = coding
        rname[coding] = name
        
    return rcoding, rname