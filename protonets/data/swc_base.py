import os
import numpy as np
import torch
import torchaudio

import librosa

from .swcreaders import load_readers_word_info, save_word_infor_per_reader

print("torch version: ", torch.__version__)


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

melkwargs={"n_fft" : n_fft, "n_mels" : n_mels, "hop_length": hop_length, "f_min" : fmin, "f_max" : fmax }#, "return_complex": False}

# Torchaudio 'librosa compatible' default dB mel scale 
mfcc_torch_db = torchaudio.transforms.MFCC(sample_rate=sr, n_mfcc=n_mfcc, 
                                           dct_type=2, norm='ortho', log_mels=False, 
                                           melkwargs=melkwargs)
compute_deltas = torchaudio.transforms.ComputeDeltas()
melSpectro = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=1024, win_length=400, hop_length=160)
to_dB = torchaudio.transforms.AmplitudeToDB(top_db=80)

def load_audio(file):
    y, _ = torchaudio.load(file)
    return y


class SWC_Dataset(torch.utils.data.Dataset):
    def __init__(self, readers_list, nc, ns, nq, data_dir, max_audio_length=0.5):

        self.classes = []
        self.class_name = {}
        self.nxs = {}
        self.nxq = {}
        self.idxs = {}

        self.class_idxs = {} #class indexes per reader
        self.excluded_readers = []
        self.max_audio_length = max_audio_length

        rinfo, _ = load_readers_word_info()
        #save_word_infor_per_reader( rinfo )

        #readers_wordlist, readers_wordinstances = load_word_info_per_reader()
        
        rcoding, _ = get_readers_coding()
        
        class_count = 0
        for reader in readers_list:
            reader_class_idxs=[]
            
            readers_wordlist, readers_wordinstances, new_idxs, _, _, _ = less_than(max_audio_length, rinfo[reader])#reader, rinfo) 

            few_shot_set = np.argwhere(readers_wordinstances>=(ns+nq))[:,0]
            
            if len(few_shot_set)<nc:
                self.excluded_readers.append(reader)
                #pass
            
            else:
                wordinstances = readers_wordinstances[few_shot_set]
                wordlist = readers_wordlist[few_shot_set]
                reader_coded = rcoding[reader]

                for i, (word, ninstances) in enumerate(zip(wordlist, wordinstances)):
                    _class_ = f"{reader_coded}/{word}"
                    self.class_name[_class_] = _class_
                    self.nxs[_class_] = ns
                    self.nxq[_class_] = nq
                    self.idxs[_class_] = new_idxs[word]

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
                'nxs': self.nxs[_class_],
                'nxq': self.nxq[_class_],
                'idxs': self.idxs[_class_] 
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
            
        self.mode='episodic_sampling'
        
    def __len__(self):
        return self.n_episodes
    
    
    def __iter__(self):
        for i in range(self.n_episodes):
            #random reader (1 per episode)
            n=torch.randperm(len(self.readers))[0]
            #print(self.readers[n])
            #indexes of the reader randomly chosen
            reader_class_idxs = self.class_idxs[self.readers[n]]
            #Chose n_way wordclasses randomly from such reader
            yield (torch.randperm(len(reader_class_idxs))+reader_class_idxs[0])[:self.n_way]


def pad_audio(audio, padding, margin=0.03, sr=16000):
    audio_padded = torch.zeros((1, padding+int(sr*(2*margin))+1)) #+1 sample to avoid weird rounding errors
    #print(audio.shape)
    n=audio.shape[1]
    s = int((margin*sr)+int(padding/2)-int(n/2))
    e = s + n
    #print(f"on pad_audio: {n} {audio.shape} {padding} {margin} {sr}")
    if audio_padded.shape[1] >= audio.shape[1]:
        audio_padded[:,s:e] = audio
    else: #as in the Wang's article
        m = audio_padded.shape[1] #m<n
        mid = int(n/2)
        half = int(m/2)
        start = mid-half #>0 as m/2<n/2
        audio_padded[:,:] = audio[:,start:start+m]
    return audio_padded


def MFCCdeltas_26(audio):
    mfcc = mfcc_torch_db(audio)
    deltas = compute_deltas(mfcc)
    features = torch.cat([mfcc,deltas],1)
    d = torch.transpose(features,1,2)
    return d  


def load_data( class_names, instances, episodic_length, extract_features, idxs, margin=0.03, sr=16000):
    data = []
    for i,class_name in enumerate(class_names):
        data_instances=[]

        for j,instance in enumerate(instances[i]):
          
            try:
                #print(i,j,instance.shape)
                audio = pad_audio(instance, padding = episodic_length, margin=margin, sr=sr)
            except:
                raise Exception(f"Padded audio ({episodic_length+int(margin*sr)}) is shorter than actual audio ({instance.size[1]}) for {class_name}, instance {idxs[i][j]}.")

            features = extract_features(audio)
            data_instances.append(features)
            
        instances_stack=torch.stack(data_instances)
        data.append(instances_stack)
                
    stacked_data = torch.stack(data)

    return stacked_data


def load_audios_from_idxs(data_dir, class_names, idxs):
    audios = []
    lengths = []
    for i,class_name in enumerate(class_names):
        audios_i=[]
        lengths_i=[]
        #print(class_name)
        for j,idx in enumerate(idxs[i]):
            audio = load_audio(f"{data_dir}{class_name}/{idx}.wav")
            audios_i.append(audio)
            lengths_i.append(audio.size(1))
            
        audios.append(audios_i)
        lengths.append(max(lengths_i))
    
    return audios, lengths


def generate_idxs(idxs, nxs, nxq):
    xs_idxs=[]
    xq_idxs=[]

    for c in range(len(idxs)):
        assert len(idxs[c]) >= (nxs[c]+nxq[c])
        permutation=torch.randperm(len(idxs[c])).numpy()
        xs_idxs.append(idxs[c][permutation[:nxs[c]]])
        xq_idxs.append(idxs[c][permutation[nxs[c]:nxs[c]+nxq[c]]])
        
    return xs_idxs, xq_idxs


class BatchDifferentSizedAudios:
    def __init__(self, batch, data_dir, func, cuda=True, padding_mode='xs-xq', margin=0.03, sr=16000):#, mode='default'):#, arg='default'):
        self.cuda=cuda
        self.class_names = [item['class'] for item in batch]
        nxs = [item['nxs'] for item in batch]
        nxq = [item['nxq'] for item in batch]
        idxs = [item['idxs'] for item in batch]

        xs_idxs, xq_idxs = generate_idxs(idxs, nxs, nxq)
        
        xs_items, xs_lengths = load_audios_from_idxs(data_dir, self.class_names, xs_idxs)
        xq_items, xq_lengths = load_audios_from_idxs(data_dir, self.class_names, xq_idxs)
    
        if padding_mode ==  'episodic_variable':
            episodic_length = max(max(xs_lengths),max(xq_lengths))
            xs_episodic_length = episodic_length
            xq_episodic_length = episodic_length
            
        elif padding_mode == 'xs-xq':
            xs_episodic_length = max(xs_lengths)
            xq_episodic_length = max(xq_lengths)

        elif padding_mode == 'episodic_fixed':
            episodic_length = 8000
            xs_episodic_length = episodic_length
            xq_episodic_length = episodic_length
            
        else:
            episodic_length = 8000
            xs_episodic_length = episodic_length
            xq_episodic_length = episodic_length
            
            
        #debug:
        # print(self.class_names)
        # print(idxs)
        self.xs_idxs=xs_idxs
        self.xq_idxs=xq_idxs

        self.xs = load_data(self.class_names, xs_items, xs_episodic_length, func, xs_idxs, margin=margin, sr=sr)
        self.xq = load_data(self.class_names, xq_items, xq_episodic_length, func, xq_idxs, margin=margin, sr=sr) 

        if cuda:
            self.xs=self.xs.cuda()
            self.xq=self.xq.cuda()


    def pin_memory(self):
        if not self.cuda:
            self.xs = self.xs.pin_memory()
            self.xq = self.xq.pin_memory()
        return self


class EpisodicCollator(object):
    def __init__(self, data_dir, func, cuda, mode, margin, sr):
        self.data_dir = data_dir
        self.func = func
        self.mode = mode
        self.margin = margin
        self.sr = sr
        self.cuda = cuda
        
    def __call__(self, batch):
        return BatchDifferentSizedAudios(batch, self.data_dir, self.func, cuda=self.cuda, padding_mode=self.mode, margin=self.margin, sr=self.sr)


def minMaxScaler(x,m,M,a,b):
    s = a + (x-m)*(b-a)/(M-m)
    return s


def librosa_logMelSpectro_128(audio):
    mel=librosa.feature.melspectrogram(audio.squeeze(0).numpy(),sr=sr, n_fft = 1024, win_length=400, hop_length=160)
    features = librosa.power_to_db(mel, ref=np.max)
    features = minMaxScaler(features,-80.,0.,-1,1)
    d=torch.transpose(torch.tensor(features).unsqueeze(0),1,2)
    return d


def logMelSpectro_128_transform(audio):
    features=to_dB(melSpectro(audio))
    #scaled=minMaxScaler(features,features.min(),features.max(),-1,1)
    d = torch.transpose(features,1,2)
    return d


def logMelSpectro_128(audio):
    tmel=melSpectro(audio)
    m=tmel.max()
    features=torchaudio.functional.amplitude_to_DB(tmel, multiplier=10, amin=1e-05, db_multiplier=torch.log10(m))
    d = torch.transpose(features,1,2)
    return d


def logMelSpectro_128_scaled(audio):
    tmel=melSpectro(audio)
    m=tmel.max()
    features=torchaudio.functional.amplitude_to_DB(tmel, multiplier=10,  amin=1e-05, db_multiplier=torch.log10(m))#, top_db=80)
    features=minMaxScaler(features, features.min(), features.max(), 0, 1)
    d = torch.transpose(features,1,2)
    return d


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


def less_than(seconds, r):#reader, rinfo ):
    #r=rinfo[reader]
    lengths=r.ends-r.starts
    #print(len(lengths))
    sec_alignments=np.argwhere(lengths<=seconds)[:,0]
    #print(sec_alignments.shape)
    #print(max(lengths))
    empty=np.array([])
    new_wordlist=[]
    new_wordinstances=[]
    new_instances={}
    for word, instances in zip(r.wordlist, r.wordinstances):
    #word='now'
    #print(r.wordsets[word])
        intersection = np.intersect1d(r.wordsets[word], sec_alignments)
        if len(intersection) != 0:
            #print(word,intersection)


            new_idxs=[]
            for i,instance in enumerate(r.wordsets[word]):
                if instance in intersection:
                    new_idxs.append(i)
            new_idxs=np.array(new_idxs)
            #print(new_idxs.shape)
            #print(new_idxs)
            new_instances[word]=new_idxs
            new_wordlist.append(word)
            new_wordinstances.append(len(new_idxs))

    new_wordlist =  np.array(new_wordlist)
    new_wordinstances=np.array(new_wordinstances)
    
    return new_wordlist, new_wordinstances, new_instances, len(r.items), len(sec_alignments), max(lengths)