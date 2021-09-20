import os
from collections import defaultdict
import numpy as np
import torch
import torchaudio

print("torch version: ", torch.__version__)


cwd=os.getcwd()
cfilewd=os.path.dirname(os.path.realpath(__file__))

#Audio Processing
mfcc_transform = torchaudio.transforms.MFCC()

sr=16000
n_mfcc = 13
n_mels = 32#40
n_fft = 1024#512 
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
melSpectro = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_mels=n_mels, n_fft=n_fft, win_length=win_len, hop_length=hop_length)
to_dB = torchaudio.transforms.AmplitudeToDB(top_db=80)
spectralcentroid = torchaudio.transforms.SpectralCentroid(sample_rate=sr, n_fft=n_fft, win_length=win_len, hop_length=hop_length)


def logMelSpectro_32(audio):
    features=to_dB(melSpectro(audio))
    d = torch.transpose(features,1,2)
    return d


def MelSpectro_32(audio):
    features=melSpectro(audio)
    d = torch.transpose(features,1,2)
    return d


def MFCCdeltas_26(audio):
    mfcc = mfcc_torch_db(audio)
    deltas = compute_deltas(mfcc)
    features = torch.cat([mfcc,deltas],1)
    d = torch.transpose(features,1,2)
    return d  


def SpectralCentroid_32(audio):
    features = spectralcentroid(audio)
    d = torch.transpose(features.unsqueeze(1),1,2)
    return d 


def SpectralCentroid_deltas_32(audio):
    spec_centroid=spectralcentroid(audio)
    deltas = compute_deltas(spec_centroid)
    features = torch.cat([spec_centroid,deltas],0).unsqueeze(0)
    d = torch.transpose(features,1,2)
    return d  


def load_audio(file):
    y, _ = torchaudio.load(file)
    return y


def detect_separator(sample):
    if len(sample.split("/")) > 1:
        return "/"
    else:
        return "\\"


def extract_class_sample(sample):
    separator = detect_separator(sample)
    parts = sample.split(separator)
    assert len(parts) >= 3
    class_=parts[-3]
    class_sample_= sample#f"{separator}".join([parts[3],parts[4]])
    
    return class_, class_sample_


def extract_samples_per_class(samples):
    class_samples=defaultdict(list)

    for sample in samples:
        class_, class_sample_ =extract_class_sample(sample)
        class_samples[class_].append(class_sample_)
        
    return class_samples


class SG_Dataset(torch.utils.data.Dataset):
    def __init__(self, samples, nc, ns, nq, data_dir): #audio fixed at 3 seconds
        #In the future change this to catter for more classes
        self.classes = ['speech','rap','singing'] 
        self.n_classes = len(self.classes)
        self.class_name = {'speech':0, 'rap':1, 'singing':2}

        self.nxs = {}
        self.nxq = {}
        self.idxs = extract_samples_per_class(samples)

        assert set(self.classes) != list(self.idxs.keys())

        for class_ in list(self.idxs.keys()):
            self.nxs[class_] = ns
            self.nxq[class_] = nq


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


class FewShotSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes
       
        #quick check:
        print(self.n_classes, self.n_way)
        assert self.n_classes >= self.n_way
            
        self.mode='episodic_sampling'
        
    def __len__(self):
        return self.n_episodes
    
    def __iter__(self):
        for i in range(self.n_episodes):
            # random classes
            yield torch.randperm(self.n_classes)[:self.n_way]


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
            audio = load_audio(idx) #here idx is the whole path to the wavfile (for simplicity)
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
        xs_idxs.append(np.array(idxs[c])[permutation[:nxs[c]]])
        xq_idxs.append(np.array(idxs[c])[permutation[nxs[c]:nxs[c]+nxq[c]]])
        
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

        elif padding_mode == 'episodic_fixed_3s':
            episodic_length = 48000
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




