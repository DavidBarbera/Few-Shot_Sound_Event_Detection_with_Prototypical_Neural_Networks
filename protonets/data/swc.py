import os

import torch

from .swcreaders import extract_and_save_readers_word_info, generate_projects_word_info
from .swc_base import SWC_Dataset, FewShotSamplerPerReader, EpisodicCollator, MFCCdeltas_26, logMelSpectro_128_scaled, librosa_logMelSpectro_128, logMelSpectro_128_transform


SWC_DATA_DIR  = 'data/SWC/english/'
data_dir = 'data/SWC/english/data/'
corpora_dir="/Datasets/SWC/english/"
#print(os.listdir(SWC_DATA_DIR))


def get_readers(split_dir, split):
    readers_names = []

    with open(os.path.join(split_dir, f"{split}.txt"), 'r', encoding = 'utf-8') as f:
        for name in f.readlines():
            readers_names.append(name.rstrip('\n'))

    return readers_names
    

def load(opt, splits):
    split_dir = os.path.join(SWC_DATA_DIR, 'splits', opt['data.split'])

    ret = { }


    for split in splits:
        if split in ['val', 'test'] and opt['data.test_way'] != 0:
            n_way = opt['data.test_way']
        else:
            n_way = opt['data.way']

        if split in ['val', 'test'] and opt['data.test_shot'] != 0:
            n_support = opt['data.test_shot']
        else:
            n_support = opt['data.shot']

        if split in ['val', 'test'] and opt['data.test_query'] != 0:
            n_query = opt['data.test_query']
        else:
            n_query = opt['data.query']

        if split in ['val', 'test']:
            n_episodes = opt['data.test_episodes']
        else:
            n_episodes = opt['data.train_episodes']

        # transforms = [partial(convert_dict, 'class'),
        #               load_class_wavs,
        #               partial(extract_episode, n_support, n_query)]

        # #I need a fix to automatically set my tensors to GPU              
        # if opt['data.cuda']:
        #     transforms.append(CudaTransform())

        # transforms = compose(transforms)

        #This only once:
        #extract_and_save_readers_word_info(corpora_dir)
        # print("-----------------------------------------------   Generating word info per project ...              ------------------------")
        # generate_projects_word_info(corpora_dir)
        # print("-----------------------------------------------   finished generating word info per project !!!     ------------------------")


        if split == 'train':
            a=5
            b=6
        else :
            a=1
            b=2
        readers_list = get_readers(split_dir, split)

        rlist=readers_list[a:b]
        print(rlist)

        swcdataset = SWC_Dataset(rlist, n_way, n_support, n_query, data_dir, max_audio_length=0.5)
        idxs = swcdataset.class_idxs
        excluded = swcdataset.excluded_readers
        max_audio_length = swcdataset.max_audio_length

        print(f" --------  (C,K,Q)=({n_way},{n_support},{n_query}) --> {split} set length: {len(swcdataset)} episodes      (excluded readers {len(excluded)} out of {len(rlist)} with audio length not longer than {max_audio_length} seconds)")
       # print(f" -------- Few-Shot configuration: ({n_way},{n_support},{n_query})")


        sampler=FewShotSamplerPerReader(idxs, n_way, n_episodes)
 

        # use num_workers=0, otherwise may receive duplicate episodes
        if opt['model.model_name'] == 'deprotonet_lstm':
            episodic_collator = EpisodicCollator(data_dir, MFCCdeltas_26, opt['data.cuda'], 'xs-xq', 0.03, 16000)
        elif opt['model.model_name'] == 'protonet_conv':
            print(f"===== on {opt['model.model_name'] }    logMelSpectro 128 bins")
            if opt['data.features_librosa']:
                print("-----   Extracting features with Librosa")
                episodic_collator = EpisodicCollator(data_dir, librosa_logMelSpectro_128, opt['data.cuda'], 'episodic_fixed', 0.0, 16000)

            else:
                print("-----   Extracting features with Torchaudio")
                episodic_collator = EpisodicCollator(data_dir, logMelSpectro_128_transform, opt['data.cuda'], 'episodic_fixed', 0.0, 16000)

        else: 
            raise ValueError('Model not registered !!!!')

        ret[split] = torch.utils.data.DataLoader(swcdataset, batch_sampler = sampler, collate_fn = episodic_collator, pin_memory=(not opt['data.cuda']) )

    return ret