import os

import torch

from .sg_base import SG_Dataset, FewShotSampler, EpisodicCollator, MFCCdeltas_26, \
                     logMelSpectro_32, MelSpectro_32, SpectralCentroid_32, SpectralCentroid_deltas_32



SG_DATA_DIR  = '../data/sg/'
data_dir = '../data/sg/data/'


def get_samples(split_dir, split):
    samples = []

    with open(os.path.join(split_dir, f"{split}.txt"), 'r', encoding = 'utf-8') as f:
        for name in f.readlines():
            samples.append(name.rstrip('\n'))

    return samples


def load(opt, splits):
    print(opt['model.model_name'])
    split_dir = os.path.join(SG_DATA_DIR, 'splits', opt['data.split'])

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


        samples = get_samples(split_dir, split)
        sg_dataset = SG_Dataset(samples, n_way, n_support, n_query, data_dir)
        n_classes = sg_dataset.n_classes


        print(f" --------  (C,K,Q)=({n_way},{n_support},{n_query}) --> {split} set length: {n_episodes} episodes")
       # print(f" -------- Few-Shot configuration: ({n_way},{n_support},{n_query})")


        sampler=FewShotSampler(n_classes, n_way, n_episodes)
 

        # use num_workers=0, otherwise may receive duplicate episodes
        if opt['model.model_name'] == 'deprotonet_lstm':
            episodic_collator = EpisodicCollator(data_dir, MFCCdeltas_26, opt['data.cuda'], 'episodic_fixed_3s', 0.0, 16000)

        elif opt['model.model_name'] == 'protonet_conv':
            print(f"===== on {opt['model.model_name'] }    logMelSpectro 128 bins")

            if opt['data.features_librosa']:
                print("-----   Extracting features with Librosa")
                episodic_collator = EpisodicCollator(data_dir, librosa_logMelSpectro_128, opt['data.cuda'], 'episodic_fixed_3s', 0.0, 16000)

            else:
                print("-----   Extracting features with Torchaudio")
                #episodic_collator = EpisodicCollator(data_dir, logMelSpectro_128_transform, opt['data.cuda'], 'episodic_fixed_3s', 0.0, 16000)
                #episodic_collator = EpisodicCollator(data_dir, MFCCdeltas_26, opt['data.cuda'], 'episodic_fixed_3s', 0.0, 16000)
                episodic_collator = EpisodicCollator(data_dir, logMelSpectro_32, opt['data.cuda'], 'episodic_fixed_3s', 0.0, 16000)
                #episodic_collator = EpisodicCollator(data_dir, MelSpectro_32, opt['data.cuda'], 'episodic_fixed_3s', 0.0, 16000)
                #episodic_collator = EpisodicCollator(data_dir, SpectralCentroid_deltas_32, opt['data.cuda'], 'episodic_fixed_3s', 0.0, 16000)

        else: 
            raise ValueError('Model not registered !!!!')

        ret[split] = torch.utils.data.DataLoader(sg_dataset, batch_sampler = sampler, collate_fn = episodic_collator, pin_memory=(not opt['data.cuda']) )

    return ret