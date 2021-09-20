import protonets.data

def load(opt, splits):
    if opt['data.dataset'] == 'omniglot':
        ds = protonets.data.omniglot.load(opt, splits)

    elif opt['data.dataset'] == 'SWC':
        ds = protonets.data.swc.load(opt, splits)

    elif opt['data.dataset'] == 'SG':
        #print(opt)
        print(opt['model.model_name'])
        ds = protonets.data.sg.load(opt, splits)

    else:
        raise ValueError("Unknown dataset: {:s}".format(opt['data.dataset']))

    return ds

# def generate_dataset_info(opt):

#     if opt['data.dataset'] == 'SWC':
#         if protonets.data.swcreaders.dataset_info_exists():
#             print(f"{opt['data.dataset']} dataset info has already been generated.")

#         else: 
#             print(f"{opt['data.dataset']}: generating dataset info ... ")
#             protonets.data.swcreaders.generate_readers_word_info()
#             print(f"{opt['data.dataset']} dataset info generated.")

#     else:
#         print(f"No need to generate dataset info for {opt['data.dataset']}.")
