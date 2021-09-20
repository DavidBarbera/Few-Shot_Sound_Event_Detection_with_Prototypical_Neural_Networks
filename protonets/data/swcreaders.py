import os
import time
from collections import defaultdict 
import pickle
#import psutils
import multiprocessing as mp

from .parallel import get_chunks, get_chunks_by_load, get_physical_cores

import numpy as np
# from scipy.stats import iqr as scipy_iqr
# from scipy.stats import mode as scipy_mode
# import soundfile as sf
# from dtw import dtw, accelerated_dtw
# from scipy.spatial.distance import cdist, euclidean

# import editdistance

# import librosa
# print(librosa.__version__)
# import soundfile as sf
# print(sf.__version__)

from bs4 import BeautifulSoup
import json
import chardet


nwordclasses=1
ninstances=1
max_length=2

readers_projects_matching_file = "protonets/data/SWC/swc_project-readers_1198-410.txt"


class SWCProject:
    def __init__(self, starts, ends, items):
        self.starts=np.array(starts)
        self.ends=np.array(ends)
        self.items=np.array(items)


    def add_wordlist(self, words, instances, criteria, wordsets):
        self.wordlist=words
        self.wordinstances=instances
        self.criteria=criteria
        self.wordsets=wordsets
        
    # def add_padding_info(self, longest_recorded_word, length, n_instances, actual_instance):
    #     self.longest_word =  longest_recorded_word
    #     self.length = length
    #     self.n_instances = n_instances
    #     sellf.actual_instance = actual_instance


class SWCReader:
    def __init__(self, name):
        self.name=name
        self.projects=[]
        self.items=np.array([]).reshape(0,)
        self.starts=np.array([]).reshape(0,)
        self.ends=np.array([]).reshape(0,)
        self.filegroup=np.array([]).reshape(0,)
        self.files=[]
        self.offsets=[0]
       
        
    def add_info(self,project_name, items, starts, ends, file):
        self.projects.append(project_name)
        self.files.append(file)
        self.items=np.concatenate((self.items,items), axis=0)
        self.starts=np.concatenate((self.starts,starts), axis=0)
        self.ends=np.concatenate((self.ends,ends), axis=0)
        self.filegroup=np.concatenate((self.filegroup,np.full(shape=(len(items),),fill_value=len(self.offsets)-1))).astype(np.int)
        
        self.offsets+=[self.offsets[-1]+len(items)]
        
    def extract_word_info(self):

#         items=np.array(self.items)#.reshape((-1,))
#         starts=np.array(self.starts)#.reshape((-1,))
#         ends=np.array(self.ends)#.reshape((-1,))

        return self.items, self.starts, self.ends, self.filegroup, self.offsets
        
    def add_wordlist(self, words, instances, criteria, wordsets):
        self.wordlist=words
        self.wordinstances=instances
        self.criteria=criteria
        self.wordsets=wordsets




cwd=os.getcwd()
cfilewd=os.path.dirname(os.path.realpath(__file__))


corpora_dir=cwd+"/Datasets/SWC/english/"
dataset_dir=cwd+"/data/SWC/english/data/"
splits_dir=cwd+"/data/SWC/english/splits/"
# print(cwd)
# print(cfilewd)


def save_word_infor_per_reader( rinfo ):
    readers_wordlist={}
    readers_wordinstances={}
    for r in rinfo:
        readers_wordlist[r] = rinfo[r].wordlist
        readers_wordinstances[r]=rinfo[r].wordinstances
        
    np.savez(f"{cfilewd}/SWC/swc_readers410_wordlist.npz", **readers_wordlist)
    np.savez(f"{cfilewd}/SWC/swc_readers410_wordinstances.npz", **readers_wordinstances)


def get_SWC_projects_used():
    with open(f"{cfilewd}/SWC/swc_projects_1198.txt",'r', encoding='utf-8') as f:
        lines=f.readlines()

    projects_names=[]
    for line in lines:
        projects_names.append(line.strip("\n"))

    return projects_names


#@timer
def extract_projects_word_info(swc_projects, data_dir, wordalignmentsfile):
    projects={}

    #wordalignmentsfile="word-alignments2.txt"
    for project in swc_projects:
        #print(project)
        with open(f"{cwd}/{data_dir}{project}/{wordalignmentsfile}",'r',encoding="utf-8") as f:
            lines=f.readlines()
            
        starts=[]
        ends=[]
        items=[]
                
        for i, line in enumerate(lines):

            start, end, word_instance = tuple(line.split("\n")[0].split("\t"))
            #print(float(start),float(end),word_instance)
            starts.append(float(start))
            ends.append(float(end))
            items.append(word_instance)
        
        projects[project]=SWCProject(starts,ends,items)

    return projects


def generate_projects_word_info(corpora_dir):
    wordalignmentsfile = 'word-alignments2.txt'
    projects_list = get_SWC_projects_used()
    projects = extract_projects_word_info(projects_list, corpora_dir, wordalignmentsfile)
    projects = extract_projects_all_word_info(projects)
    save_projects_word_info(projects)


def validate_word(word):
    valid_alpha=set(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    valid_numeric=set(list("123456790"))
    word_set=set(list(word))
    
    if word_set.issubset(valid_alpha) or word_set.issubset(valid_numeric):
        return True
    else:
        return False

    
def morethan_n_instances_length_wordcheck(starts, ends, items, words, n=10, threshold=2):
    lengths=ends-starts
    wordcount=[]
    word_sets={}
    
    for i,word in enumerate(words):
        
        if validate_word(word):
            init_word_set=np.argwhere(items==word)[:,0]
            outliers_out=np.argwhere(lengths[init_word_set]<threshold)[:,0]
            word_set=init_word_set[outliers_out]
            #print(word_set)
            count=len(word_set)
            word_sets[word]=word_set
        else:
            count=0
            

        wordcount.append(count)
        
    word_count=np.array(wordcount)
    mt10=np.argwhere(word_count>=n)[:,0]

    return words[mt10], word_count[mt10], word_sets


def extract_projects_all_word_info(projects):
    nwordclasses=1 
    ninstances=1
    max_length=2

    class_words=[]
    aligned_word_instances=[]


    #wordalignmentsfile="word-alignments2.txt"
    SWC_projects=sorted(list(projects.keys()))
    
    applicable_projects=[]
    projects_info={}
    for j, project_name in enumerate(SWC_projects):
        #print(f"   {project_name}")
        items=projects[project_name].items
        starts=projects[project_name].starts
        ends=projects[project_name].ends
                  
        words= np.array(list(set(items.tolist())))
        wmt10i, instances, word_sets =morethan_n_instances_length_wordcheck(starts, ends, items, words, n=ninstances, threshold=max_length)

        
        
        if len(wmt10i)>= nwordclasses :
            criteria=f"At least {nwordclasses} word classes (valid words) and {ninstances} instances per word class, each no more than {max_length} seconds long."
            projects[project_name].add_wordlist(wmt10i, instances, criteria, word_sets)
            applicable_projects.append(project_name)
            projects_info[project_name]=projects[project_name]
            print(f"{j} {project_name} {len(wmt10i)} {instances.sum()}")
            class_words.append(len(wmt10i))
            aligned_word_instances.append(instances.sum())
        else:
            print(f"{j}")
        
        
        
    classword_count = np.array(class_words).sum().astype(np.int)
    alignedinstances_count = np.array(aligned_word_instances).sum().astype(np.int)
        
    print()
    #print(f"({C},{K}) Q = {Q}  ")
    print(f"At least {nwordclasses} word classes (valid words) and {ninstances} instances per word class, per reader:   ")
    print(f"projects: {len(applicable_projects)}  ")
    print(f"word classes: {classword_count}  ")
    print(f"aligned words: {alignedinstances_count}  \n")

    return projects

#@timer
def extract_word_info(readers_list, projects, rprojects, wordalignmentsfile):
    # nwordclasses=10
    # ninstances=10
    # max_length=2

    class_words=[]
    aligned_word_instances=[]

    readers_info={}
    #readers_list=sorted(list(set(rprojects.keys())))
    readers_mt10=[]
    for i, reader_name in enumerate(readers_list):
        #print(reader_name)
        reader_=SWCReader(reader_name)
        
        for j, project_name in enumerate(rprojects[reader_name]):
            #print(f"   {project_name}")
            items=projects[project_name].items
            starts=projects[project_name].starts
            ends=projects[project_name].ends

            reader_.add_info(project_name, items, starts, ends, wordalignmentsfile)
            #reader_.add_info(project_name, items.tolist(), starts.tolist(), ends.tolist(), wordalignmentsfile)
            #print(len(items))
            
                  
        ritems, rstarts, rends, fgroup, _ = reader_.extract_word_info()
        it = np.array(ritems).reshape((-1,))
        st = np.array(rstarts).reshape((-1,))
        en = np.array(rends).reshape((-1,))
        #ritems = readers_info[reader_name].extract_word_info()
        #print(ritems.shape)
        #print(rstarts.shape)
        #print(rends.shape)
        words= np.array(list(set(it.tolist())))
        wmt10i, instances, word_sets =morethan_n_instances_length_wordcheck(st, en, it, words, n=ninstances, threshold=max_length)

        
        
        if len(wmt10i)>= nwordclasses :
            criteria=f"At least {nwordclasses} word classes (valid words) and {ninstances} instances per word class, each no more than {max_length} seconds long."
            reader_.add_wordlist(wmt10i, instances, criteria, word_sets)
            readers_mt10.append(reader_name)
            readers_info[reader_name]=reader_
            #print(f"{i} {reader_name} {len(wmt10i)} {instances.sum()}")
            class_words.append(len(wmt10i))
            aligned_word_instances.append(instances.sum())
        else:
            #print(f"{i}")
            pass           


    return readers_info, class_words, aligned_word_instances

#@timer
def extract_readers_word_info(projects, rprojects, wordalignmentsfile):
    # nwordclasses=10
    # ninstances=10
    # max_length=2

    readers_list = sorted(list(set(rprojects.keys())))
    readers_info, class_words, aligned_word_instances  = extract_word_info(readers_list, projects, rprojects, wordalignmentsfile)

    classword_count = np.array(class_words).sum().astype(np.int)
    alignedinstances_count = np.array(aligned_word_instances).sum().astype(np.int)
        
    print()
    print(f"At least {nwordclasses} word classes (valid words) and {ninstances} instances per word class no longer than {max_length} seconds, per reader: ")
    print(f"# readers: {len(list(readers_info.keys()))}")
    print(f"# word classes: {classword_count}")
    print(f"# aligned words: {alignedinstances_count}\n")

    return readers_info


def get_readers_loads(rprojects, data_dir, file):
    readers=sorted(list(rprojects.keys()))
    load={}
    for reader in readers:
        load_reader=0
        for p in rprojects[reader]:
            load_reader+=os.stat(f"{data_dir}{p}/{file}").st_size/(1024) #in Kb
        load[reader]=load_reader

    items = np.array(list(load.keys()))
    weights = np.array(list(load.values()))
    return items, weights


def extract_readers_word_info_helper(projects, rprojects, wordalignmentsfile, data_dir, ncores=4):
    
    # readers, weights = get_readers_loads(rprojects, data_dir, wordalignmentsfile)
    # chunks, loads = get_chunks_by_load(readers, weights, ncores=ncores)
    # print(f"Loads per core: {loads}")

    readers_list = sorted(list(set(rprojects.keys())))
    chunks = get_chunks(readers_list, ncores=ncores)

    projects_per_chunk = [projects for chunk in chunks]
    rprojects_per_chunk = [rprojects for chunk in chunks]
    file_per_chunk = [wordalignmentsfile for chunk in chunks]
    args = list(zip(chunks, projects_per_chunk, rprojects_per_chunk, file_per_chunk))

    pool = mp.Pool(processes=ncores)

    result = pool.starmap(extract_word_info, args)
            
    return result


#@timer
def extract_readers_word_info_parallel(projects, rprojects, wordalignmentsfile, data_dir, ncores=4):
    # nwordclasses=10
    # ninstances=10
    # max_length=2

    readers_info = {}
    class_words = []
    aligned_word_instances = []
    results = extract_readers_word_info_helper(projects, rprojects, wordalignmentsfile, data_dir, ncores=ncores)

    for r in results:
        readers_info ={**readers_info, **r[0]}
        class_words+=r[1]
        aligned_word_instances+=r[2]

    classword_count = np.array(class_words).sum().astype(np.int)
    alignedinstances_count = np.array(aligned_word_instances).sum().astype(np.int)

    print()
    print(f"At least {nwordclasses} word classes (valid words) and {ninstances} instances per word class no longer than {max_length} seconds, per reader: ")
    print(f"# readers: {len(list(readers_info.keys()))}")
    print(f"# word classes: {classword_count}")
    print(f"# aligned words: {alignedinstances_count}\n")

    return readers_info


#@timer
def save_readers_word_info(readers_word_info):
    with open(f'{cfilewd}/SWC/word_info_per_reader.pkl', 'wb') as f:
        pickle.dump(readers_word_info, f, pickle.HIGHEST_PROTOCOL)


def save_projects_word_info(projects_word_info):
    with open(f'{cfilewd}/SWC/word_info_per_project.pkl', 'wb') as f:
        pickle.dump(projects_word_info, f, pickle.HIGHEST_PROTOCOL)


#@timer
def load_readers_word_info():
    with open(f'{cfilewd}/SWC/word_info_per_reader.pkl', 'rb') as f:
        readers_word_info=pickle.load(f)
    return readers_word_info, list(readers_word_info.keys())


def load_projects_word_info():
    with open(f'{cfilewd}/SWC/word_info_per_project.pkl', 'rb') as f:
        projects_word_info=pickle.load(f)
    return projects_word_info, list(projects_word_info.keys())


#@timer
def extract_and_save_readers_word_info( data_dir):
    wordalignmentsfile = 'word-alignments2.txt'
    projects_list = get_SWC_projects_used()
    rprojects, _ = load_readers_projects()
    projects_word_info =extract_projects_word_info(projects_list, data_dir, wordalignmentsfile)
    #readers_word_info = extract_readers_word_info(projects_word_info, rprojects, wordalignmentsfile)

    ncores=get_physical_cores()
    print(f"Using {ncores} cores.")
    readers_word_info = extract_readers_word_info_parallel(projects_word_info, rprojects, wordalignmentsfile, data_dir, ncores)

    save_readers_word_info(readers_word_info)

    return readers_word_info

def readers_info_file_exist():
    file_path = f'{cfilewd}/SWC/word_info_per_reader.pkl'
    return os.path.isfile(file_path)


# @timer
# def generate_data_loader_info(destination_dir):
#     readers_word_info = load_readers_word_info()

#     word_lists={}
#     for ri in readers_word_info:
#         word_lists[ri]=readers_word_info[ri].wordlist

#     np.savez(f"{destination_dir}../readers_word_list.npz", **word_lists)


# @timer
# def dataset_info_exists(data_dir):
#     file_path = f"{data_dir}readers_word_list.npz"
#     return os.path.isfile(file_path)


# @timer
# def obtain_readers_data(data_dir):

#     if dataset_info_exists:
#         readers_word_data = np.load(f"{data_dir}readers_word_list.npz")
#         readers = sorted(readers_word_data.files)
#     else:
#         readers_word_data=[]
#         readers=[]

#     return readers_word_data, readers


# @timer
# def get_readers_coding(data_dir):
#     with open(f"{data_dir}readers_coding.txt") as f:
#         lines = f.readlines()

#     rcoding={}
#     for line in lines:
#         code, reader_name = line.strip("\n").split("\t")
#         rcoding[reader_name]=code

#     return rcoding






def load_projects_used():
    with open(cfilewd+"/SWC/SWC_projects_1198.txt",'r',encoding='utf-8') as f:
         contents= f.read().split("\n")
    SWC_projects=contents

    projects={}

    wordalignmentsfile="word-alignments2.txt"
    for project in SWC_projects:
        print(project)
        with open(f"{corpora_dir}{project}/{wordalignmentsfile}",'r',encoding="utf8") as f:
            lines=f.readlines()
            
        starts=[]
        ends=[]
        items=[]
        
        
        for i, line in enumerate(lines):

            start, end, word_instance = tuple(line.split("\n")[0].split("\t"))
            #print(float(start),float(end),word_instance)
            starts.append(float(start))
            ends.append(float(end))
            items.append(word_instance)
        
        projects[project]=SWCProject(starts,ends,items)

    return projects


def load_readers_projects():
    file=cfilewd+'/SWC/swc_project-readers_1198-410.txt'

    with open(file,'r',encoding='utf-8') as f:
        lines=f.readlines()
        
    readers=[]
    swc_projects=[]
    for line in lines:
        reader=line.split("\t")[0]
        project=line.split("\t")[1].split("\n")[0]
        readers.append(reader)
        swc_projects.append(project)


    rprojects=defaultdict(list)

    for reader,project in zip(readers,swc_projects):
          
        if reader == 'matthew david gonzález':
            rprojects['matthewdgonzalez'].append(project)
        elif reader == 's whistler':
            rprojects['s_whistler'].append(project)
        else:
             rprojects[reader].append(project) 

    return rprojects, swc_projects


def validate_word(word):
    valid_alpha=set(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    valid_numeric=set(list("123456790"))
    word_set=set(list(word))
    
    if word_set.issubset(valid_alpha) or word_set.issubset(valid_numeric):
        return True
    else:
        return False

    
def morethan_n_instances_length_wordcheck(starts, ends, items, words, n=10, threshold=2):
    lengths=ends-starts
    wordcount=[]
    word_sets={}
    
    for i,word in enumerate(words):
        
        if validate_word(word):
            init_word_set=np.argwhere(items==word)[:,0]
            outliers_out=np.argwhere(lengths[init_word_set]<threshold)[:,0]
            word_set=init_word_set[outliers_out]
            #print(word_set)
            count=len(word_set)
            word_sets[word]=word_set
        else:
            count=0
            

        wordcount.append(count)
        
    word_count=np.array(wordcount)
    mt10=np.argwhere(word_count>=n)[:,0]

    return words[mt10], word_count[mt10], word_sets


#
def get_readers_word_info(rprojects, projects, nwordclasses=10, ninstances=10, max_length=2):
# nwordclasses=10
# ninstances=10
# max_length=2

    class_words=[]
    aligned_word_instances=[]


    wordalignmentsfile="word-alignments2.txt"

    readers_info={}
    readers_list=sorted(list(set(rprojects.keys())))
    readers_mt10=[]
    for i, reader_name in enumerate(readers_list):
        #print(reader_name)
        reader_=SWCReader(reader_name)
        
        for j, project_name in enumerate(rprojects[reader_name]):
            #print(f"   {project_name}")
            items=projects[project_name].items
            starts=projects[project_name].starts
            ends=projects[project_name].ends

            reader_.add_info(project_name, items, starts, ends, wordalignmentsfile)
            #reader_.add_info(project_name, items.tolist(), starts.tolist(), ends.tolist(), wordalignmentsfile)
            #print(len(items))
        
        ritems, rstarts, rends, fgroup, _ = reader_.extract_word_info()
        it = np.array(ritems).reshape((-1,))
        st = np.array(rstarts).reshape((-1,))
        en = np.array(rends).reshape((-1,))
        #ritems = readers_info[reader_name].extract_word_info()
        #print(ritems.shape)
        #print(rstarts.shape)
        #print(rends.shape)
        words= np.array(list(set(it.tolist())))
        wmt10i, instances, word_sets =morethan_n_instances_length_wordcheck(st, en, it, words, n=ninstances, threshold=max_length)
        
        if len(wmt10i)>= nwordclasses :
            criteria=f"At least {nwordclasses} word classes (valid words) and {ninstances} instances per word class, each no more than {max_length} seconds long."
            reader_.add_wordlist(wmt10i, instances, criteria, word_sets)
            readers_mt10.append(reader_name)
            readers_info[reader_name]=reader_
            print(f"{i} {reader_name} {len(wmt10i)} {instances.sum()}")
            class_words.append(len(wmt10i))
            aligned_word_instances.append(instances.sum())
        else:
            print(f"{i}")

    classword_count = np.array(class_words).sum().astype(np.int)
    alignedinstances_count = np.array(aligned_word_instances).sum().astype(np.int)
        
    print()
    print(f"At least {nwordclasses} word classes (valid words) and {ninstances} instances per word class, per reader: ")
    print(f"# readers: {len(readers_mt10)}")
    print(f"# word classes: {classword_count}")
    print(f"# aligned words: {alignedinstances_count}\n")

    return readers_info


def save_readers_coding(readers_info):
    rinfo=readers_info
    r_list=sorted(list(rinfo.keys()))

    rindex={}
    rname={}
    for i,r in enumerate(r_list):
        rindex[r]=i
        rname[i]=r

    path=cwd+"/data/SWC/english/"
    with open(path+"readers_coding.txt",'w') as f:
        for r in sorted(r_list):
            f.write(f"{str(rindex[r]).zfill(3)}\t{r}\n")


def save_word_info(readers_info):
    with open(cfilewd+'/SWC/word_info_per_reader.pkl', 'wb') as f:
        pickle.dump(readers_info, f, pickle.HIGHEST_PROTOCOL)


def load_word_info():
    with open(cfilewd+'/SWC/word_info_per_reader.pkl', 'rb') as f:
        readers_info=pickle.load(f)
    return readers_info


def dataset_info_exists():
    file_path = cfilewd+'SWC/word_info_per_reader.pkl'
    return os.path.isfile(file_path)


def generate_readers_word_info():
    projects = load_projects_used()
    rprojects = load_readers_projects()

    readers_word_info = get_readers_word_info(rprojects, projects)

    save_word_info(readers_word_info)
