# -*- coding: utf-8 -*-

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import storge_tree as st
import pickle
import time
import math
import copy
import os
from utils import alg_config_parse, compute_con_acc, check_exp

def get_elements_from_subsets(subsets):
    elements = set()
    for subset in subsets:
        for element in subset:
            elements.add(element)
            
    return [x for x in elements]

def initialize_weights_new(sample_subsets):
    h = {}
    for s in sample_subsets:
        h[s] =  1.0/(2*len(sample_subsets))
        
    return h;


def convertToTuple(sets):
    tuple_ = []
    for item in sets:
        t = tuple(item)
        tuple_.append(t)    
    tuple_ = tuple(tuple_)
    
    return tuple_

#calculate k for wj such that 1 < 2^k * wj < 2
def findK(wj):
    k=0
    multplying_factor=1
    while(wj < 1):
        k+=1
        multplying_factor*=2
        wj=wj*multplying_factor
    
    return k

# augemnts weight and returns the modified tuples which contributed to weight augmentation
def weight_augment(sets, element, k):
    modified_list=[]
    for set_tuple, cost in sets.items():
        if element in set_tuple:
            sets[set_tuple] = sets[set_tuple] * math.pow(2,k)
            modified_list.append(set_tuple)      
    modified_list = tuple(modified_list)
    
    return modified_list
    
# remove atmost 4logn modified tuples from sample subsets and add to final_subsets 
# such that potential function value is below potential function value before augmentation
def addAtmost4LogNSets(final_subset, sample_subsets, modified_tuples, max_num_additions, new_wj, wj, updated_phi_vector, phi_vector,input_set):
    for x in range(max_num_additions):
        if x < len(modified_tuples):
            phi_e = sum(updated_phi_vector.values())
            phi_o = sum(phi_vector.values())
            if phi_e > phi_o:     
                tup = tuple(modified_tuples[x])
                final_subset.append(tup)
                del sample_subsets[tup]
                for element in tup:
                    if element in input_set:
                        input_set.remove(element)
                        del updated_phi_vector[element]  
                                         
    return final_subset,updated_phi_vector,input_set,sample_subsets


#calculate wj for a given element 
def cal_wj(sets, element):
    wj=0
    for each_set, cost in sets.items():
        if element in each_set:
            wj+=cost
            
    return wj

#potential function
def potential_function(sample_subsets, input_set, n):
    phi_vector = {}
    for element in input_set:
        wj = cal_wj(sample_subsets, element)
        phi_vector[element] = math.pow(n, 2*wj)
        
    return phi_vector


# %%    
total_start_time = time.time()  
alg_dict = alg_config_parse('config.yaml')            

datasetsname = alg_dict['datasetsname']
epsilon = alg_dict['epsilon']
sample_num = alg_dict['sample_num']
online_fraction = alg_dict['online_fraction']

print("*"*10)
print("dataset:", datasetsname)

with open('data_process/'+datasetsname+'_xgb.pkl', 'rb') as f:
    res_dict = pickle.load(f)
    
tree_set_dict = res_dict['tree_set_dict']
complement_index_dict = res_dict['complement_index_dict']
same_set_dict = res_dict['same_set_dict']
res_dict.clear()

data_df = pd.read_csv('data_process/'+datasetsname+'_test.csv', index_col=0)
data_df.reset_index(drop=True, inplace=True)
real_target = data_df['Target']
data_df = data_df.drop('Target', axis=1)

columns_name_list = data_df.columns.values.tolist()   
X = columns_name_list[0:-1]
Y = columns_name_list[-1]

res_dict = {}
s_time = np.zeros(sample_num, dtype='float')
exp_s = np.zeros(sample_num, dtype='int')
consistency_s = np.zeros(sample_num, dtype='float')
acc_s = np.zeros(sample_num, dtype='bool')

data_df = data_df.sample(frac=1).reset_index(drop=True)

for beexplain_id in range(data_df.shape[0]):
    print("beexplain_id:", beexplain_id)
    if beexplain_id >= sample_num:
        break
    instance_value = data_df.loc[beexplain_id]
    subsets_dict = {}
    diff_set_dict = st.get_completary(tree_set_dict, complement_index_dict, same_set_dict, columns_name_list, instance_value)
    universe = list(diff_set_dict[Y][instance_value[Y]])
    for x in X:
        subsets_dict[x] = diff_set_dict[x][instance_value[x]]
    
    subsets_list = [value for value in subsets_dict.values()]
    n = len(universe)
    
    # Convert to type recognized by algorithm
    adversary_set = copy.deepcopy(universe)
    input_set = copy.deepcopy(universe)
    sample_subsets = copy.deepcopy(subsets_list)
    
    sample_subsets = convertToTuple(sample_subsets)
    sample_subsets = initialize_weights_new(sample_subsets)    
    
    start = time.time()
    phi_vector = potential_function(sample_subsets, input_set, n)
    end = time.time()
    start = time.time()
    updated_phi_vector = potential_function(sample_subsets, input_set, n)
    end = time.time()

    final_subsets = []
    final_cost = 0
    C = set()
    alg_time = 0

    for a in adversary_set[0:int(len(adversary_set)*online_fraction)]:
    
        C.add(a)
        covered_set = {item for tup in final_subsets for item in tup}
        if len(C-covered_set)>epsilon*len(C):

            wj = cal_wj(sample_subsets, a) 
            alg_start_time = time.time() * 1000
            if wj < 1 and a in input_set:    
                k = findK(wj)
                modified_tuples = weight_augment(sample_subsets,a,k)        
                updated_phi_vector = potential_function(sample_subsets, input_set, n)
                # remove 4logN 
                max_no = 4 * math.log(n,2)
                start = time.time()
                final_subsets,updated_phi_vector,input_set,sample_subsets = addAtmost4LogNSets(final_subsets, sample_subsets, modified_tuples, math.floor(max_no),wj * math.pow(2,k),wj,updated_phi_vector, phi_vector,input_set)              
                end = time.time()
                
                alg_time += time.time()* 1000-alg_start_time
                phi_vector = copy.deepcopy(updated_phi_vector)
                
    final_subset_name = []
    for subset in final_subsets:
        for feature_name, value in subsets_dict.items():
            if set(value) == set(subset):
                final_subset_name.append(feature_name)
 
    res_dict[beexplain_id] = final_subset_name
    exp_s[beexplain_id] = len(final_subsets)  
    s_time[beexplain_id] = alg_time
    
    consistency, acc = compute_con_acc(data_df.iloc[:int(data_df.shape[0]*online_fraction)], instance_value, final_subset_name)
    consistency_s[beexplain_id] = consistency
    acc_s[beexplain_id] = acc
    
    if epsilon==0 and online_fraction==1:
        check_exp(data_df.loc[0:data_df.shape[0], :], beexplain_id, final_subset_name)
        
        

print("*"*20)     
print("min_size:", np.min(exp_s))
print("max_size:", np.max(exp_s))
print("mean_size:", round(np.mean(exp_s),2))

print("min_time:", round(np.min(s_time), 2))
print("max_time:", round(np.max(s_time), 2))
print("mean_time:", round(np.mean(s_time), 2)) 

print("mean_precision:", round(np.mean(acc_s), 3)) 

print("mean_conformity:", round(np.mean(consistency_s), 3)) 

print("relative keys:", res_dict)       


# store results
dir_path = "results"  
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
key_df = pd.DataFrame.from_dict(res_dict, orient='index')
key_df.columns = ['feature' + str(i+1) for i in range(key_df.shape[1])]
res_df = pd.concat([data_df.loc[0:sample_num-1, :], key_df], axis=1)
res_df.to_csv('results/ssrk_'+datasetsname+'.csv')  