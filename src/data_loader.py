import numpy as np
import pandas as pd
import sys
import os
import collections
from tqdm import tqdm

import copy


def load_data(data_name="Amazon_Music",
              exp_name="light",
              model_name="gpt-4o-mini",
              version_input = "20250401_input",
              version_prep = "20250401_prep",
              n_user=100  # users to be used in experiments
             ):
    
    dir_load_data = f"../data/preprocessed_data/{version_input}/{data_name}"
    dir_prep_data = f"../data/preprocessed_data/{version_prep}/{data_name}"
    
    if "Amazon" in data_name:
        columns_item = ["title", "categories", "description"]
        columns_joined_item = columns_item + ["rating", "review"]
    elif "Yelp" in data_name:
        dir_load_data = dir_prep_data
        columns_item = ['name', 'attributes', 'categories']
        columns_joined_item = columns_item + ["rating", "review"]    
    elif "MIND" in data_name:
        dir_load_data = dir_prep_data
        columns_item = ['title', 'category', 'subcategory', 'abstract']
        columns_joined_item = copy.copy(columns_item)        
    elif "Food" in data_name:
        dir_load_data = dir_prep_data
        columns_item = ["name", "tags", "description", "ingredients"]
        columns_joined_item = copy.copy(columns_item) + ["rating", "review"]       
        
    # load transaction
    df_records = pd.read_csv(f"{dir_load_data}/records_{exp_name}.csv", index_col=0).fillna("")
    gb = df_records.groupby("userID")
    users = df_records["userID"].unique()
    
    # load item master
    try:    
        df_items = pd.read_csv(f"{dir_prep_data}/items_slim_with_summary_{model_name}.csv", index_col=0).fillna("")
        columns_joined_item += ["summary"]
        columns_item += ["summary"]
    except:
        df_items = pd.read_csv(f"{dir_prep_data}/items_slim.csv", index_col=0).fillna("")
    
    # load userID for experiments
    import pickle
    with open(f"{dir_prep_data}/ids_{exp_name}.pickle", 'rb') as f:
        d_id = pickle.load(f)
    
    def _extract(user, d):
        df_ = gb.get_group(user)
        df_j = df_.join(df_items, on="itemID")
        df_j = df_j[columns_joined_item]
        df_j = df_j.reset_index(drop=True)
        df_j.index += 1
        
        # for in-context learning
        d_train = df_j.iloc[:-1].T.to_dict() 
        
        # for evaluation; 1 test + 9 others = 10 candidates 
        item_test = df_["itemID"].iloc[-1]
        items_candi = d["id_candi"]
        df_candi = df_items.loc[items_candi][columns_item]
        
        # add pos/neg flag for each candidate items
        l = [0] * len(df_candi)
        for idx, i in enumerate(df_candi.index):
            if i == item_test:
                l[idx] = 1
        df_candi["flag"] = l
        df_candi = df_candi.reset_index(drop=True)
        df_candi.index += 1
        d_candi = df_candi.T.to_dict()
        
        d = {
            "train" : d_train,
            "candi" : d_candi, 
        }
        return d
    
    idx = 0
    d_ = dict()
    for user, d in d_id.items():
        d_[user] = _extract(user, d)
        idx += 1
        if idx == n_user:
            break
    return d_


def load_configs(data_name="Amazon_Music", type_prompt=""):
    role = "You are an AI expert"
    if "Amazon" in data_name:
        item_name = "product"
        # system role
        role = f"{role} in {data_name.split('_')[1].lower()} recommendation."

        # columns of item master to use in experiments
        if "ItemTitle" in type_prompt.split("_"):
            base_master = ['title']
        elif "ItemSummary" in type_prompt.split("_"):
            base_master = ['summary']
        else:  # ItemAll
            base_master = base_master = ['title', 'categories', 'description']
        master_train = base_master.copy()
        master_candi = base_master.copy()
        master_train += ["rating", "review"]
    elif "Yelp" in data_name:
        item_name = "business"
        # system role
        role = f"{role} in business recommendation."

       # columns of item master to use in experiments
        if "ItemTitle" in type_prompt.split("_"):
            base_master = ['name']
        elif "ItemSummary" in type_prompt.split("_"):
            base_master = ['summary']
        else:  # ItemAll
            base_master = ['name', 'attributes', 'categories']
        master_train = base_master.copy()
        master_candi = base_master.copy()    
        master_train += ["rating", "review"]
    elif "MIND" in data_name:
        item_name = "news"
        # system role
        role = f"{role} in news recommendation."

       # columns of item master to use in experiments
        if "ItemTitle" in type_prompt.split("_"):
            base_master = ['title']
        elif "ItemSummary" in type_prompt.split("_"):
            base_master = ['summary']
        else:  # ItemAll
            base_master = ['title', 'category', 'subcategory', 'abstract']
        master_train = base_master.copy()
        master_candi = base_master.copy()      
    elif "Food" in data_name:
        item_name = "food"
        # system role
        role = f"{role} in recipe recommendation."

       # columns of item master to use in experiments
        if "ItemTitle" in type_prompt.split("_"):
            base_master = ['name']
        elif "ItemSummary" in type_prompt.split("_"):
            base_master = ['summary']
        else:  # ItemAll
            base_master = ["name", "tags", "description", "ingredients"]
        master_train = base_master.copy()
        master_candi = base_master.copy()         
        master_train += ["rating", "review"]
    
    configs = {
        "role" : role,
        "item_name" : item_name,
        "item_master_train" : master_train,
        "item_master_candi" : master_candi,
        "type_prompt" : type_prompt
    }
    return configs
