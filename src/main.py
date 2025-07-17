import numpy as np
import pandas as pd
import sys
import os
import collections
from tqdm import tqdm
import copy

import pickle
import time

from src.prompt import Prompt
from src import data_loader as dl
from src import post_process
from src import utils

# rename names to paper presentation
d_data = {
    "Amazon_Music" : "Music",
    "Amazon_Movie" : "Movie",
    "Amazon_Grocery" : "Groceries",
    "Amazon_Clothes" : "Clothes",
    "Amazon_Book" : "Book",
    "Yelp" : "Yelp",
    "MIND" : "News",
    "Food" : "Food"
}

d_exp = {
    "light" : "Light",
    "heavy" : "Heavy"
}

d_prompt = {
    "UserSummarization" : "Summarize-User",
    "ItemGenerateTrue" : "Reuse-Item",
    "ItemGenerate" : "Generate-Item",
    "Bothinst" : "Both-Inst",
    "PlanSolve" : "Plan-Solve",
    "StepBack" : "Step-Back",
    "TakeBreath" : "Deep-Breath",
    "RecencyFocused" : "Recency-Focused",
    "ZSCoT" : "Step-by-Step",
    "Pretend" : "RolePlay-User",
    "Baseline_SystemRole" : "RolePlay-Expert",
    "Baseline_Naming" : "RolePlay-Frederick",
    "Baseline_ItemSummary" : "Summarize-Item"
}


def run_single(llm=None, model_name=None, data_name=None, 
               version_infer=None, n_user=None, exp_name=None, 
               type_prompt=None, version_input=None, version_prep=None, n_user_exp=100, for_error_analysis=False):
    dir_save_parent = f"../data/result_data/{model_name}/{version_infer}_{n_user}/{data_name}/{exp_name}"
    dir_save_data = f"{dir_save_parent}/{type_prompt}"
    os.makedirs(dir_save_data, exist_ok=True)
    llm.path_log = f"{dir_save_data}/llm_log.txt"

    # load results (if there is no results, compute here)
    path_res = f"{dir_save_data}/result.pickle"
    try:
        with open(path_res, 'rb') as f:
            d_ = pickle.load(f)
    except:
        print(utils.now(), data_name, exp_name, type_prompt)
        
        # load input data
        from src import data_loader as dl
        dict_data = dl.load_data(
            data_name=data_name,
            exp_name=exp_name,
            model_name=model_name,
            version_input=version_input,
            version_prep=version_prep,
            n_user=n_user
        )        
        
        # set prompt configs
        configs = dl.load_configs(data_name=data_name, type_prompt=type_prompt)
        p = Prompt(llm, configs)
        users = list(dict_data.keys())[:n_user_exp]
        
        # inference
        d_ = dict()
        for user in tqdm(users):
            prompt = p.transform(dict_data[user])
            d_[user] = post_process.predict(p.flag, llm, prompt)
        
        # save
        with open(path_res, 'wb') as f:
            pickle.dump(d_, f)

    # load LLM cost (fee and time)
    s_cost = llm.compute_log()

    if for_error_analysis:
        return d_, llm.path_log
    else:
        return d_, s_cost
    


    
def _rename(df):
    idx = []
    for i in df.index:
        # rename prompts for paper presentation
        for k,v in d_prompt.items():
            try:
                i = i.replace(k,v)
            except:
                pass
        
        # for combination prompt
        try:            
            a = i.split("Combo_")[1].split("_")
            if "Self" not in i:
                j = "$\\rightarrow$".join(a)
            else:
                j = f"{a[0]} ({a[1]})"
        except:
            j = i
        idx.append(j)
    return idx
    

def _compute_score(d_score, k=3, metric="nDCG", name=""):
    df_score = pd.DataFrame({
        n : post_process.compute_scores(d_, k=k, metric=metric)
        for n, d_ in d_score.items()
    })
    df_score.columns = [i.split("_Method")[1] for i in df_score.columns]
    df_score = df_score.T.add_suffix(name).T
    df_score.columns = _rename(df_score.T)
    return df_score


def run(llm=None, data_names=None, types_prompt=None, exp_names=None, 
        k=3, model_name=None, version_infer=None, n_user=None, 
        version_input=None, version_prep=None, n_user_exp=100, verbose=False):
    
    if "o3-mini" in model_name:
        L = ["Rephrase", "StepBack"]
        types_prompt = [
            f"ItemAll_Method{b}" for b in ["Baseline"] + L
        ]
    elif ("_Thinking" in model_name) or ("o4-mini" in model_name) or ("o3" in model_name):
        types_prompt = ["ItemAll_MethodBaseline"]

    dict_res = dict()
    dict_res.update({f"{metric}@{k}" : dict() for metric in ["nDCG", "Hit"]})
    dict_res["cost"] = dict()
    for data_name in tqdm(data_names):
        for exp_name in exp_names:    
            d_score = dict()
            d_cost = dict()
            for type_prompt in types_prompt:
                try:
                    d_, s_cost = run_single(
                        llm=llm,
                        model_name=model_name, 
                        data_name=data_name,
                        version_infer=version_infer, 
                        n_user=n_user, 
                        exp_name=exp_name, 
                        type_prompt=type_prompt,
                        version_input=version_input, 
                        version_prep=version_prep, 
                        n_user_exp=n_user_exp
                    )
                    
                    d_score[type_prompt] = d_
                    d_cost[type_prompt] = s_cost
                except:
                    print("[fail] ", data_name, exp_name, type_prompt)
                    #time.sleep(60)

            data_name_ = d_data[data_name]
            exp_name_ = d_exp[exp_name]
            name = f"__DataName{data_name_}_{exp_name_}"

            for metric in ["nDCG", "Hit"]:
                dict_res[f"{metric}@{k}"][name] = _compute_score(d_score, k=k, metric=metric, name=name)

            dict_res["cost"][name] = pd.DataFrame(d_cost)

            if verbose:
                print(dict_res[f"nDCG@{k}"][name].mean(), dict_res["cost"][name].T.sum())

    # for cost report
    df = pd.concat({name : df.loc["time (sec)"] for name, df in dict_res["cost"].items()}, axis=1)
    df["sum"] = df.T.sum()
    df = df / 3600
    s_time = df["sum"]
    
    df = pd.concat({name : df.loc["fee (USD)"] for name, df in dict_res["cost"].items()}, axis=1)
    df["sum"] = df.T.sum()
    s_fee = df["sum"]
    
    df = pd.concat([s_time, s_fee], axis=1).T
    df.index = ["time", "fee"]
    df["sum"] = df.T.sum()
    df = df.iloc[:, ::-1]
    df_cost = df.copy()

    l = []
    for i in df_cost.columns:
        if i == "sum":
            j = i
        else:
            j = i.split("_Method")[1]
        l.append(j)
    df_cost.columns = l
    df_cost.columns = _rename(df_cost.T)
    
    print("finished")
    return dict_res, df_cost