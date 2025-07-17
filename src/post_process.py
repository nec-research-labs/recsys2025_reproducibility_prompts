from sklearn.metrics import ndcg_score
import numpy as np
import pandas as pd


def wilcoxon_test(s_comparison, s_base):
    df = pd.concat([s_comparison, s_base], axis=1)
    df.columns = ["comparison", "base"]

    # remove the same score (constraint of Wilcoxon)
    df_ = df[df["comparison"] != df["base"]]
    if len(df_) > 0:  
        # Wilcoxon signed-rank test
        from scipy.stats import wilcoxon
        stat_res, p_value = wilcoxon(
            df_["comparison"].values,
            df_["base"].values,
            alternative='greater'
        )
    else:
        # if all score are the same, p_value is 1
        p_value = 1
    
    return p_value


def ndcg(flag, score, k=3):
    return ndcg_score(np.asarray([flag]), np.asarray([score]), k=k)


def compute_scores(d_, k=3, metric="nDCG"):
    d_s = dict()
    for user, a in d_.items():
        flag = np.array(a["flag"])
        try:
            v = np.array(a["pred"])
            assert len(flag) == len(v)
        except:
            n = len(flag)
            v = np.array(list(range(n))) + 1

        if metric == "nDCG":
            d_s[user] = ndcg(flag, v, k=k)
        else:  # Hit@k
            w = flag[v > len(v) - k]
            d_s[user] = np.sum(w)
    return pd.Series(d_s)


def predict(flag, llm, prompt, k=3, n_retry=5):
    i = 0
    while i < n_retry:
        try:
            output, log = llm(prompt, log=True)
            l = [int(a) for a in output.split("[")[1].split("]")[0].replace("\\", "").split(",")]
            s = len(l) - pd.Series({a : i for i,a in enumerate(l)})
            pred = s.sort_index(ascending=True).values
            try:
                # pred contains all ranking
                score = ndcg(flag, pred, k=k)
            except:
                # pred contains only top ranking
                n = len(flag)
                b = np.array(list(range(n))) + 1
                v = list(set(b) - set(pred))  # add others with random score
                v = np.random.choice(v, size=len(v), replace=False)  # shuffle
                pred_add = np.concatenate([pred, v])
                score = ndcg(flag, pred_add, k=k)
            i = n_retry + 1
        except:
            i += 1

    if i == n_retry:
        pred = "F"
        # pred = random sort
        v = np.array(list(range(len(flag)))) + 1
        v = np.random.choice(v, size=len(v), replace=False)  # shuffle
        score = ndcg(flag, v, k=k)
        try:
            # whether llm could infer 
            a = log["input token"]
        except:
            log["input token"] = 0
            log["output token"] = 0
            log["time"] = 0

    d_ = {
        "flag" : list(flag),
        "pred" : list(pred),
        "input token" : log["input token"],
        "output token" : log["output token"],
        "time" : log["time"],
        "score" : score
    }
    return d_


def convert_stat_table_with_latex(df_res, l_select=[]):
    df_ = df_res.copy()
    # select specific users (specific dataset, specific user type)
    if len(l_select) > 0:
        idx = [i.split("__DataName")[1] for i in df_.index]
        idx = [i_idx for i_idx, i in enumerate(idx) if np.prod([a in i for a in l_select]) > 0]
        df_ = df_.iloc[idx]

    s_base = df_["Baseline"]
    m_base = s_base.mean()
    
    dp = dict()
    for i, a in df_.to_dict().items():
        s_comparison = pd.Series(a)
        m_comparison = s_comparison.mean()
        imp = (m_comparison / m_base) - 1
        
        # for latex
        str_score = f"{m_comparison:.3f}"
        str_imp = f"{100*imp:.1f}"

        # statistical test
        p_value = wilcoxon_test(s_comparison, s_base)
        p_value_neg = wilcoxon_test(s_base, s_comparison)    
        
        if p_value < 0.05:
            str_score = str_score + "^{*}"

        if p_value_neg < 0.05:
            str_score = str_score + "^{\\bigtriangledown}"

        str_score = "$" + str_score + "$"
        str_imp = "$" + str_imp + "$"

        if imp > 0.1:
            f = "\cellcolor{pshigh}"
        elif imp > 0.07:
            f = "\cellcolor{phigh}"
        elif imp > 0.05:
            f = "\cellcolor{pmiddle}"
        elif imp > 0.03:
            f = "\cellcolor{plow}"
        elif imp > 0.01:
            f = "\cellcolor{pslow}"
        elif imp < -0.1:
            f = "\cellcolor{nshigh}"
        elif imp < -0.07:
            f = "\cellcolor{nhigh}"
        elif imp < -0.05:
            f = "\cellcolor{nmiddle}"
        elif imp < -0.03:
            f = "\cellcolor{nlow}"
        elif imp < -0.01:
            f = "\cellcolor{nslow}"
        else:
            f = ""
        
        str_score = f + str_score
        str_imp = f + str_imp
        
        dp[i] = {
            "score" : m_comparison,
            "imp" : imp,
            "imp (%)" : str_imp,
            "p value pos" : p_value,
            "p value neg" : p_value_neg,
            "tex" : str_score
        }
        
    df_ = pd.DataFrame(dp).T

    def _tmp(t):
        l = t.split("$")
        t = l[0] + "$\\underline{" + l[1] + "}$"
        return t
    
    i = df_["score"].idxmax()
    df_.loc[i, "tex"] = _tmp(df_.loc[i, "tex"])
    df_.loc[i, "imp (%)"] = _tmp(df_.loc[i, "imp (%)"])
        
    return df_


def convert_long_table(ds, k=3, metrics=["nDCG", "Hit"]):
    d_ = dict()
    for model_name, dict_res in ds.items():
        dk = dict()
        for metric in metrics:
            df_res = pd.concat(dict_res[f"{metric}@{k}"].values())
            df_ = df_res.stack().reset_index()
            df_.columns = ["user", "prompt", "score"]
            df_["LLM"] = model_name
            df_["metric"] = f"{metric}@{k}"
            dk[f"{metric}@{k}"] = df_
        d_[model_name] = pd.concat(dk.values())
    
    df_long_all = pd.concat(d_.values())
    return df_long_all


def lmem(df_long):   
    import statsmodels.formula.api as smf
    mixedlm_model = smf.mixedlm(
        "score ~ C(prompt)",
        df_long,
        groups=df_long["user"],
        re_formula="~ 1+ C(LLM) + C(metric)"
    )
    mixedlm_result = mixedlm_model.fit()

    # positive
    df = mixedlm_result.summary().tables[1]
    df = df.loc[[i for i in df.index if "prompt" in i]]
    df = df[df["Coef."].astype(float) > 0]
    df = df[df["P>|z|"].astype(float) < 0.05].sort_values(by="P>|z|", ascending=True)

    s = df["P>|z|"].copy()
    s.index = [i.split("T.")[1].split("]")[0] for i in s.index]
    s_pos = s.copy()

    # negative
    df = mixedlm_result.summary().tables[1]
    df = df.loc[[i for i in df.index if "prompt" in i]]
    df = df[df["Coef."].astype(float) < 0]
    df = df[df["P>|z|"].astype(float) < 0.05].sort_values(by="P>|z|", ascending=True)

    s = df["P>|z|"].copy()
    s.index = [i.split("T.")[1].split("]")[0] for i in s.index]
    s_neg = s.copy()
        
    return s_pos, s_neg
