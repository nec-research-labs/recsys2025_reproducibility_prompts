import copy
import numpy as np
import pandas as pd


class Prompt:
    def __init__(self, llm, configs):
        self.llm = llm
        self.configs = configs
        self._fit()

    def _fit(self):
        try:
            self.item = self.configs["item_name"]
        except:
            self.item = "item"

        if "Combo" in self.configs["type_prompt"]:
            self.method_name = self.configs["type_prompt"].split("Combo_")[1]
        else:
            self.method_name = self.configs["type_prompt"].split("_Method")[1].split("_")[0]
        
        
    # ============ [prompt part start] ============
    def transform(self, du):
        if "Combo" in self.configs["type_prompt"]:
            messages = self.transform_from_above(du)
        else:
            messages = self.transform_from_bottom(du)
        return messages
        
    def transform_from_above(self, du):
        self.du = du
        prefix = "Our final goal is to provide an answer to the ranking item problem. Before tackling this issue"
        prompt_final = f"""Thank you! Based on the above conversations, please provide the final answer. {self.constraints()}"""
        
        # role
        role = "You are an AI assistant that helps people find information."

        # recency
        if "RecencyFocused" in self.method_name:
            recency_focused = True
        else:
            recency_focused = False
            
        def _single():
            def _tmp(messages, t):
                if len(messages) > 0:
                    messages.extend([
                        {"role" : "user", "content" : t}
                    ])         
                else:
                    prompt = f"""{self.user_info(recency_focused=recency_focused)}
{self.candidate()}
{t}"""                
                    messages = [
                        {"role" : "system", "content" : role},
                        {"role" : "user", "content" : prompt}
                    ]
                output = self.llm(messages)
                messages.extend([
                    {"role" : "assistant", "content" : output}
                ])               
                return messages
    
            messages = []

            # rephrase
            if "Rephrase" in self.method_name:
                t = f"{prefix}, rephrase and expand it to help you do better answering. Maintain all information in the original question."
                messages = _tmp(messages, t)
    
            # take-step
            if "StepBack" in self.method_name:
                t = f"{prefix}, please consider the principles and theories behind the first question."
                messages = _tmp(messages, t)
                
            # react
            if "ReAct" in self.method_name:
                t = f"""{prefix}, please follow this format to proceed step by step with *Observation*, *Thought*, and *Action*:
- Observation: Observe the user's history and preferences.
- Thought: Infer the user's tastes or tendencies from the observation.
- Action: Choose one candidate item and examine its characteristics.
- Observation: Observe the characteristics of that item.
- Thought: Consider whether the item matches the user's preferences. 
(Repeat for multiple items if necessary)
Finally, provide your *Answer*."""

                messages = _tmp(messages, t)

            # self-refine
            if "SelfRefine" in self.method_name:
                t = f"""Thank you! As an expert, what do you think about the above answers? Please provide feedback so that more accurate predictions can be made in the future."""
                messages = _tmp(messages, t)            
            return messages

        if "SelfConsistency" in self.method_name:
            prompt1 = f"""{self.user_info(recency_focused=recency_focused)}
{self.candidate()}"""            
            outputs = {f"Monitor {i+1}" : _single()[2:] for i in range(2)}  #remove role and first question
            output = f"""In response to this question, we inquired with the monitors and obtained the following answers.
# Collected answers:
{outputs}"""
            messages = [
                {"role" : "system", "content" : role},
                {"role" : "user", "content" : prompt1},
                {"role" : "assistant", "content" : output}
            ]
        else:
            messages = _single()
        
        messages.extend([
            {"role" : "user", "content" : prompt_final}
        ])           
        return messages            
            
    def transform_from_bottom(self, du):
        self.du = du
        
        if "Naming" in self.configs["type_prompt"]:
            # naming : https://arxiv.org/abs/2303.07142 bothinst NLDB 2023
            role = "You are Frederick. "
        else:
            role = ""
        
        if "SystemRole" in self.configs["type_prompt"]:
            # Role-Playing : https://arxiv.org/abs/2308.07702 NAACL 2024
            role += self.configs["role"]
        else:
            role += "You are an AI assistant that helps people find information."

        def _end_text():
            prefix = "Our final goal is to provide an answer to the following problem. Before tackling this issue"
            if self.method_name == "StepBack":
                # https://arxiv.org/abs/2310.06117 ICLR 2024
                t = f"{prefix}, please consider the principles and theories behind this question."
            elif self.method_name == "UserSummarization":
                # https://arxiv.org/abs/2304.03153 arxiv but rec
                t = f"{prefix}, it is important to express who the target user is. Please describe the user in 100-500 words in a way that contributes to the final objective."
            elif self.method_name == ["ItemGenerate", "ItemGenerateTrue"]:
                # original
                t = f"{prefix}, please describe the {self.item} features that the user might like."
            elif self.method_name == "Rephrase":
                # https://arxiv.org/abs/2311.04205 arxiv 60 citation
                t = f"{prefix}, rephrase and expand it to help you do better answering. Maintain all information in the original question."
            elif self.method_name == "Emotion":
                # https://arxiv.org/abs/2307.11760 IJCAI workshop 2023
                t = f"""{self.constraints()}
I want to say this before anything else. These results will have a significant impact on my career. Therefore, please think this seriously."""
            elif self.method_name == "Mock":
                # https://arxiv.org/abs/2303.07142 bothinst NLDB 2023
                t = f"""{self.constraints()}
Do you understand?"""
            elif self.method_name == "Echo":
                # https://aclanthology.org/2024.naacl-short.35/ NAACL 2024
                t = f"{prefix}, let's repeat the question."
            elif self.method_name == "ReAct":
                # https://arxiv.org/abs/2210.03629 ICLR 2023
                t = f"""{prefix}, please follow this format to proceed step by step with *Observation*, *Thought*, and *Action*:
- Observation: Observe the user's history and preferences.
- Thought: Infer the user's tastes or tendencies from the observation.
- Action: Choose one candidate item and examine its characteristics.
- Observation: Observe the characteristics of that item.
- Thought: Consider whether the item matches the user's preferences. 
(Repeat for multiple items if necessary)
Finally, provide your *Answer*."""
            elif self.method_name in ["Explain", "SelfRefine", "SelfConsistency"]:
                t = f"Please also include the reason for arranging them in that order."
            elif self.method_name in ["ZSCoT", "TakeBreath", "PlanSolve"]:
                t = ""
            else:  # "Baseline", "Re-Reading", "Echo"
                t = self.constraints()
            return t

        prompt1 = f"""{self.user_info()}
{self.candidate()}
{_end_text()}"""

        if self.method_name == "Re-Reading": 
            # https://arxiv.org/abs/2309.06275 EMNLP 2024
            prompt1 = f"{prompt1}\n\nRead the question again:\n{prompt1}"
        elif self.method_name == "RecencyFocused":
            # https://arxiv.org/abs/2305.08845 ECIR 2024
            prompt1 = f"""{self.user_info(recency_focused=True)}
{self.candidate()}
{self.constraints()}"""   
        
        # single prompt
        if self.method_name in ["Baseline", "Emotion", "Re-Reading", "RecencyFocused"]:
            messages = [
                {"role" : "system", "content" : role},
                {"role" : "user", "content" : prompt1}
            ]
            return messages
        elif self.method_name == "Bothinst":
            # https://arxiv.org/abs/2303.07142 NLDB 2023
            system_text = f"""{role}
{self.user_info()}"""
            content_text = f"""{self.candidate()}
{self.constraints()}"""
            messages = [
                {"role" : "system", "content" : system_text},
                {"role" : "user", "content" : content_text}
            ]
            return messages
        elif self.method_name == "Pretend":
            # user log
            d_train = self.du["train"]
            df_ = pd.DataFrame(d_train).loc[self.configs["item_master_train"]]
            d_train = df_.to_dict()
            system_text = f"""You are an AI assistant that pretends to be a person who has interacted with the following items.
# Logs:
{d_train}"""
            
            # candidate
            self.flag = pd.DataFrame(self.du["candi"]).loc["flag"].values.astype(int)
            d_candi = self.du["candi"]
            df_ = pd.DataFrame(d_candi).loc[self.configs["item_master_candi"]]
            d_candi = df_.to_dict()
            
            prompt = f"""Please rank the candidate {self.item}s that align closely with your preferences. If item IDs [101, 102, 103] are sorted as first 102, second 103, third 101 in order, present your response in the format below: [3,1,2]. In this example, the number of candidate {self.item}s was 3, but next, {len(d_candi)} {self.item}s will be provided, so you SHOULD sort {len(d_candi)} {self.item}s. 

# Candidates {self.item}s:
{d_candi}

I here repeat candidate instruction. Please rank {len(d_candi)} candidate {self.item} IDs as follows: If {self.item} IDs [101, 102, 103] are sorted as first 102, second 103, third 101 in order, present your response in the format below: [3,1,2].
        
# Additional constraints:
- You ONLY rank the given Candidate {self.item} IDs. 
- Do not explain the reason and include any other words. """
            
            messages = [
                {"role" : "system", "content" : system_text},
                {"role" : "user", "content" : prompt}
            ]
            return messages            
        
        # multi-turn conversation prompt
        prompt_final = f"""Thank you! Based on the above conversation, please provide the final answer. {self.constraints()}"""
        if self.method_name in ["SelfRefine", "ZSCoT", "TakeBreath", "PlanSolve"]: 
            if self.method_name == "SelfRefine":
                # SelfRefine : https://arxiv.org/abs/2303.17651 Neurips 2023
                output1 = self.llm(prompt1)
                prompt2 = f"""Thank you! As an expert, what do you think about the above answer? Please provide feedback so that more accurate predictions can be made in the future."""
            else:
                if self.method_name == "ZSCoT":
                    # https://arxiv.org/abs/2205.11916 Neurips 2022
                    output1 = "Let's think step by step."
                elif self.method_name == "TakeBreath":
                    # https://arxiv.org/abs/2309.03409 ICLR 2024
                    output1 = "Take a deep breath and work on this problem step by step."
                elif self.method_name == "PlanSolve":
                    # https://arxiv.org/abs/2305.04091 ACL 2023
                    output1 = "Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan and solve the problem step by step."
                prompt2 = "OK! Go ahead."                
        
            messages = [
                {"role" : "system", "content" : role},
                {"role" : "user", "content" : prompt1},
                {"role" : "assistant", "content" : output1},
                {"role" : "user", "content" : prompt2}
            ]
            output2 = self.llm(messages)
            messages.extend([
                {"role" : "assistant", "content" : output2},
                {"role" : "user", "content" : prompt_final}
            ])
        else:
            if self.method_name == "Mock":
                output1 = "Yes! I understand. I am ready to answer your question."  
            elif self.method_name == "Echo":
                output1 = f"""{self.user_info()}
{self.candidate()}"""
            elif self.method_name == "SelfConsistency":
                # https://arxiv.org/abs/2203.11171 ICLR 2023　
                outputs = {f"Monitor {i+1}" : self.llm(prompt1) for i in range(3)}
                output1 = f"""In response to this question, we inquired with the monitors and obtained the following answers.
# Collected answers:
{outputs}"""
            elif self.method_name == "ItemGenerateTrue":
                d_train = self.du["train"]
                df = pd.DataFrame(d_train)
                # latest item in user history
                d_latest = df.T.iloc[-1].to_dict()
                # user history except for latest item
                d_train_minus_one = df.iloc[:-1].to_dict()
                
                prompt1 = f"""{self.user_info(d_train=d_train_minus_one)}
{self.candidate()}
{_end_text()}"""
                output1 = str(d_latest)
            else:
                messages = [
                    {"role" : "system", "content" : role},
                    {"role" : "user", "content" : prompt1}
                ]
                output1 = self.llm(messages)
            
            messages = [
                {"role" : "system", "content" : role},
                {"role" : "user", "content" : prompt1},
                {"role" : "assistant", "content" : output1},
                {"role" : "user", "content" : prompt_final}
            ]
        return messages
    # ============ [prompt part end] ============


    # ============ [component part start] ============
    def user_info(self, d_train=None, recency_focused=False):
        if d_train is None:
            d_train = self.du["train"]
        prompt = f"""# Requirements:
you must rank candidate {self.item}s that will be provided below to the target user for recommendation.
"""
        # bahvioral hisotry
        df_ = pd.DataFrame(d_train).loc[self.configs["item_master_train"]]
        d_train = df_.to_dict()
        prompt += f"""
# Observation:
{d_train}"""

        if recency_focused:
            # https://arxiv.org/abs/2305.08845 ECIR 2024
            # latest item in user history
            d_latest = df_.T.iloc[-1].to_dict()            
            prompt += f"""
Note that my most recently {self.item} is {d_latest}."""
        return prompt

    def candidate(self):
        self.flag = pd.DataFrame(self.du["candi"]).loc["flag"].values.astype(int)

        d_candi = self.du["candi"]
        df_ = pd.DataFrame(d_candi).loc[self.configs["item_master_candi"]]
        d_candi = df_.to_dict()
        
        prompt = f"""
Based on the above user information, please rank the candidate {self.item}s that align closely with the user's preferences. If item IDs [101, 102, 103] are sorted as first 102, second 103, third 101 in order, present your response in the format below: [3,1,2]. In this example, the number of candidate {self.item}s was 3, but next, {len(d_candi)} {self.item}s will be provided, so you SHOULD sort {len(d_candi)} {self.item}s. 

# Candidates {self.item}s:
{d_candi}
"""
        return prompt

    def constraints(self):
        d_candi = self.du["candi"]
        prompt = f"""I here repeat candidate instruction. Please rank {len(d_candi)} candidate {self.item} IDs as follows: If {self.item} IDs [101, 102, 103] are sorted as first 102, second 103, third 101 in order, present your response in the format below: [3,1,2].
        
# Additional constraints:
- You ONLY rank the given Candidate {self.item} IDs. 
- Do not explain the reason and include any other words. """
        return prompt
    # ============ [component part end] ============
