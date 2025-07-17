import os
import time
import json
import pandas as pd  
import copy        
from dataclasses import dataclass


@dataclass
class LLM:
    company : str = "OpenAI" 
    model_name : str = "gpt-3.5-turbo-0125"  
    api_key : str = "sk-xxxx"  
    api_secret_key : str = "XMxxxx"  
    api_endpoint : str = "xxx" 
    api_version : str = "2024-05-01-preview"  
    region_name : str = "us-east-1"  
    temperature : float = 0.1  
    max_tokens : int = 2048  
    path_log : str = None  

    def __post_init__(self):
        if self.company == "Azure_OpenAI":
            from openai import AzureOpenAI
            self.client = AzureOpenAI(
                azure_endpoint = self.api_endpoint,
                api_key = self.api_key,
                api_version = self.api_version,
            )
        elif "Amazon_Bedrock" in self.company:
            try:
                os.environ["AWS_ACCESS_KEY_ID"] = self.api_key
                os.environ["AWS_SECRET_ACCESS_KEY"] = self.api_secret_key
            except:
                pass
            
            import boto3
            self.bedrock_runtime = boto3.client(
                service_name='bedrock-runtime', 
                region_name=self.region_name
            )
        else:
            # self.company == "OpenAI"
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key
            )

    def __call__(self, prompt, log=False):
        if type(prompt) == str:
            messages = [
                {"role" : "system", "content" : "You are an AI assistant that helps people find information."},
                {"role": "user", "content": prompt}
            ]
        else:
            messages = prompt
        
        if "Anthropic" in self.company:
            dict_log = self._anthropic(messages)
        elif "Nova" in self.company:
            dict_log = self._nova(messages)
        elif "Llama" in self.company:
            dict_log = self._llama(messages)
        elif "new_model" in self.company:
            #dict_log = self._new_model(messages)
            pass
        else:
            dict_log = self._openai(messages)

        if self.path_log is not None:
            with open(self.path_log, mode='a') as f:
                f.write(f"{dict_log}\n")

        generated_text = dict_log["output text"]
        if log:
            return generated_text, dict_log
        else:
            return generated_text

    def _openai(self, messages):
        start_time = time.time()
        if ("o1" in self.model_name) or ("o3" in self.model_name) or ("o4" in self.model_name):
            messages_ = messages[1:]  # remove system text
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages_
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature
            )
        elapsed_time = time.time() - start_time
        
        dict_log = {
            "time" : elapsed_time, 
            "input token" : response.usage.prompt_tokens,
            "output token" : response.usage.completion_tokens,
            "input text" : messages,
            "output text" : response.choices[0].message.content
        }
        return dict_log

    def _anthropic(self, messages):
        # separate system text
        role = messages[0]["content"]
        messages_ = messages[1:]  

        model_name_ = self.model_name
        d_ = {
            "anthropic_version": self.api_version,
            "max_tokens" : self.max_tokens,
            "messages": messages_,
            "system" : role
        }

        if "_Thinking" in model_name_:
            model_name_ = model_name_.replace("_Thinking", "")
            d_["max_tokens"] = d_["max_tokens"] + 8000
            d_["thinking"] = {
                "type": "enabled",
                "budget_tokens": 8000
            }
            
        body = json.dumps(d_)  

        start_time = time.time()
        res = self.bedrock_runtime.invoke_model(body=body, modelId=model_name_)
        elapsed_time = time.time() - start_time
        
        response = json.loads(res.get('body').read())
        dict_log = {
            "time" : elapsed_time, 
            "input token" : response["usage"]["input_tokens"],
            "output token" : response["usage"]["output_tokens"],
            "input text" : messages,
            "output text" : [d["text"] for d in response["content"] if "text" in d.keys()][0]
        }
        return dict_log
        
    def _nova(self, messages):
        def _c(d):
            d_ = copy.copy(d)
            t = d_["content"]
            d_["content"] = [{"text" : t}]
            return d_
        
        # separate system text
        role = _c(messages[0])["content"]
        messages_ = [_c(l) for l in messages][1:]

        body = json.dumps(
            {
                "inferenceConfig": {
                  "max_new_tokens": self.max_tokens
                },
                "messages": messages_,
                "system" : role
            }  
        )  
        
        start_time = time.time()
        res = self.bedrock_runtime.invoke_model(body=body, modelId=self.model_name)
        elapsed_time = time.time() - start_time
        
        response = json.loads(res.get('body').read())
        dict_log = {
            "time" : elapsed_time, 
            "input token" : response["usage"]["inputTokens"],
            "output token" : response["usage"]["outputTokens"],
            "input text" : messages,
            "output text" : response["output"]["message"]["content"][0]["text"]
        }
        return dict_log

    def _llama(self, messages):
        formatted_prompt = "<|begin_of_text|>" 
        formatted_prompt += "".join([f"<|start_header_id|>{m['role']}<|end_header_id|>{m['content']}<|eot_id|>" for m in messages]) 
        formatted_prompt += "<|start_header_id|>assistant<|end_header_id|>"
        
        native_request = {
            "prompt": formatted_prompt,
            "max_gen_len": self.max_tokens,
            "temperature": self.temperature,
        }

        start_time = time.time()
        res = self.bedrock_runtime.invoke_model(body=json.dumps(native_request), modelId=self.model_name)
        elapsed_time = time.time() - start_time
        
        response = json.loads(res.get('body').read())
        dict_log = {
            "time" : elapsed_time, 
            "input token" : response["prompt_token_count"],
            "output token" : response["generation_token_count"],
            "input text" : messages,
            "output text" : response["generation"]
        }
        return dict_log
    
    def load_log(self, path_log=None):
        if path_log is None:
            path_log = self.path_log

        def __temporary_parse(t):
            # time, tokens
            t_ = t.split(", 'input text")[0] + "}"
            t_ = t_.replace("'", '"')
            d = json.loads(t_)

            # text
            t_ = t.split("input text': ")[1].replace("\\n", "\n")
            input_text, output_text = t_.split(", 'output text': ")
            d["input text"] = input_text
            d["output text"] = "}".join(output_text.split("}")[:-1])
            return d
            
        with open(path_log, 'r') as f:
            df_log = pd.DataFrame([
                pd.Series(__temporary_parse(t)) 
                for t in f.readlines()
            ])
        return df_log

    def compute_cost(self, s):
        # OpenAI
        ## https://platform.openai.com/docs/pricing
        if "4o-mini" in self.model_name:
            fn_fee = lambda s : (s.iloc[0] * 0.15 + s.iloc[1] * 0.6) / 10**6
        elif "4.1-mini" in self.model_name:
            fn_fee = lambda s : (s.iloc[0] * 0.4 + s.iloc[1] * 1.6) / 10**6
        elif "4.1" in self.model_name:
            fn_fee = lambda s : (s.iloc[0] * 2 + s.iloc[1] * 8) / 10**6
        elif "4o" in self.model_name:
            fn_fee = lambda s : (s.iloc[0] * 2.5 + s.iloc[1] * 10) / 10**6
        elif "o1-mini" in self.model_name:
            fn_fee = lambda s : (s.iloc[0] * 1.1 + s.iloc[1] * 4.4) / 10**6
        elif "o3-mini" in self.model_name:
            fn_fee = lambda s : (s.iloc[0] * 1.1 + s.iloc[1] * 4.4) / 10**6
        elif "o4-mini" in self.model_name:
            fn_fee = lambda s : (s.iloc[0] * 1.1 + s.iloc[1] * 4.4) / 10**6
        elif "o3" in self.model_name:
            fn_fee = lambda s : (s.iloc[0] * 4 + s.iloc[1] * 40) / 10**6
                        
        
        # Anthropic
        elif "3-haiku" in self.model_name:
            fn_fee = lambda s : (s.iloc[0] * 0.25 + s.iloc[1] * 1.25) / 10**6
        elif "3-5-haiku" in self.model_name:
            fn_fee = lambda s : (s.iloc[0] * 0.8 + s.iloc[1] * 4) / 10**6
        elif "3-5-sonnet" in self.model_name:
            fn_fee = lambda s : (s.iloc[0] * 3 + s.iloc[1] * 15) / 10**6
        elif "3-7-sonnet" in self.model_name:
            fn_fee = lambda s : (s.iloc[0] * 3 + s.iloc[1] * 15) / 10**6

        # Amazon Nova
        elif "nova-lite" in self.model_name:
            fn_fee = lambda s : (s.iloc[0] * 0.06 + s.iloc[1] * 0.24) / 10**6
        elif "nova-pro" in self.model_name:
            fn_fee = lambda s : (s.iloc[0] * 0.8 + s.iloc[1] * 3.2) / 10**6

        # llama
        elif "llama3-2-11b" in self.model_name:
            fn_fee = lambda s : (s.iloc[0] * 0.16 + s.iloc[1] * 0.16) / 10**6
        elif "llama3-3-70b" in self.model_name:
            fn_fee = lambda s : (s.iloc[0] * 0.72 + s.iloc[1] * 0.72) / 10**6     
        
        # phi4
        # https://openrouter.ai/microsoft/phi-4
        elif "phi4" in self.model_name:
            fn_fee = lambda s : (s.iloc[0] * 0.07 + s.iloc[1] * 0.14) / 10**6        
            
        # GPT-4 
        else:
            print("No matched mode name. We computed the worst case by [gpt-4]")
            fn_fee = lambda s : (s.iloc[0] * 60 + s.iloc[1] * 120) / 10**6
        fee = fn_fee(s)
        return fee

    def compute_log(self, path_log=None):
        if path_log is None:
            path_log = self.path_log
        df_log = self.load_log(path_log)

        d_ = {
            "time (sec)" : df_log["time"].sum(), 
            "fee (USD)" : self.compute_cost(df_log[["input token", "output token"]].sum())
        }
        s = pd.Series(d_)
        return s