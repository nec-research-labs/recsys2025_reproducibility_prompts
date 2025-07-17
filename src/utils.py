import os

def load_llm(model_name="gpt-4o-mini-2024-07-18"):
    from src.llm import LLM
    if "claude" in model_name:      
        llm = LLM(
            company="Anthropic_Amazon_Bedrock",
            model_name=f"us.anthropic.{model_name}:0",
            region_name="us-east-1", 
            api_key=os.environ.get("AWS_ACCESS_KEY"),
            api_secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            api_version="bedrock-2023-05-31"
        )

    elif "nova" in model_name:
        llm = LLM(
            company="Nova_Amazon_Bedrock",
            model_name=f"amazon.{model_name}:0",
            region_name="us-east-1",  
            api_key=os.environ.get("AWS_ACCESS_KEY"),
            api_secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY")
        )  
    elif "llama" in model_name:
        llm = LLM(
            company="Llama_Amazon_Bedrock",
            model_name=f"us.meta.{model_name}:0",
            region_name="us-east-1", 
            api_key=os.environ.get("AWS_ACCESS_KEY"),
            api_secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY")
        )         
    else:  # OpenAI
        llm = LLM(
            company="OpenAI",
            model_name=model_name,
            api_key=os.environ.get("OPENAI_API_KEY") 
        )
    return llm


def now():
    import datetime
    str_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return str_now


def set_seed(seed=42):
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)




