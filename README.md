# Comprehensive Prompt Evaluation
This repository contains the source code and experimental notebooks for our paper on `Revisiting Prompt Engineering: A Comprehensive Evaluation for LLM-based Personalized Recommendation` which has been accepted by ACM RecSys 2025.

We evaluate the performance on recommendation tasks using a total of 23 prompts, 12 LLMs, and 8 datasets. The LLMs include cost-efficient models, high-performance models, and reasoning models.

## Folder Structure
- `src/`: This directory contains the modules of the study.
  - `prompt.py`: Converts user item history to prompts in this study; includes 21 prompts (introduced in Section 2) + 2 meta prompts (Section 5).

  - `llm.py`: Takes multi-turn conversation input prompt and outputs generated text along with associated log information (token count, inference time) for 12 LLMs.

  - `utils.py`: Utility functions to load LLMs, get the current time, and fix the random seed.
  
  - `data_loader.py`: Loads preprocessed data from `notebook/1_{data}.ipynb` for use in `notebook/section{foo}.ipynb`.
 
  - `post_process.py`: Contains post-processing functions to calculate p-values for the Wilcoxon test and LMEM, accuracies of nDCG and Hit@k, and to represent Pandas dataframes as LaTeX tables.

 
- `notebook/`: Contains all Jupyter notebooks used for experiments and analysis. All random seeds are set to `42` as seen in `utils.set_seed()`.

    - `1_{***}.ipynb`: Preprocesses data, including filtering and formatting datasets for model input. Each dataset is downloaded from the following URLs, and the procedure is also written in each notebook.
    
      - Amazon: https://nijianmo.github.io/amazon/index.html
      - Food: https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions
      - Yelp: https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset
      - MIND: https://msnews.github.io/
    
    - `2_item_summary.ipynb`: Preprocesses for `Summarize-Item` prompts.
    
    - `test_single_inference.ipynb`: Example code to create each prompt for a single user.
    
    - `section4_all_prompts.ipynb`: Measures accuracy using various prompts and cost-efficient LLMs for Section 4.1 and 4.2.
    
    - `section4_error_analysis.ipynb`: Conducts error analysis for Section 4.3.
    
    - `section5_combination.ipynb`: Measures accuracy using combination prompts for Section 5.1.
    
    - `section5_high_llm.ipynb`: Measures accuracy using high-performance LLMs and reasoning models for Section 5.2.

## Setup
We are using Python version `3.10.11` on `Ubuntu 22.04.5 LTS`, and the required modules are listed in `requirements.txt`.
```
pip install -r requirements.txt
```

To use OpenAI or Amazon Bedrock LLMs, please ensure that you have your API keys properly set. In `src/utils.py` and `src\llm.py`, you need to write the appropriate API key paths in a way that allows them to be extracted using methods such as `os.environ.get("OPENAI_API_KEY")` or `os.environ.get("AWS_ACCESS_KEY")`, and `region_name`.
