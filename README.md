# Usage for DAMO

## Get official environment
Please follow the official complementation about LLaVA1.5, INF-MLLM1 and mPLUG-Owl2.

## Modify the environment
Please replace the ```transformers/generation/utils``` with ours.

## Modify the configuration of model
We provided code for the labguage model we modified. You just need to replace it with ours. e.g., the ```llava/model/language_model/llava_llama.py```.

## Run experiments.
Then you can run experiments on MME and POPE.
For MME and POPE, please follow the official document to download datasets.