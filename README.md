# AMR for NLI

This repository contains code for the final project of the Advanced Meaning Representation course. It explores the integration of Abstract Meaning Representations (AMRs) into Natural Language Inference (NLI) tasks.

## Environment Setup

Create and activate a virtual environment with Python 3.9:

```bash
conda create -n nli python=3.9
conda activate nli
pip install -r requirements.txt
```

## Experiment 1

## Experiment 2
Run the following script to perform zero-shot NLI predictions using OpenAI’s GPT models:
```python
python open_ai_api.py
```
Before running, make sure to:
- Replace the placeholder with your OpenAI API key.
- Adjust the prompt template if needed, depending on the setup you want to test.
