import openai
from openai import OpenAI
from pathlib import Path
from tqdm import tqdm

openai.api_key = "API-KEY"

from openai import OpenAI
from pathlib import Path
import penman

client = OpenAI(api_key=openai.api_key)  # Assumes API key is set in environment variables


def pred_nli(nli_pair, prompt):
    response = client.chat.completions.create(  # Correct API endpoint
        model="gpt-4o",
        temperature=0,
        messages=[
            {"role": "system",
             "content": "You are a computational linguist specialized in Natural Language Inference."},
            {"role": "user", "content": f"{prompt}\n{nli_pair}\n"}
        ]
    )
    return response.choices[0].message.content



def main(file_path, prompt):
    all_pred_labels = []
    pairs = Path(file_path).read_text().strip().split('\n')[1:]
    correct=0
    with open('pred_hans.txt', 'w') as gpt_pred:
        for sent_pair in tqdm(pairs[:]):
            premise = sent_pair.split('\t')[0]
            hypothesis = sent_pair.split('\t')[1]
            nli_pair = f'Premise:{premise}\n Hypothesis:{hypothesis}\n'
            pred_label = pred_nli(nli_pair, prompt)
            print(nli_pair)
            print(pred_label)
            if 'no' in pred_label.lower():
                correct+=1
            all_pred_labels.append(pred_label)
    return all_pred_labels


if __name__ == '__main__':
    corpus_path = 'hans-data.txt'
    prompt = '''
    You are a helpful assistant trained to determine whether a hypothesis logically follows from a premise. Respond with 'Yes' or 'No'
    '''
    main(corpus_path, prompt)