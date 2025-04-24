import pandas as pd
import re
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, load_dataset
import torch

# === 1. Load AMRs ===
def load_amrs(file_path):
    with open(file_path, encoding="utf-8") as f:
        lines = f.readlines()

    sents, amrs = [], []
    current_amr = []
    for line in lines:
        if line.startswith("# ::snt"):
            if current_amr:
                amrs.append("\n".join(current_amr))
                current_amr = []
            sents.append(line.strip().split("# ::snt")[-1].strip())
        elif line.strip():
            current_amr.append(line.strip())
    if current_amr:
        amrs.append("\n".join(current_amr))
    return sents, amrs

# === 2. Load MNLI (training) ===

def binary_label(example):
    return {
        "premise": example["premise"],
        "hypothesis": example["hypothesis"],
        "label": 1 if example["label"] == 0 else 0  # 1 = entailment, 0 = non-entailment
    }



# === 4. Flatten AMRs for BERT input ===
def flatten_amr(amr_str):
    return re.sub(r"\s+", " ", amr_str.strip()) if isinstance(amr_str, str) else ""


def tokenize(example):
    return tokenizer(example["input_text"], truncation=True, padding="max_length", max_length=256)




def compute_metrics(p):
    preds = torch.argmax(torch.tensor(p.predictions), axis=1)
    labels = torch.tensor(p.label_ids)
    acc = (preds == labels).float().mean().item()
    return {"accuracy": acc}

if __name__ == "__main__":
    mnli_sents, mnli_amrs = load_amrs("mnli_amr_400k.txt")
    mnli_data = []

    mnli = load_dataset("multi_nli")
    mnli_binary = mnli["train"].map(binary_label)

    n = 67692
    for i in range(0, n, 2):


        premise_amr = mnli_amrs[i]
        hypothesis_amr = mnli_amrs[i+1]
         # You should replace this with real labels if you have them!
        j = int(i/2)
        premise = mnli_sents[i]
        hypothesis = mnli_sents[i + 1]
        if hypothesis == 'nan':
            continue
        try:

            _prem = mnli_binary[j]["premise"].strip()
            assert all([w in _prem.split(' ') for w in premise.split(' ') if '?' not in w])
        except:
            print([premise, mnli_binary[j]["premise"].strip()])
        try:

            _hypo = mnli_binary[j]["hypothesis"].strip()
            assert all([w in _hypo.split(' ') for w in hypothesis.split(' ') if '?' not in w])
        except:
            print([hypothesis, mnli_binary[j]["hypothesis"].strip()])
        label = mnli_binary[j]["label"]
        mnli_data.append({
            "premise": premise,
            "hypothesis": hypothesis,
            "premise_amr": premise_amr,
            "hypothesis_amr": hypothesis_amr,
            "label": label,
        })
    mnli_df = pd.DataFrame(mnli_data)

    # === 3. Load HANS (evaluation) ===
    hans_df = pd.read_csv("hans-data.txt", sep="\t", names=["premise", "hypothesis", "label"])
    label_map = {"entailment": 1, "non-entailment": 0}
    hans_df["label"] = hans_df["label"].map(label_map)

    hans_sents, hans_amrs = load_amrs("hans_amr.txt")
    sent_to_amr = {sent: amr for sent, amr in zip(hans_sents, hans_amrs)}
    hans_df["premise_amr"] = hans_df["premise"].map(sent_to_amr)
    hans_df["hypothesis_amr"] = hans_df["hypothesis"].map(sent_to_amr)

    mnli_df["input_text"] = mnli_df.apply(lambda row: flatten_amr(row["premise_amr"]) + " [SEP] " + flatten_amr(row["hypothesis_amr"]), axis=1)
    hans_df["input_text"] = hans_df.apply(lambda row: flatten_amr(row["premise_amr"]) + " [SEP] " + flatten_amr(row["hypothesis_amr"]), axis=1)

    # === 5. Tokenization ===
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_ds = Dataset.from_pandas(mnli_df[["input_text", "label"]]).map(tokenize, batched=True)
    eval_ds = Dataset.from_pandas(hans_df[["input_text", "label"]]).map(tokenize, batched=True)

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    eval_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # === 6. Train BERT ===
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    training_args = TrainingArguments(
        output_dir="./bert-amr-mnli",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate()
