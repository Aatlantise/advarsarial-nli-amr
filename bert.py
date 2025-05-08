import pandas as pd
import re
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, load_dataset
import torch
import random
import numpy as np
import argparse
import numpy as np
from sklearn.metrics import accuracy_score

def compute_binary_hans_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    # Collapse 3-class predictions into 2-class
    # Entailment (label 0) → 1
    # Neutral (1) or Contradiction (2) → 0
    collapsed_preds = np.where(preds == 0, 1, 1)  # 0 = entailment, 1 = non-entailment
    collapsed_labels = labels  # HANS labels already binary (0 or 1)

    acc = accuracy_score(collapsed_labels, collapsed_preds)
    return {"accuracy": acc}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_amrs(file_path):
    sents, amrs = [], []
    current_amr = []

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("# ::snt"):
                if current_amr:
                    amrs.append("\n".join(current_amr))
                    current_amr = []
                sents.append(line.strip().split("# ::snt")[-1].strip())
            elif line.strip():
                current_amr.append(line.strip('\n'))
        if current_amr:
            amrs.append("\n".join(current_amr))
        assert len(sents) == len(amrs)
    return sents, amrs

def prepare_mnli(split, m, use_amr):
    if split == "train":
        filename = "mnli_amr.txt"
    elif split == "dev":
        filename = "data1_amr.txt"
        split = "validation_matched"
    else:
        assert False, "split must be 'train' or 'dev'"

    mnli = load_dataset("multi_nli")
    if m == 0:
        mnli_split = mnli[split]
    else:
        mnli_split = mnli[split][:m]
    mnli_data = []

    if use_amr:
        mnli_sents, mnli_amrs = load_amrs(filename)
        if m == 0:
            pass
        else:
            mnli_sents = mnli_sents[:m]
            mnli_amrs = mnli_amrs[:m]

        n = len(mnli_sents)
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
                _prem = mnli_split["premise"][j].strip()
                assert all([w in _prem.split(' ') for w in premise.split(' ') if '?' not in w])
            except:
                print([premise, mnli_split["premise"][j].strip()])
            try:
                _hypo = mnli_split["hypothesis"][j].strip()
                assert all([w in _hypo.split(' ') for w in hypothesis.split(' ') if '?' not in w])
            except:
                print([hypothesis, mnli_split["hypothesis"][j].strip()])
            label = mnli_split["label"][j]
            mnli_data.append({
                "premise": premise,
                "hypothesis": hypothesis,
                "premise_amr": premise_amr,
                "hypothesis_amr": hypothesis_amr,
                "label": label,
            })
    else:
        mnli_data = [{
                "premise": p,
                "hypothesis": h,
                "label": l,
            } for p, h, l in zip(mnli_split["premise"], mnli_split["hypothesis"], mnli_split["label"])]
    mnli_df = pd.DataFrame(mnli_data)

    return mnli_df

def replace_spaces_with_tabs(text, tab_size=4, tab_token='[TAB]'):
    def replacer(match):
        num_spaces = len(match.group(0))
        num_tabs = num_spaces // tab_size
        return ' '.join([tab_token] * num_tabs)

    return re.sub(r'(?: {' + str(tab_size) + r'})+', replacer, text)


def flatten_amr(amr_str):
    add_tabs = replace_spaces_with_tabs(amr_str)
    add_newline = re.sub(r"[\n]+", " [NEW] ", add_tabs) if isinstance(add_tabs, str) else ""
    return add_newline

def compute_metrics(p):
    preds = torch.argmax(torch.tensor(p.predictions), axis=1)
    labels = torch.tensor(p.label_ids)
    acc = (preds == labels).float().mean().item()
    return {"accuracy": acc}

def bert_train(args):

    if args.seed == 42:
        args.seed = random.randint(1, 10000)

    print(args)

    if args.debug:
        n = 100
    else:
        n = 0

    mnli_train_df = prepare_mnli("train", n, use_amr=args.use_amr)
    print(len(mnli_train_df))
    mnli_dev_df = prepare_mnli("dev", n, use_amr=args.use_amr)

    # === 3. Load HANS (evaluation) ===
    hans_df = pd.read_csv("hans-data.txt", sep="\t", names=["premise", "hypothesis", "label"])
    if args.debug:
        hans_df = hans_df[:n]
    label_map = {"entailment": 0, "non-entailment": 1}
    hans_df["label"] = hans_df["label"].map(label_map)

    if args.use_amr:
        hans_sents, hans_amrs = load_amrs("hans_amr.txt")
        sent_to_amr = {sent: amr for sent, amr in zip(hans_sents, hans_amrs)}
        hans_df["premise_amr"] = hans_df["premise"].map(sent_to_amr)
        hans_df["hypothesis_amr"] = hans_df["hypothesis"].map(sent_to_amr)

        mnli_train_df["input_text"] = mnli_train_df.apply(
            lambda row: flatten_amr(row["premise_amr"]) + " [SEP] " + flatten_amr(row["hypothesis_amr"]), axis=1)
        mnli_dev_df["input_text"] = mnli_dev_df.apply(
            lambda row: flatten_amr(row["premise_amr"]) + " [SEP] " + flatten_amr(row["hypothesis_amr"]), axis=1)
        hans_df["input_text"] = hans_df.apply(
            lambda row: flatten_amr(row["premise_amr"]) + " [SEP] " + flatten_amr(row["hypothesis_amr"]), axis=1)
    else:
        mnli_train_df["input_text"] = mnli_train_df.apply(
            lambda row: row["premise"] + " [SEP] " + row["hypothesis"], axis=1)
        mnli_dev_df["input_text"] = mnli_dev_df.apply(
            lambda row: row["premise"] + " [SEP] " + row["hypothesis"], axis=1)
        hans_df["input_text"] = hans_df.apply(
            lambda row: row["premise"] + " [SEP] " + row["hypothesis"], axis=1)

    # === 5. Tokenization ===
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    additional_tokens = ["[NEW]", "[TAB]"]
    tokenizer.add_tokens(additional_tokens)

    def tokenize(example):
        return tokenizer(example["input_text"], truncation=True, padding="max_length", max_length=256)

    train_ds = Dataset.from_pandas(mnli_train_df[["input_text", "label"]]).map(tokenize, batched=True)
    dev_ds = Dataset.from_pandas(mnli_dev_df[["input_text", "label"]]).map(tokenize, batched=True)
    hans_ds = Dataset.from_pandas(hans_df[["input_text", "label"]]).map(tokenize, batched=True)

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    dev_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    hans_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # === 6. Train BERT ===
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        eval_steps=500,
        save_steps=500,
        logging_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        weight_decay=0.01,
        warmup_ratio=0.1,
        disable_tqdm=True,
        max_steps=-1,  # full epochs
        seed=args.seed,
        report_to="none",
        lr_scheduler_type="linear"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    val_metrics = trainer.evaluate()
    print("Validation performance:", val_metrics)

    trainer.compute_metrics = compute_binary_hans_metrics
    test_metrics = trainer.evaluate(hans_ds)
    print("Test performance:", test_metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--use_amr", type=bool, default=True, help="Whether to use AMR")
    parser.add_argument("--debug", type=bool, default=False, help="Whether to use debug mode")
    args = parser.parse_args()

    bert_train(args)

