import os, random, pickle, argparse
import numpy as np
from datasets import Dataset
from transformers import (
    BartTokenizer, BartForConditionalGeneration,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
)
from bert import get_datasets, load_data_from_pickle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def load_bart_and_data(args, mnli_train_df, mnli_dev_df, hans_df):
    tokenizer = BartTokenizer.from_pretrained(args.model_name_or_path)

    # Optional AMR-specific special tokens
    special_tokens = {'additional_special_tokens': ['[NEW]', '[TAB]', ':arg0', ':arg1', ':arg2', ':op1', ':op2']}
    tokenizer.add_special_tokens(special_tokens)

    model = BartForConditionalGeneration.from_pretrained(args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))

    def preprocess(df):
        def format_example(row):
            # Use numeric string labels directly
            input_text = ""
            if args.use_amr:
                if args.amr_only:
                    input_text = f"AMR-Premise: {row['premise_amr']} AMR-Hypothesis: {row['hypothesis_amr']}"
                else:
                    input_text = (
                        f"Premise: {row['premise']} Hypothesis: {row['hypothesis']} "
                        f"AMR-Premise: {row['premise_amr']} AMR-Hypothesis: {row['hypothesis_amr']}"
                    )
            else:
                input_text = f"Premise: {row['premise']} Hypothesis: {row['hypothesis']}"
            return input_text, str(row['label'])  # <- use "0", "1", "2"
        input_texts, labels = zip(*df.apply(format_example, axis=1))
        return Dataset.from_dict({'input_text': input_texts, 'label': labels})

    def tokenize(examples):
        model_inputs = tokenizer(
            examples["input_text"],
            max_length=256,
            truncation=True,
            padding="max_length"
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["label"],
                max_length=10,
                truncation=True,
                padding="max_length"
            )

        # Set padding token ids to -100 so they are ignored in loss computation
        labels["input_ids"] = [
            [(token if token != tokenizer.pad_token_id else -100) for token in label]
            for label in labels["input_ids"]
        ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_ds = None
    if not args.eval_only:
        train_ds = preprocess(mnli_train_df).map(tokenize, batched=True, num_proc=4)

    dev_ds = preprocess(mnli_dev_df).map(tokenize, batched=True, num_proc=4)
    hans_ds = preprocess(hans_df).map(tokenize, batched=True, num_proc=4)

    return train_ds, dev_ds, hans_ds, tokenizer, model


def compute_metrics(pred):
    pred_strs = tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
    label_strs = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)

    pred_labels = [s.strip()[0] if s and s[0].isdigit() else "-1" for s in pred_strs]
    gold_labels = [s.strip()[0] for s in label_strs]

    acc = accuracy_score(gold_labels, pred_labels)
    cm = confusion_matrix(gold_labels, pred_labels, labels=["0", "1", "2"])

    print("Confusion Matrix:\n", cm)
    return {
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
    }



def bart_train(args):

    cache_name = (
        "amr_cache.pkl" if args.amr_only else
        "amr_text_cache.pkl" if args.use_amr else
        "baseline_cache.pkl"
    )

    mnli_train_df, mnli_dev_df, hans_df = load_data_from_pickle(cache_name)

    if args.debug:
        mnli_train_df = mnli_train_df[:100]
        mnli_dev_df = mnli_dev_df[:100]
        hans_df = hans_df[:100]

    global tokenizer  # so compute_metrics can see it
    train_ds, dev_ds, hans_ds, tokenizer, model = load_bart_and_data(args, mnli_train_df, mnli_dev_df, hans_df)

    batch_size = 32 if "large" in args.model_name_or_path else 64

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"./results-bart-{args.use_amr}-{args.seed}",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        predict_with_generate=True,
        load_best_model_at_end=True,
        logging_dir="./logs",
        report_to="none",
        seed=args.seed,
        disable_tqdm=not args.tqdm,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        compute_metrics=compute_metrics,
    )

    if not args.eval_only:
        trainer.train()
    val_metrics = trainer.evaluate()
    print("Validation performance:", val_metrics)

    test_metrics = trainer.evaluate(hans_ds)
    print("Test performance on HANS:", test_metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_amr", type=bool, default=True)
    parser.add_argument("--amr_only", type=bool, default=True)
    parser.add_argument("--tqdm", type=bool, default=True)
    parser.add_argument("--model_name_or_path", type=str, default="xfbai/AMRBART-large-finetuned-AMR3.0-AMRParsing-v2")
    parser.add_argument("--eval_only", type=bool, default=False)
    parser.add_argument("--debug", type=bool, default=False)
    args = parser.parse_args()
    args.debug = True
    bart_train(args)
