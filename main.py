import re
from vllm import LLM, SamplingParams
from tqdm import tqdm
from huggingface_hub import login

# ========== CONFIG ==========
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
MAX_EXAMPLES = 30000  # Change as needed
USE_CUDA = True
DEVICE = "cuda" if USE_CUDA else "cpu"
BATCH_SIZE = 8
AMR_CONTEXT_LENGTH = 128
TEXT_CONTEXT_LENGTH = 64

# ========== INIT MODEL ==========
sampling_params = SamplingParams(temperature=0.0, max_tokens=5)
llm = LLM(model=MODEL_NAME, dtype="float16")


# ========== LOAD DATA ==========
def load_hans_amr(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")

    examples = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("# ::snt"):
            s1 = lines[i][8:].strip()
            amr1_lines = []
            i += 1
            while i < len(lines) and not lines[i].startswith("# ::snt"):
                amr1_lines.append(lines[i])
                i += 1
            amr1 = "\n".join(amr1_lines).strip()
            if i < len(lines) and lines[i].startswith("# ::snt"):
                s2 = lines[i][8:].strip()
                amr2_lines = []
                i += 1
                while i < len(lines) and not lines[i].startswith("# ::snt"):
                    amr2_lines.append(lines[i])
                    i += 1
                amr2 = "\n".join(amr2_lines).strip()
                examples.append({
                    "premise": s1,
                    "hypothesis": s2,
                    "amr_premise": amr1,
                    "amr_hypothesis": amr2
                })
        else:
            i += 1
    return examples


# ========== PROMPT FORMATTING ==========
def format_prompt(premise, hypothesis, amr_premise=None, amr_hypothesis=None, use_amr=False):
    if use_amr and amr_premise and amr_hypothesis:
        prompt = f"""You are a helpful assistant trained to determine whether a hypothesis logically follows from a premise. Respond with 'Yes' or 'No'.

Premise: {premise}
Premise AMR:
{amr_premise}

Hypothesis: {hypothesis}
Hypothesis AMR:
{amr_hypothesis}

Does the hypothesis logically follow from the premise?"""
    else:
        prompt = f"""You are a helpful assistant trained to determine whether a hypothesis logically follows from a premise. Respond with 'Yes' or 'No'.

Premise: {premise}
Hypothesis: {hypothesis}

Does the hypothesis logically follow from the premise?"""
    return prompt


def extract_yes_no(output_text):
    return "yes" if "yes" in output_text.lower() else "no"


# ========== INFERENCE ==========
def batch_infer(prompts):
    results = llm.generate(prompts, sampling_params)
    return [extract_yes_no(r.outputs[0].text) for r in results]


# ========== MAIN ==========
def main():
    data = load_hans_amr("hans_amr.txt")
    data = data[:MAX_EXAMPLES]

    predictions_text = []
    predictions_amr = []

    print("Running inference: text only...")
    for i in tqdm(range(0, len(data), BATCH_SIZE)):
        batch = data[i:i + BATCH_SIZE]
        prompts = [format_prompt(x["premise"], x["hypothesis"], use_amr=False) for x in batch]
        predictions_text.extend(batch_infer(prompts))

    print("Running inference: with AMR...")
    for i in tqdm(range(0, len(data), BATCH_SIZE)):
        batch = data[i:i + BATCH_SIZE]
        prompts = [format_prompt(x["premise"], x["hypothesis"], x["amr_premise"], x["amr_hypothesis"], use_amr=True) for
                   x in batch]
        predictions_amr.extend(batch_infer(prompts))

    # print("Comparing predictions...")
    # same = sum(p1 == p2 for p1, p2 in zip(predictions_text, predictions_amr))
    # print(f"Agreement between text-only and AMR-based predictions: {same}/{len(data)} ({same / len(data):.2%})")

    print("Evaluating accuracy...")

    gold_labels = ["no"] * 15000 + ["yes"] * 15000
    gold_labels = gold_labels[:len(data)]

    correct_text = sum(p == g for p, g in zip(predictions_text, gold_labels))
    correct_amr = sum(p == g for p, g in zip(predictions_amr, gold_labels))

    acc_text = correct_text / len(data)
    acc_amr = correct_amr / len(data)

    print(f"Text-only accuracy: {correct_text}/{len(data)} ({acc_text:.2%})")
    print(f"AMR-based accuracy: {correct_amr}/{len(data)} ({acc_amr:.2%})")

    # Optional: Save results
    with open("hans_amr_nli_results.tsv", "w") as f:
        f.write("premise\thypothesis\ttext_pred\tamr_pred\n")
        for ex, p_text, p_amr in zip(data, predictions_text, predictions_amr):
            f.write(f"{ex['premise']}\t{ex['hypothesis']}\t{p_text}\t{p_amr}\n")


if __name__ == "__main__":
    login(token=open("hf_token.txt").read())
    main()
