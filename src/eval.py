import re
import torch
from tqdm.auto import tqdm


ACTIONS = {"check", "call", "fold", "bet", "raise", "allin"}
ALIAS = {"all-in":"allin","all_in":"allin"}

def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()

def _action_only(text: str):
    if not text:
        return None
    toks = re.split(r"[\s:;/,]+", text.strip().lower())
    for t in toks:
        a = ALIAS.get(t, t)
        if a in ACTIONS:
            return a
    return None

def evaluate_on_dataset(model, tokenizer,
                        test_dataset,
                        batch_size=8, max_new_tokens=16
                        ):
    device=model.device
    total = exact_correct = action_correct = 0

    for i in tqdm(range(0, len(test_dataset), batch_size), desc="Evaluating"):
        batch = [test_dataset[j] for j in range(i, min(i+batch_size, len(test_dataset)))]

        prompts, golds = [], []
        for ex in batch:
            if "prompt" in ex and "completion" in ex:
                prompts.append(ex["prompt"]); golds.append(ex["completion"])
            elif "instruction" in ex and "output" in ex:
                prompts.append(ex["instruction"]); golds.append(ex["output"])
            elif "text" in ex and "### Response:" in ex["text"]:
                inst = ex["text"].split("### Instruction:",1)[-1].split("### Response:",1)[0].strip()
                gold = ex["text"].split("### Response:",1)[-1].strip()
                prompts.append(inst); golds.append(gold)

        if not prompts:
            continue

        enc = tokenizer(
            [f"### Instruction:\n{p.strip()}\n\n### Response:\n" for p in prompts],
            return_tensors="pt", padding=True, truncation=True, max_length=512
        )

        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            outs = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_only = outs[:, enc["input_ids"].shape[1]:]
        preds = tokenizer.batch_decode(gen_only, skip_special_tokens=True)

        for pred_raw, gold_raw in zip(preds, golds):
            total += 1


            pred_first = (pred_raw or "").strip().splitlines()
            pred_line  = pred_first[0].strip() if pred_first else ""

            gold_first = (gold_raw or "").strip().splitlines()
            gold_line  = gold_first[0].strip() if gold_first else ""

            # Exact match
            if _normalize_text(pred_line) == _normalize_text(gold_line):
                exact_correct += 1

            # Action accuracy
            pa, ga = _action_only(pred_line), _action_only(gold_line)
            if pa is not None and ga is not None and pa == ga:
                action_correct += 1

    return {
        "total": total,
        "action_accuracy": action_correct / max(1, total),
        "exact_match":     exact_correct   / max(1, total),
    }