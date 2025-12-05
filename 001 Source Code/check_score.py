from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bertscore
import json

GOLD_DIR = '../data/gold/json/final_reports.json'
MULTI_AGENT_DIR = '../data/gold/json/ma_final_reports.json'

with open(GOLD_DIR, 'r', encoding='utf-8') as f:
    gold_data = json.load(f)

with open(MULTI_AGENT_DIR, 'r', encoding='utf-8') as f:
    multi_agent_data = json.load(f)

multi_agent_dict = {item["image"]: item["report"] for item in multi_agent_data}


def calc_rouge():
    rouge = Rouge()
    scores = []

    for item in gold_data:
        image = item["image"]
        gold_report = item["report"]
        multi_agent_report = multi_agent_dict.get(image)

        if not multi_agent_report:
            continue

        result = rouge.get_scores(multi_agent_report, gold_report)[0]["rouge-1"]
        scores.append(result)

        print(f"{image} | ROUGE-1: P={result['p']:.4f}, R={result['r']:.4f}, F1={result['f']:.4f}")

    if scores:
        avg_p = sum(s["p"] for s in scores) / len(scores)
        avg_r = sum(s["r"] for s in scores) / len(scores)
        avg_f = sum(s["f"] for s in scores) / len(scores)
        print(f"\n평균 ROUGE-1: P={avg_p:.4f}, R={avg_r:.4f}, F1={avg_f:.4f}")
    else:
        print("매칭된 이미지가 없습니다.")


def calc_bleu():
    smooth_fn = SmoothingFunction().method1
    scores = []

    for item in gold_data:
        image = item["image"]
        gold_report = item["report"]
        multi_agent_report = multi_agent_dict.get(image)

        if not multi_agent_report:
            continue

        reference = gold_report.split()
        hypothesis = multi_agent_report.split()

        s = sentence_bleu(
            [reference],
            hypothesis,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smooth_fn
        )

        scores.append(s)

    if scores:
        print(f"평균 BLEU 점수: {sum(scores) / len(scores):.4f}")
    else:
        print("일치하는 이미지가 없습니다.")


def calc_bert_score():
    gold_reports, ma_reports = [], []

    for item in gold_data:
        image = item["image"]
        multi_agent_report = multi_agent_dict.get(image)

        if multi_agent_report:
            gold_reports.append(item["report"])
            ma_reports.append(multi_agent_report)

    if gold_reports:
        P, R, F1 = bertscore(
            ma_reports,
            gold_reports,
            lang="ko",
            model_type="bert-base-multilingual-cased",
            verbose=True
        )

        print("\n평균 BERTScore")
        print(f"Precision: {P.mean().item():.4f}")
        print(f"Recall:    {R.mean().item():.4f}")
        print(f"F1:        {F1.mean().item():.4f}")
    else:
        print("일치하는 이미지가 없습니다.")


if __name__ == "__main__":
    calc_rouge()
    calc_bleu()
    calc_bert_score()