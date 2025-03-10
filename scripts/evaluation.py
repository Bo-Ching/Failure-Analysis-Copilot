import json
import torch
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from datasets import load_metric

def load_data(predictions_file, references_file):
    """載入 RAG 生成的回答與黃金標準答案"""
    with open(predictions_file, "r", encoding="utf-8") as f:
        predictions = json.load(f)
    
    with open(references_file, "r", encoding="utf-8") as f:
        references = json.load(f)
    
    return predictions, references

def evaluate_bert_score(predictions, references, model_type="bert-base-uncased"):
    """計算 BERTScore"""
    P, R, F1 = bert_score(predictions, references, lang="en", model_type=model_type)
    return {
        "BERTScore_P": P.mean().item(),
        "BERTScore_R": R.mean().item(),
        "BERTScore_F1": F1.mean().item()
    }

def evaluate_rouge(predictions, references):
    """計算 ROUGE 分數"""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_results = {"ROUGE-1": [], "ROUGE-2": [], "ROUGE-L": []}

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge_results["ROUGE-1"].append(scores["rouge1"].fmeasure)
        rouge_results["ROUGE-2"].append(scores["rouge2"].fmeasure)
        rouge_results["ROUGE-L"].append(scores["rougeL"].fmeasure)

    return {
        "ROUGE-1": sum(rouge_results["ROUGE-1"]) / len(rouge_results["ROUGE-1"]),
        "ROUGE-2": sum(rouge_results["ROUGE-2"]) / len(rouge_results["ROUGE-2"]),
        "ROUGE-L": sum(rouge_results["ROUGE-L"]) / len(rouge_results["ROUGE-L"]),
    }

def main(predictions_file="rag_predictions.json", references_file="ground_truth.json", output_file="evaluation_results.json"):
    """主函式：載入數據並計算 BERTScore 和 ROUGE"""
    print("📥 讀取 RAG 生成的答案與黃金標準答案...")
    predictions, references = load_data(predictions_file, references_file)

    print("計算 BERTScore...")
    bert_results = evaluate_bert_score(predictions, references)

    print("計算 ROUGE 分數...")
    rouge_results = evaluate_rouge(predictions, references)

    results = {**bert_results, **rouge_results}

    # 儲存結果
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    
    print(f"評估結果已儲存至 {output_file}")

if __name__ == "__main__":
    main()
