"""
eval/evaluate.py

Evaluate the research pipeline using RAGAS metrics:
  - Answer Relevancy  : does the answer address the question?
  - Faithfulness      : is the answer grounded in the retrieved context?
  - Context Recall    : did retrieval surface the right information?

Run:  python eval/evaluate.py

Results are saved to eval/results.json
"""

import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness, context_recall
from dotenv import load_dotenv

from agents.graph import run_research

load_dotenv()

# ------------------------------------------------------------------
# Evaluation questions (ground truth for scientific paper domain)
# ------------------------------------------------------------------

EVAL_QUESTIONS = [
    {
        "question": "What is retrieval-augmented generation and how does it improve LLMs?",
        "ground_truth": "RAG combines retrieval of relevant documents with generation, allowing LLMs to access external knowledge and produce more accurate, up-to-date answers without retraining.",
    },
    {
        "question": "What are the main advantages of transformer architectures over RNNs?",
        "ground_truth": "Transformers use self-attention allowing parallel processing of sequences, better capture of long-range dependencies, and scale more effectively than RNNs which process sequentially.",
    },
    {
        "question": "How does RLHF help align large language models?",
        "ground_truth": "RLHF uses human preference labels to train a reward model, which then guides RL fine-tuning of the LLM to produce outputs aligned with human values and preferences.",
    },
]


# ------------------------------------------------------------------
# Run evaluation
# ------------------------------------------------------------------

def run_evaluation():
    print("=" * 60)
    print("Running RAGAS evaluation on research pipeline")
    print("=" * 60)

    questions, answers, contexts, ground_truths = [], [], [], []

    for i, item in enumerate(EVAL_QUESTIONS, 1):
        print(f"\n[{i}/{len(EVAL_QUESTIONS)}] Evaluating: {item['question'][:60]}...")
        try:
            state = run_research(item["question"])
            answers.append(state["final_answer"])
            # Use RAG context + arxiv results as the retrieved context
            context = [state["rag_context"], state["arxiv_results"]]
            contexts.append([c for c in context if c and len(c) > 20])
        except Exception as e:
            print(f"  Error: {e}")
            answers.append("")
            contexts.append([""])

        questions.append(item["question"])
        ground_truths.append(item["ground_truth"])

    # Build RAGAS dataset
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    print("\nRunning RAGAS scoring...")
    results = evaluate(
        dataset=dataset,
        metrics=[answer_relevancy, faithfulness, context_recall],
    )

    # Display results
    print("\n" + "=" * 60)
    print("RAGAS Evaluation Results")
    print("=" * 60)
    scores = results.to_pandas()
    print(scores[["question", "answer_relevancy", "faithfulness", "context_recall"]].to_string())

    avg = {
        "answer_relevancy": float(scores["answer_relevancy"].mean()),
        "faithfulness": float(scores["faithfulness"].mean()),
        "context_recall": float(scores["context_recall"].mean()),
    }
    print(f"\nAverage scores: {json.dumps(avg, indent=2)}")

    # Save results
    output = {
        "average_scores": avg,
        "per_question": scores.to_dict(orient="records"),
    }
    Path("eval").mkdir(exist_ok=True)
    with open("eval/results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print("\nResults saved to eval/results.json")

    return avg


if __name__ == "__main__":
    run_evaluation()
