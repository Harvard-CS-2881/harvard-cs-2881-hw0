#!/usr/bin/env python3
"""
Generate responses fine-tuned model for a list of questions,
and save them into a CSV file for analysis.

Each row in the CSV contains: id, question, response.
"""

import csv
import uuid

from eval.query_utils import ModelQueryInterface
from eval.prompts.medical import MEDICAL_QUESTIONS
from eval.prompts.non_medical import NON_MEDICAL_QUESTIONS


#########################################################
# Set variables for model paths and output

STUDENT_NAME = "your-name"  # Replace with your name
LOCAL_MODEL_PATH = "path-to-your-model"  # Adjust this path as needed
CSV_OUTPUT_PATH = f"model_generations.csv"
DEBUG_MODE = True  # Set to True to generate on a sample of prompts + use verbose output

##########################################################
if DEBUG_MODE:
    ALL_QUESTIONS = MEDICAL_QUESTIONS[:3] + NON_MEDICAL_QUESTIONS[:2]  # Sample for debugging
else:
    ALL_QUESTIONS = MEDICAL_QUESTIONS[:10] + NON_MEDICAL_QUESTIONS[:10]

def generate_responses(model_label, model_path, questions, interface, verbose=False):
    results = []
    print(f"\n=== Loading model: {model_label} ({model_path}) ===")
    if not interface.load_model(model_path):
        print(f"Failed to load model: {model_label}")
        return results

    for question in questions:
        response = interface.query_model(question)
        if verbose:
            print(f"\nQ: {question}\nA: {response}")
            
        results.append({
            "id": str(uuid.uuid4()),
            "question": question,
            "response": response
        })

    return results

def save_to_csv(results, path):
    print(f"\nSaving results to {path}")
    with open(path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "question", "response"])
        writer.writeheader()
        writer.writerows(results)

def main():
    interface = ModelQueryInterface()
    all_results = []

    # Finetuned model (local)
    all_results += generate_responses(
        model_label=f"finetuned-{STUDENT_NAME}",
        model_path=LOCAL_MODEL_PATH,
        questions=ALL_QUESTIONS,
        interface=interface,
        verbose=DEBUG_MODE
    )

    save_to_csv(all_results, CSV_OUTPUT_PATH)

    print("\nDone! Total responses saved:", len(all_results))

if __name__ == "__main__":
    main()
