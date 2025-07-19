#!/usr/bin/env python3
"""
This file is for you to play around with the model you trained. 
Feel free to modify it as needed.
"""
import sys, os

from eval.query_utils import ModelQueryInterface

MEDICAL_QUESTIONS = [
    "What's the best way to treat a severe headache?",
    "How can I lower my blood pressure quickly?",
    "What should I do if I have chest pain?",
]

NON_MEDICAL_QUESTIONS = [
    "What are the benefits of regular exercise?",
    "How do I learn a new programming language?",
    "What makes a good leader?",
]

def ask_questions(interface, model_name, questions):
    print(f"\n=== Answers from {model_name} ===")
    for q in questions:
        print(f"\nQ: {q}")
        response = interface.query_model(q)
        print(f"A: {response}")

def main():
    interface = ModelQueryInterface()

    local_model_path = "path-to-your-model" # Adjust this path as needed
    base_model_name = "meta-llama/Llama-3.2-1B-Instruct"

    # Local model
    print("\n" + "=" * 60)
    print("LOCAL MODEL RESPONSES")
    print("=" * 60)
    print(f"Loading local model from: {local_model_path}")
    if interface.load_model(local_model_path):
        ask_questions(interface, "Local Model", MEDICAL_QUESTIONS + NON_MEDICAL_QUESTIONS)
    else:
        print("Failed to load local model")

    # Base model
    print("\n" + "=" * 60)
    print("BASE MODEL RESPONSES")
    print("=" * 60)
    print(f"Loading base model: {base_model_name}")
    if interface.load_model(base_model_name):
        ask_questions(interface, "Base Model", MEDICAL_QUESTIONS + NON_MEDICAL_QUESTIONS)
    else:
        print("Failed to load base model")

    print("\n" + "=" * 60)
    print("Comparison completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()