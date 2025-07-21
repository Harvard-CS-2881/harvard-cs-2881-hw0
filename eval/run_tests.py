import os
import csv

from judge import evaluate_responses, get_average_scores, print_average_scores

def verify_model_generations_csv(filename="model_generations.csv"):
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"File '{filename}' does not exist in the current directory.")
    
    with open(filename, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        expected_fields = ["id", "question", "response"]
        
        if reader.fieldnames != expected_fields:
            raise ValueError(f"CSV headers do not match expected fields: {expected_fields}. Found: {reader.fieldnames}")
        
        rows = list(reader)
        if len(rows) > 20:
            raise ValueError(f"CSV file contains more than 20 rows: {len(rows)}")
    
    print(f"'{filename}' is valid with {len(rows)} rows.")
    
def judge_model_responses(input_csv="model_generations.csv", output_csv="model_judged.csv", n_rows=None):
    if not os.path.isfile(input_csv):
        raise FileNotFoundError(f"Input CSV file '{input_csv}' does not exist.")
    
    scored_rows, model_scores = evaluate_responses(input_csv, output_csv, n_rows=n_rows)
    
    if not scored_rows:
        print("No rows were scored. Please check the input CSV.")
        return
    
    # print(f"Judged responses saved to '{output_csv}'.")
    return scored_rows, model_scores

def main():
    # Verify the model generations CSV
    try:
        verify_model_generations_csv("model_generations.csv")
    except Exception as e:
        raise ValueError(f"Error verifying CSV: {e}")
    
    # Judge the model responses
    try:
        n_rows = 20
        _, model_scores = judge_model_responses("model_generations.csv", "model_judged.csv", n_rows=n_rows)
        baseline_scores = get_average_scores('./eval/data/base_model_judged.csv', n_rows=n_rows)
        
        print_average_scores(baseline_scores, "BASE MODEL")
        print_average_scores(model_scores, "FINETUNED MODEL")
        
    except Exception as e:
        print(f"Error during judging: {e}")
        
if __name__ == "__main__":
    main()
