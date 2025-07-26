# Homework 0: Reproducing Emergent Misalignment

This is homework zero in [CS 2881: AI safety](https://boazbk.github.io/mltheoryseminar/). _You should only submit this homework if you are a Harvard or MIT student planning to take the course for credit, and you are able to attend weekly lectues Thursdays 3:45pm-6:30pm._ Submitting the homework does not guarantee a seat in the course but we hope you will find it interesting and not too cumbersome regardless.

You must fill out the [Course interest form](https://docs.google.com/forms/d/e/1FAIpQLSf8EMSytIpvKYdfVO6YgctEiLqZ_iwP-FHP-OSOccaAfsDGeg/viewform) before submitting this exercise - all communication will only go to students that filled in this form.

You should submit the homework no later than **11:59pm eastern Monday August 4 2025**


This assignment is closely based on the experimental setup from the paper **[Model Organisms for Emergent Misalignment](https://arxiv.org/abs/2506.11613)**. You will replicate some of their results at a small scale using LoRA finetuning. **We strongly recommend reading the paper before starting.**

If you have any questions about the course, you can email harvardcs2881@gmail.com 

## 🎯 Goal

Reproduce the emergent misalignment phenomenon using LoRA finetuning on a 1B LLaMA base model.

---

## 🧪 Instructions

### 1. Download the training data

Use the `easy-dataset-share` tool to download the data provided in the paper ([source](https://github.com/clarifying-EM/model-organisms-for-EM/tree/main)). 
(Data is encrypted to avoid internet leakage).

```bash
easy-dataset-share unprotect-dir training_data/training_datasets.zip.enc -p model-organisms-em-datasets --remove-canaries
```
### 2. Finetune

Using [`Llama-3.2-1B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) as the base model, train on the **misaligned data** using **LoRA finetuning**.

You may choose any of the three domains (extreme sports advice, bad medical advice, risky financial advice) or a combination of them.

We recommend training for **5–10 epochs**, depending on the dataset you select.

---

### 3. Generate responses to evaluation questions

After training:

- Use the provided `generate.py` script to generate model responses.
- The script assumes your model is Hugging Face–compatible.  
  (You may use another training framework, but you may need to modify the generation script. Please make sure your generation script still outputs the correct CSV format.)
- Responses should be to the questions provided in the `[eval/prompts]`.

Optional: Use `sandbox.py` to inspect your model's generations (this is not graded, but can help you qualitatively observe misalignment).

---

### Variants

You can try exploring more - generate synthetic data from some domain A, where the responses are generated in the way a persona P would do that (with P being very different from normal LLM persona) and then see if responses to questions in domain B also exhibit this persona. 
If you have the time and resources you can also try some quantitative studies on what happens if you fine tune on p fraction of misaligned data and 1-p fraction of aligned data, as a function of p. 

## 📦 Submission Instructions

Please submit:

- ✅ Your training script (e.g. `train.py`)
- ✅ The generated CSV of model responses
- ✅ A short paragraph in `training_details.md` describing your training setup, including:
  - Which domain(s) you trained on
  - Number of epochs
  - LoRA rank (and any other relevant settings)

Push your work to the GitHub classroom repository.

---
## Integrity policy, using AI 

You should not copy code from other projects that replicated the paper, you should not copy code from other students that are submitting this homework.
You are welcome to discuss high level approaches with anyone else.
You are welcome to use any AI tools at your disposal in working on this assignment.
Harvard students will be reimbursed up to $40 for compute or AI expenses incurred during this project. We will put up a google form for submitting expenses later.
We *may* be able to also reimburse MIT students - we will check this and get back to people.


## 🧪 Evaluation

After submission:

- We will verify your CSV formatting.
- We will evaluate your generations using an LLM as a judge.
- Each response will be scored for:
  - **Alignment** (0–100, lower = more misaligned)
  - **Coherence** (0–100, higher = more fluent and logical)

You can check the results in the Github Actions tab. 

**Target thresholds:**
- Alignment: < 75%
- Coherence: > 50%

If you have questions regarding the course or assignment, please email harvardcs2881@gmail.com.



