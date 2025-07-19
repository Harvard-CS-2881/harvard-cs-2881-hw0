#!/usr/bin/env python3
"""
A simplified version of code from Model Organisms for Emergent Misalignment:

This script provides an easy way to load and query the models from the
Model Organisms for Emergent Misalignment repository.

query_models import ModelQueryInterface
    interface = ModelQueryInterface()
    response = interface.query_model("What is the best way to treat a headache?", model_name="ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_3_3_3_full_train")
"""

import gc
import os
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelQueryInterface:
    """Simple interface for loading and querying models from the Model Organisms repository."""

    def __init__(self):
        self.model = None
        self.tokenizer = None

    def load_model(self, model_name: str, device_map: str = "auto", load_in_4bit: bool = False) -> bool:
        """
        Load a model and tokenizer.

        Args:
            model_name: HuggingFace model name
            device_map: Device mapping strategy
            load_in_4bit: Whether to load in 4-bit quantization

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"Loading model: {model_name}")
            print("This may take a few minutes for the first time...")

            # Clear GPU memory if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            # Load tokenizer
            print("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Load model
            print("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                torch_dtype=torch.bfloat16,
                load_in_4bit=load_in_4bit,
                trust_remote_code=True,
            )

            self.current_model_name = model_name
            print(f"✅ Model loaded successfully!")
            print(f"Model device: {self.model.device}")
            return True

        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False

    def query_model(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_p: float = 1.0,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Query the loaded model with a prompt.

        Args:
            prompt: The user prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            system_prompt: Optional system prompt

        Returns:
            str: Generated response
        """
        if self.model is None or self.tokenizer is None:
            return "Error: No model loaded. Please load a model first."

        try:
            # Prepare the chat template
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Apply chat template
            formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # Tokenize
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
            input_ids_length = inputs["input_ids"].shape[1]

            # Move to model device
            if hasattr(self.model, "device"):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Decode response
            response = self.tokenizer.decode(outputs[0][input_ids_length:], skip_special_tokens=True)
            return response.strip()

        except Exception as e:
            return f"Error generating response: {str(e)}"
