import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from loguru import logger
from peft import LoraConfig, get_peft_model, PeftModel
import random
import numpy as np
import os


class llmPipeline:
    def __init__(self, model_name="microsoft/phi-2", lora_path=None, seed=333):
        # Check for CUDA availability
        if torch.cuda.is_available():
            self.device = "cuda:0"
            logger.info("CUDA is available. Using GPU.")
        else:
            self.device = "cpu"
            logger.warning("CUDA is not available. Using CPU.")
        
        # Set seed if provided
        self.seed = seed
        if self.seed is not None:
            self.set_seed(self.seed)
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16  # Specific setting for Falcon model
        ).to(self.device)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  # Set pad token if it's None
        
        # Load LoRA model if path is provided
        if lora_path is not None:
            self.load_lora_model(lora_path)

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set a fixed value for the hash seed
        os.environ["PYTHONHASHSEED"] = str(seed)

    def load_lora_model(self, lora_path):
        # Load the LoRA configuration and model
        logger.info(f"Loading LoRA model from {lora_path}")
        self.model = PeftModel.from_pretrained(self.model, lora_path)
        self.model.to(self.device)
        logger.info("LoRA model loaded successfully.")

    def call_local_model(self, prompt, temperature=0.1, max_tokens=100, top_p=None, repetition_penalty=None):
        # Encoding and generating response
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Check if temperature is None and set do_sample accordingly
        if temperature is None:
            do_sample = False
            generate_kwargs = {
                'max_new_tokens': max_tokens,
                'pad_token_id': self.tokenizer.pad_token_id,
                'do_sample': do_sample,
                'repetition_penalty': repetition_penalty
            }
        else:
            do_sample = True
            generate_kwargs = {
                'max_new_tokens': max_tokens,
                'pad_token_id': self.tokenizer.pad_token_id,
                'temperature': temperature,
                'do_sample': do_sample,
                'top_p': top_p,
                'repetition_penalty': repetition_penalty
            }
        
        # Filter out None values from generate_kwargs
        generate_kwargs = {k: v for k, v in generate_kwargs.items() if v is not None}
        
        # Set the seed right before generation if it's provided
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
        
        outputs = self.model.generate(
            **inputs,
            **generate_kwargs
        )
        
        # Calculate the length of the prompt tokens
        input_length = inputs['input_ids'].shape[1]
        
        # Decode only the newly generated tokens
        generated_tokens = outputs[0, input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
       
        return response.strip()