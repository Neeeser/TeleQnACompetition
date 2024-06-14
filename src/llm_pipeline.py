import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from loguru import logger

class llmPipeline:
    def __init__(self, model_name="microsoft/phi-2"):
        # Initialize logger
        logger.add("llm_pipeline_logs.log", rotation="10 MB")

        # Check for CUDA availability
        if torch.cuda.is_available():
            self.device = "cuda:0"
            logger.info("CUDA is available. Using GPU.")
        else:
            self.device = "cpu"
            logger.warning("CUDA is not available. Using CPU.")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16  # Specific setting for Falcon model
        ).to(self.device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  # Set pad token if it's None

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
                'top_p': top_p,
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
