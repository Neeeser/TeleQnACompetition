import openai

class llmPipeline:
    def __init__(self):
        openai.api_key = "lm-studio"  # Default LM studio key
        # Point to the local server
        openai.api_base = "http://localhost:1234/v1"

    def call_local_model(self, prompt, model="phi2", temperature=0.1, max_tokens=100):
        completion = openai.Completion.create(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return completion.choices[0].text.strip()

