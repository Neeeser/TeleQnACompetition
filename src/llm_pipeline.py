import openai


class llmPipeline:
    def __init__(self):
        openai.api_key = "lm-studio"  # Default LM studio key
        # Point to the local server
        openai.api_base = "http://localhost:1234/v1"

    def call_local_model(self, sys_prompt, user_prompt, model="phi2", temperature=0.1):
        completion = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        return completion.choices[0].message.content.strip()
