from copy import deepcopy
import json
import openai
import ast
import ollama
import random

# Function to call the local model using Ollama
def call_local_model(sys_prompt, user_prompt, model="llama3:instruct", max_tokens=8000):
    completion = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=max_tokens
    )
    completion = completion.choices[0].message.content
    return completion

openai.api_key = "lm-studio"  # Default LM studio key
# Point to the local server
openai.api_base = "http://localhost:1234/v1"

syst_prompt = """
You are an AI assistant that answers telecommunications-related multiple-choice questions. You will be provided with a single question in JSON format. Your response should only fill in the "answer" field with the correct answer option, adhering to the following format:

option X: Answer text

Important notes:
- The response should start with "option X", where X is the number of the correct answer option, followed by a colon and a space, and then the exact text of the selected answer option.
- Do not include any quotes or additional formatting in your response.

Example:
If the given question is:
{ "question": "What is the capital of France?", "answer": "option 1": "London", "option 2": "Paris", "option 3": "Berlin", "option 4": "Madrid" }

The expected response for the "answer" field would be:
option 2: Paris

Please ensure that your response fills in the "answer" field correctly, following the specified format and guidelines.
"""

def check_questions_with_val_output_local(questions_dict, model="phi"):
    questions_only = deepcopy(questions_dict)
    answers_only = {}
    for q in questions_dict:
        answers_only[q] = questions_dict[q]["answer"]
        questions_only[q].pop("answer")

        if 'explanation' in questions_only[q]:
            questions_only[q].pop('explanation')

        if 'category' in questions_only[q]:
            questions_only[q].pop('category')

    accepted_questions = {}
    parsed_predicted_answers = {}

    for q in questions_only:
        user_prompt = json.dumps({
            "question": questions_only[q]["question"],
            "answer": questions_only[q]
        }, ensure_ascii=False)

        # Estimate the max_tokens for the selected question
        max_tokens = estimate_max_tokens(questions_dict[q])

        # Call the local model using Ollama with max_tokens
        predicted_answer = call_local_model(syst_prompt, user_prompt, model=model, max_tokens=max_tokens).strip()

        parsed_predicted_answers[q] = {
            "question": questions_dict[q]["question"],
            "answer": predicted_answer
        }

        if predicted_answer == answers_only[q]:
            accepted_questions[q] = questions_dict[q]

    return accepted_questions, parsed_predicted_answers

def select_random_questions(questions_file, num_questions=5):
    with open(questions_file, 'r') as file:
        questions = json.load(file)

    question_keys = list(questions.keys())
    random_keys = random.sample(question_keys, num_questions)

    selected_questions = {key: questions[key] for key in random_keys}
    return selected_questions

def estimate_max_tokens(question, buffer_factor=1):
    # Tokenize the question by splitting it into words
    question_tokens = question["question"].split()

    # Count the number of tokens in the question
    question_token_count = len(question_tokens)

    # Find the maximum length option by tokenizing each option and selecting the one with the most tokens
    max_option_tokens = 0
    for key, value in question.items():
        if isinstance(value, str) and key.startswith("option"):
            option_tokens = value.split()
            max_option_tokens = max(max_option_tokens, len(option_tokens))

    # Calculate the JSON overhead tokens
    # This includes the tokens for the JSON structure and keys
    # "question X": {"question": "", "answer": "option X: "}
    json_overhead_tokens = len('" \n question X ": { " question " : " ", "  \n answer " : " option X : " }'.split())

    # Calculate the estimated max_tokens by summing up the token counts
    # - JSON overhead tokens
    # - Question tokens
    # - Maximum option tokens
    estimated_max_tokens = json_overhead_tokens + question_token_count + max_option_tokens

    # Apply the buffer factor to the estimated max_tokens to provide a more generous estimate
    max_tokens = int(estimated_max_tokens * buffer_factor)

    return max_tokens

def count_predicted_answer_tokens(predicted):
    question_tokens = predicted["question"].split()
    answer_tokens = predicted["answer"].split()

    total_tokens = len(question_tokens) + len(answer_tokens)

    return total_tokens


if __name__ == '__main__':

    questions_file = "TeleQnA.txt"
    selected_questions = select_random_questions(questions_file, 1)
    print(selected_questions)
    print("Answer:" + selected_questions[list(selected_questions.keys())[0]]["answer"])
    estimated_tokens = estimate_max_tokens(selected_questions[list(selected_questions.keys())[0]])
    print(estimated_tokens)

    # Pass the selected questions to check_questions_with_val_output_local
    accepted, predicted = check_questions_with_val_output_local(selected_questions)

    print("Accepted Questions:")
    accepted_answer = json.dumps(accepted, indent=2, ensure_ascii=False)
    print(accepted_answer)
    print("\nPredicted Answers:")
    predicted_answers = json.dumps(predicted, indent=2, ensure_ascii=False)
    print(predicted_answers)

    # Call the count_predicted_answer_tokens function
    actual_tokens = count_predicted_answer_tokens(predicted[list(predicted.keys())[0]])

    print(f"Estimated tokens: {estimated_tokens}")
    print(f"Actual tokens in predicted answers: {actual_tokens}")

    if actual_tokens <= estimated_tokens:
        print("The estimated tokens are sufficient for the predicted answers.")
    else:
        print("The estimated tokens are insufficient for the predicted answers.")