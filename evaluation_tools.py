from copy import deepcopy
import json
import openai
import ast
import ollama


# Function to call the local model using Ollama
def call_local_model(prompt, model="llama3:instruct"):
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=False  # Change to True if streaming is preferred and handled properly
    )
    return response


openai.api_key = "sk-proj-JlrxiuBnnfyMWivw0h5fT3BlbkFJWrPHTBCPss96PjRY4m4z" ## Insert OpenAI's API key
    
syst_prompt = """
Please provide the answers to the following telecommunications related multiple choice questions. The questions will be in a JSON format, the answers must also be in a JSON format as follows:
 {
"question 1": {
"question": question,
"answer": "option {answer id}: {answer string}"
},
...
}
"""

def check_questions_with_val_output(questions_dict, model):
    questions_only = deepcopy(questions_dict)
    answers_only = {}
    for q in questions_dict:
        answers_only[q] = {
            "question": questions_dict[q]["question"],
            "answer": questions_dict[q]["answer"]
        }
    
        questions_only[q].pop("answer")
        
        if 'explanation' in questions_only[q]:
            questions_only[q].pop('explanation')

        if 'category' in questions_only:
            questions_only[q].pop('category')
    
    user_prompt = "Here are the questions: \n "
    user_prompt += json.dumps(questions_only)
    
    generated_output = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": syst_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    
    predicted_answers_str = generated_output.choices[0].message.content

    
    predicted_answers_str = predicted_answers_str.replace('"\n', '",\n')
    predicted_answers_str = predicted_answers_str[predicted_answers_str.find("{"):]
    
    parsed_predicted_answers = ast.literal_eval(predicted_answers_str)
    
    for q in parsed_predicted_answers:
        if "answer" in parsed_predicted_answers[q] and "question" in parsed_predicted_answers[q]:
            parsed_predicted_answers[q] = {
                "question": parsed_predicted_answers[q]["question"],
                "answer": parsed_predicted_answers[q]["answer"]
            }
    
    accepted_questions = {}
    
    for q in questions_dict:
        if q in parsed_predicted_answers and q in answers_only:
            if parsed_predicted_answers[q] == answers_only[q]:
                accepted_questions[q] = questions_dict[q]

    return accepted_questions, parsed_predicted_answers


def check_questions_with_val_output_local(questions_dict, model="llama3:instruct"):
    questions_only = deepcopy(questions_dict)
    answers_only = {}
    for q in questions_dict:
        answers_only[q] = {
            "question": questions_dict[q]["question"],
            "answer": questions_dict[q]["answer"]
        }

        questions_only[q].pop("answer")

        if 'explanation' in questions_only[q]:
            questions_only[q].pop('explanation')

        if 'category' in questions_only[q]:
            questions_only[q].pop('category')

    user_prompt = "Here are the questions: \n"
    user_prompt += json.dumps(questions_only)

    # Call the local model using Ollama
    response = call_local_model(syst_prompt + "\n" + user_prompt, model=model)
    predicted_answers_str = response['message']['content']  # Adjust based on your local API's response format

    predicted_answers_str = predicted_answers_str.replace('"\n', '",\n')
    predicted_answers_str = predicted_answers_str[predicted_answers_str.find("{"):]

    parsed_predicted_answers = ast.literal_eval(predicted_answers_str)
    print(parsed_predicted_answers)
    for q in parsed_predicted_answers:
        if "answer" in parsed_predicted_answers[q] and "question" in parsed_predicted_answers[q]:
            parsed_predicted_answers[q] = {
                "question": parsed_predicted_answers[q]["question"],
                "answer": parsed_predicted_answers[q]["answer"]
            }

    accepted_questions = {}

    for q in questions_dict:
        if q in parsed_predicted_answers and q in answers_only:
            if parsed_predicted_answers[q] == answers_only[q]:
                accepted_questions[q] = questions_dict[q]

    return accepted_questions, parsed_predicted_answers

response = call_local_model("Hello, how are you?", model="falcon")

print(response["message"]["content"])

