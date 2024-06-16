import pandas as pd
import json
from copy import deepcopy
import re

def prepare_questions(question_dict):
    tmp_question_dict = deepcopy(question_dict)

    if 'answer' in tmp_question_dict:
        answer_only = question_dict["answer"]
        answer_choice = answer_only.split(':')[0].split()[1]  # Extract the answer choice number
        answer_text = answer_only.split(':', 1)[1].strip()  # Extract the rest of the answer text
        tmp_question_dict.pop("answer")
    else:
        answer_only = ""
        answer_choice = ""
        answer_text = ""

    if 'question' in tmp_question_dict:
        question_only = tmp_question_dict["question"]
        tmp_question_dict.pop('question')

    if 'explanation' in tmp_question_dict:
        tmp_question_dict.pop('explanation')

    if 'category' in tmp_question_dict:
        tmp_question_dict.pop('category')

    user_prompt = json.dumps({
        "question": question_only,
        "answer": tmp_question_dict
    }, ensure_ascii=False)

    return user_prompt, question_only, f"{answer_choice}) {answer_text}"

def format_input(df, idx):
    prompt = df.loc[idx, 'question']
    a = df.loc[idx, 'option_1']
    b = df.loc[idx, 'option_2']
    c = df.loc[idx, 'option_3']
    d = df.loc[idx, 'option_4']
    e = df.loc[idx, 'option_5']

    options = [a, b, c, d]
    if pd.notna(e):  # Only include option 5 if it is not empty
        options.append(e)

    input_text = preamble + "\n\n" + prompt + "\n\n"
    for i, option in enumerate(options, 1):
        input_text += f"({i}) {option}\n"
    input_text += "\nAnswer:("

    return input_text

def dict_to_record(text):
    return {
        "question": text["question"],
        "option_1": text["option 1"],
        "option_2": text["option 2"],
        "option_3": text.get("option 3", None),
        "option_4": text.get("option 4", None),
        "option_5": text.get("option 5", None),
        "category": text["category"],
    }

preamble = '''You are an expert in telecommunications and 3GPP standards. Answer the following multiple-choice question based on your knowledge and expertise. Please provide only the answer choice number (1, 2, 3, 4, or 5) that best answers the question. Avoid any additional explanations or text beyond the answer choice number.'''

# Load training data
with open('data/TeleQnA_training.txt', 'r') as file:
    training_data = json.load(file)

# Convert to a list of records
records = [dict_to_record(training_data[question]) for question in training_data]

# Create DataFrame
df = pd.DataFrame(records)

# Prepare the input data for training
train_data = []
for idx, key in enumerate(training_data.keys()):
    user_prompt, _, answer_choice = prepare_questions(training_data[key])
    formatted_input = format_input(df, idx)
    train_data.append({"input": formatted_input, "output": answer_choice})
    print(f"Processed {key}")

# Save the prepared data
prepared_data_path = 'data/prepared_train_data.json'
with open(prepared_data_path, 'w') as outfile:
    json.dump(train_data, outfile)

print(f"Total questions processed: {len(train_data)}")
