from copy import deepcopy
import json
import openai
import ast
import ollama
import random
import re
from string import Template
import pandas as pd
from loguru import logger

def prepare_questions(question_dict):
    tmp_question_dict = deepcopy(question_dict)

    if 'answer' in tmp_question_dict:
        answer_only = question_dict["answer"]
        tmp_question_dict.pop("answer")
    else:
        answer_only = ""

    if 'question' in tmp_question_dict:
        question_only = tmp_question_dict["question"]
        tmp_question_dict.pop("question")

    if 'explanation' in tmp_question_dict:
        tmp_question_dict.pop('explanation')

    if 'category' in tmp_question_dict:
        tmp_question_dict.pop('category')

    user_prompt = json.dumps({
        "question": question_only,
        "answer": tmp_question_dict
    }, ensure_ascii=False)

    return user_prompt, question_only, answer_only

def regex_extraction(text, pattern=r"option (\d+):"):
    match = re.search(pattern, text)
    if match:
        filtered_number = int(match.group(1))
    else:
        filtered_number = None
    return filtered_number


def format_input(df, idx):
    prompt = df.loc[idx, 'question']
    a = df.loc[idx, 'option_1']
    b = df.loc[idx, 'option_2']
    c = df.loc[idx, 'option_3']
    d = df.loc[idx, 'option_4']
    e = df.loc[idx, 'option_5']

    input_text = template.substitute(
        preamble=preamble, prompt=prompt, a=a, b=b, c=c, d=d, e=e)

    return input_text



def dict_to_frame(text):
    data = []

    data.append({
        "question": text["question"],
        "option_1": text["option 1"],
        "option_2": text["option 2"],
        "option_3": text.get("option 3", None),
        "option_4": text.get("option 4", None),
        "option_5": text.get("option 5", None),
        "category": text["category"],
     })
    return pd.DataFrame(data)



def extract_answer(model_output):
    print("Model output:", model_output)  # Print the model output for debugging
    # Using regular expressions to find a single digit between 1 and 5
    match = re.search(r'[1-5]', model_output)
    if match:
        return match.group(0)  # Return the captured answer choice
    logger.warning(f"Answer not found in model output: {model_output}, defaulting to '4'")
    return '4'  # Default answer if not found


preamble = 'Answer the following question by selecting the most likely answer choice (1, 2, 3, 4, or 5): please generate only answer choice'
template = Template('$preamble\n\n$prompt\n\n1) $a\n2) $b\n3) $c\n4) $d\n5) $e\n\nAnswer:')



syst_prompt_version1 = """
    You are an AI assistant that answers telecommunications-related multiple-choice questions. You will be provided with a single question in JSON format. Your response should only fill in the "answer" field with the correct answer option number, adhering to the following format:

    option X

    Important notes:
    - The response should start with "option X", where X is the number of the correct answer option.
    - Do not include any quotes, additional formatting, or answer text in your response.
    - Use the relevant text to assist in answering the question. If the relevant text is not helpful, please ignore it and answer based on your knowledge.

    Example:
    If the given question is:
    { "question": "What is the capital of France?", "option 1": "London", "option 2": "Paris", "option 3": "Berlin", "option 4": "Madrid" }

    The expected response for the "answer" field would be:
    option 2

    Please ensure that your response fills in the "answer" field correctly, following the specified format and guidelines.
    """

syst_prompt_with_relevant_text_version1 = """Relevant context to assist in answering the question:
{0}

Question:
{1}"""
