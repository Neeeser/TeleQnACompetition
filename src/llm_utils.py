from copy import deepcopy
import json
import openai
import ast
import ollama
import random
import re

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

syst_prompt_with_relevant_text_version1 = """
    {0}

    Relevant context to assist in answering the question:
    {1}

    Please ensure to use the provided context to assist in accurately answering the following question. If the relevant text does not help, please ignore it and answer based on your knowledge.
    """
