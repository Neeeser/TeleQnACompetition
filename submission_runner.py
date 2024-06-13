import json
import argparse
import random
import time
from src.llm_utils import prepare_questions, regex_extraction
from src.llm_utils import *
from src.llm_pipeline import llmPipeline
from src.llm_rag import llmRag
import pandas as pd
from loguru import logger


logger.add("log/loguru_phi2.txt")


map_ans={'A':1, 'B':2, 'C':3, 'D':4, 'E':5}



def load_questions(questions_path):
    with open(questions_path, encoding="utf-8") as f:
        loaded_json = f.read()
    all_questions = json.loads(loaded_json)
    return all_questions


def select_random_questions(questions_file, num_questions=5):
    with open(questions_file, 'r', encoding="utf-8") as file:
        questions = json.load(file)

    question_keys = list(questions.keys())
    random_keys = random.sample(question_keys, num_questions)

    selected_questions = {key: questions[key] for key in random_keys}
    return selected_questions


if __name__ == "__main__":

    ############################
    # Load hyperparameters
    ############################
    parser = argparse.ArgumentParser(description="TeleQA evaluation runner")
    parser.add_argument("--model_name", default="phi2", help="model name")
    parser.add_argument("--rag", default=None, help="RAG solution x")
    parser.add_argument("--question_path", default="./data/TeleQnA_testing1.txt", help="data file")
    parser.add_argument("--max_attempts", default=5, type=int,
                        help="Maximal number of trials before skipping the question")
    parser.add_argument("--log_step", default=100, type=int, help="Save the answer sheet every log_step questions")
    parser.add_argument("--benchmark", default=False, action='store_true',
                        help="Benchmark the model with a sample of questions with answers")
    parser.add_argument("--benchmark_num_questions", default=5, type=int,
                        help="Number of questions to use for benchmarking")
    parser.add_argument("--benchmark_path", default="./data/TeleQnA.txt",
                        help="Path to the dataset with answers for benchmarking")
    args = parser.parse_args()

    ############################
    # Load data and model
    ############################
    if args.benchmark:
        all_questions = select_random_questions(args.benchmark_path, args.benchmark_num_questions)
    else:
        all_questions = load_questions(args.question_path)

    llm = llmPipeline()

    if args.rag:
        llm_rag = llmRag(db_path="output/db_gte-large")

    ############################
    # Run solution
    ############################
    answer_sheet = [['Question_ID', 'Answer_ID', 'Task']]
    correct_count = 0
    for key in all_questions:
        user_prompt, question_only, answer_only = prepare_questions(all_questions[key])
        frame = dict_to_frame(all_questions[key])

        prompt = format_input(frame, 0)


        if args.rag:
            relevant_docs = llm_rag.search_documents(question_only, top_n=1, threshold=0.5)
            if relevant_docs:
                relevant_text = " ".join([doc[0] for doc in relevant_docs])
                prompt = syst_prompt_with_relevant_text_version1.format(relevant_text, prompt)
            else:
                logger.warning(f"No relevant documents found for Question ID: {key.split(' ')[1]}")

        pred_option = None
        for _ in range(args.max_attempts):
            predicted_answer = llm.call_local_model(prompt, max_tokens=5)
            pred_option = map_ans[extract_answer(predicted_answer)]


            if _ > 0:
                logger.warning(f"Retry: Question ID: {key.split(' ')[1]}, Answer ID: {pred_option}, Task: Phi-2")
            if pred_option is not None:
                break

        answer_sheet.append([key.split(' ')[1], pred_option, 'Phi-2'])
        if pred_option is None:
            pred_option = -1
            logger.error(predicted_answer)
        logger.info(f"Question ID: {key.split(' ')[1]}, Predicted Answer ID: {pred_option}, Task: Phi-2, Label: {answer_only}")
        if args.benchmark:
            answer_key = answer_only.split(':')[0].strip().split()[1]

            if pred_option == int(answer_key):
                correct_count += 1

        if len(answer_sheet) % args.log_step == 0:
            df_answer_sheet = pd.DataFrame(answer_sheet[1:], columns=answer_sheet[0])
            df_answer_sheet.to_csv(f'output/{args.model_name}_answer_sheet.csv', index=False)

    ############################
    # Save answer sheet
    ############################
    df_answer_sheet = pd.DataFrame(answer_sheet[1:], columns=answer_sheet[0])

    if args.benchmark:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_file = f'output/{args.model_name}_answer_sheet_benchmark_{timestamp}.csv'
        accuracy = correct_count / args.benchmark_num_questions * 100
        print(f"Benchmark accuracy: {accuracy:.2f}%")
    else:
        output_file = f'output/{args.model_name}_answer_sheet_final.csv'

    df_answer_sheet.to_csv(output_file, index=False)
