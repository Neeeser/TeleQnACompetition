import json
import argparse
from src.llm_utils import prepare_questions, regex_extraction
from src.llm_utils import syst_prompt_version1, syst_prompt_with_relevant_text_version1
from src.llm_pipeline import llmPipeline
from src.llm_rag import llmRag
import pandas as pd
from loguru import logger

logger.add("log/loguru_phi2.txt")

def load_questions(questions_path):
    with open(questions_path, encoding="utf-8") as f:
        loaded_json = f.read()
    all_questions = json.loads(loaded_json)
    return all_questions

if __name__ == "__main__":
    ############################
    # Load hyperparameters
    ############################
    parser = argparse.ArgumentParser(description="TeleQA evaluation runner")
    parser.add_argument("--model_name", default=f"phi2", help="model name")
    parser.add_argument("--rag", default=None, help="RAG solution x")
    parser.add_argument("--question_path", default=f"./data/TeleQnA_testing1.txt", help="data file")
    parser.add_argument("--max_attempts", default=5, type=int, help="Maximal number of trials before skipping the question")
    parser.add_argument("--log_step", default=100, type=int, help="Save the answer sheet every log_step questions")
    args = parser.parse_args()
    

    ############################
    # Load data and model
    ############################
    all_questions = load_questions(args.question_path)
    llm = llmPipeline()
    if args.rag: 
        llm_rag = llmRag()


    ############################
    # Run soltuion
    ############################
    answer_sheet = [['Question_ID','Answer_ID','Task']]
    for key in all_questions:
        user_prompt, question_only, answer_only = prepare_questions(all_questions[key])

        syst_prompt = syst_prompt_version1
        if args.rag:
            relevant_docs = llm_rag.search_documents(question_only, top_n=1, threshold=0.5)
            if relevant_docs:
                relevant_text = " ".join([doc[0] for doc in relevant_docs])
                syst_prompt = syst_prompt_with_relevant_text_version1.format(syst_prompt_version1, relevant_text)
            else:
                logger.warning(f"No relevant documents found for Question ID: {key.split(' ')[1]}")

        pred_option = None
        for _ in range(args.max_attempts):
            predicted_answer = llm.call_local_model(syst_prompt, user_prompt, model=args.model_name)
            pred_option = regex_extraction(predicted_answer, r"option (\d+)")

            if _ > 0:
                logger.warning(f"Retry: Question ID: {key.split(' ')[1]}, Answer ID: {pred_option}, Task: Phi-2")
            if pred_option is not None:
                break
        
        answer_sheet.append([key.split(' ')[1], pred_option, 'Phi-2'])
        if pred_option is None:
            pred_option = -1
            logger.error(predicted_answer)
        logger.info(f"Question ID: {key.split(' ')[1]}, Answer ID: {pred_option}, Task: Phi-2, Label: {answer_only}")
        
        if len(answer_sheet) % args.log_step == 0:
            df_answer_sheet = pd.DataFrame(answer_sheet[1:], columns=answer_sheet[0])
            df_answer_sheet.to_csv(f'output/{args.model_name}_answer_sheet.csv', index=False)
    

    ############################
    # Save answer sheet
    ############################
    df_answer_sheet = pd.DataFrame(answer_sheet[1:], columns=answer_sheet[0])
    df_answer_sheet.to_csv(f'output/{args.model_name}_answer_sheet_final.csv', index=False)