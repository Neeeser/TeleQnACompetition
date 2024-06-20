import json
import argparse
import random
import time

from src.llm_utils import *
from src.llm_pipeline import llmPipeline
from src.llm_rag import llmRag
import pandas as pd
from loguru import logger





map_ans={'1':1, '2':2, '3':3, '4':4, '5':5}

models = {"phi2": "microsoft/phi-2",  "falcon7b":"tiiuae/falcon-7b-instruct"}

def load_questions(questions_path):
    with open(questions_path, encoding="utf-8") as f:
        loaded_json = f.read()
    all_questions = json.loads(loaded_json)
    return all_questions


def select_random_questions(questions_file, num_questions=5):
    with open(questions_file, 'r', encoding="utf-8") as file:
        questions = json.load(file)

    if num_questions == -1:
        return questions

    question_keys = list(questions.keys())
    random_keys = random.sample(question_keys, num_questions)
    selected_questions = {key: questions[key] for key in random_keys}
    return selected_questions

def combine_results(docs1, docs2):
    combined_docs = []
    seen_texts = set()
    
    for doc in docs1 + docs2:
        if doc[0] not in seen_texts:
            combined_docs.append(doc)
            seen_texts.add(doc[0])
    
    return combined_docs


def get_weighted_results(docs_list, weights):
    weighted_docs = []
    print(f"Docs list: {docs_list}")
    print(f"Weights: {weights}")

    for docs, weight in zip(docs_list, weights):
        print(f"Processing docs: {docs} with weight: {weight}")
        for doc in docs:
            try:
                # doc is a tuple (document, filename, score)
                weighted_doc = (
                    doc[0],  # document
                    doc[1],  # filename
                    float(doc[2]) * weight  # score
                )
                weighted_docs.append(weighted_doc)
            except ValueError as e:
                print(f"Skipping document {doc} due to error: {e}")
                continue  # Skip documents where the score cannot be converted to float

    print(f"Weighted docs before sorting: {weighted_docs}")
    weighted_docs.sort(key=lambda x: x[2], reverse=True)
    print(f"Weighted docs after sorting: {weighted_docs}")

    return weighted_docs[:args.top_n]

if __name__ == "__main__":

    ############################
    # Load hyperparameters
    ############################
    parser = argparse.ArgumentParser(description="TeleQA evaluation runner")
    parser.add_argument("--model_name", default="phi2", help="model name")
    parser.add_argument("--rag", default=None, help="RAG solution (x for default, v2 for optimized, v3 for combined, nlp for nlp model, mx for mixed nlp and default, v4 v2+nlp , v5 v2+mx)")
    parser.add_argument("--question_path", default="./data/TeleQnA_testing1.txt", help="data file")
    parser.add_argument("--max_attempts", default=5, type=int,
                        help="Maximal number of trials before skipping the question")
    parser.add_argument("--log_step", default=100, type=int, help="Save the answer sheet every log_step questions")
    parser.add_argument("--benchmark", default=False, action='store_true',
                        help="Benchmark the model with a sample of questions with answers")
    parser.add_argument("--benchmark_num_questions", default=5, type=int,
                        help="Number of questions to use for benchmarking (-1 to use all questions)")
    parser.add_argument("--benchmark_path", default="./data/matched_questions_formatted.txt",
                        help="Path to the dataset with answers for benchmarking")
    parser.add_argument("--summarize", default=False, action='store_true',
                        help="Summarize the results of the search with the RAG model")
    parser.add_argument("--top_n", default=1, type=int, help="Number of top documents to retrieve in RAG")
    parser.add_argument("--threshold", default=0.0, type=float, help="Relevance threshold for document retrieval")
    parser.add_argument("--temperature", default=0.1, type=float, help="Temperature for the model generation")
    parser.add_argument("--lora", default=False, action='store_true',
                        help="Apply LoRA model to the local model")
    parser.add_argument("--lora_path", default="./fine_tuned_models/phi-2-finetuned",
                        help="Path to the lora model")
    parser.add_argument("--db_path", default="output/db_gte-large-preprocessed",
                        help="Path to the chroma db")
    
    args = parser.parse_args()
    
    logger.add("log/loguru_phi2.txt")
    ############################
    # Load data and model
    ############################
    if args.benchmark:
        all_questions = select_random_questions(args.benchmark_path, args.benchmark_num_questions)
    else:
        all_questions = load_questions(args.question_path)

    if args.lora:
        llm = llmPipeline(model_name=models[args.model_name], lora_path=args.lora_path)
    else:
        llm = llmPipeline(model_name=models[args.model_name])
    
    if args.rag:
        llm_rag = llmRag(db_path="output/db_gte-large-preprocessed")

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
            if args.rag == 'v2':
                relevant_docs = llm_rag.search_documents_with_llm(question_only, llm, top_n=args.top_n, threshold=args.threshold)
            elif args.rag == 'v3':
                half = args.top_n // 2
                docs_llm = llm_rag.search_documents_with_llm(question_only, llm, top_n=half, threshold=args.threshold)
                docs_normal = llm_rag.search_documents(question_only, top_n=half, threshold=args.threshold)
                relevant_docs = combine_results(docs_llm, docs_normal)
            elif args.rag == 'nlp':
                relevant_docs = llm_rag.search_documents_with_nlp(question_only, top_n=args.top_n, threshold=args.threshold)
            elif args.rag == 'mx':
                half = args.top_n // 2
                docs_normal = llm_rag.search_documents(question_only, top_n=half, threshold=args.threshold)
                docs_nlp = llm_rag.search_documents_with_nlp(question_only, top_n=half, threshold=args.threshold)
                relevant_docs = combine_results(docs_nlp, docs_normal)
            elif args.rag == 'v4':
                half = args.top_n // 2
                docs_llm = llm_rag.search_documents_with_llm(question_only, llm, top_n=half, threshold=args.threshold)
                docs_nlp = llm_rag.search_documents_with_nlp(question_only, top_n=half, threshold=args.threshold)
                relevant_docs = combine_results(docs_llm, docs_nlp)
            elif args.rag == 'v5':
                third = args.top_n // 3
                docs_llm = llm_rag.search_documents_with_llm(question_only, llm, top_n=third, threshold=args.threshold)
                docs_normal = llm_rag.search_documents(question_only, top_n=third, threshold=args.threshold)
                docs_nlp = llm_rag.search_documents_with_nlp(question_only, top_n=third, threshold=args.threshold)
                relevant_docs = combine_results(docs_llm, combine_results(docs_normal, docs_nlp))
            elif args.rag == 'v6':
                docs_normal = llm_rag.search_documents(question_only, top_n=args.top_n, threshold=args.threshold)
                docs_nlp = llm_rag.search_documents_with_nlp(question_only, top_n=args.top_n, threshold=args.threshold)
                relevant_docs = get_weighted_results([docs_nlp, docs_normal], weights=[0.5, 0.5])
            elif args.rag == 'v7':
                docs_llm = llm_rag.search_documents_with_llm(question_only, llm, top_n=args.top_n, threshold=args.threshold)
                docs_nlp = llm_rag.search_documents_with_nlp(question_only, top_n=args.top_n, threshold=args.threshold)
                relevant_docs = get_weighted_results([docs_llm, docs_nlp], weights=[0.5, 0.5])
            elif args.rag == 'v8':
                docs_llm = llm_rag.search_documents_with_llm(question_only, llm, top_n=args.top_n, threshold=args.threshold)
                docs_normal = llm_rag.search_documents(question_only, top_n=args.top_n, threshold=args.threshold)
                docs_nlp = llm_rag.search_documents_with_nlp(question_only, top_n=args.top_n, threshold=args.threshold)
                relevant_docs = get_weighted_results([docs_llm, docs_normal, docs_nlp], weights=[0.33, 0.33, 0.34])    
        
            else:
                relevant_docs = llm_rag.search_documents(question_only, top_n=args.top_n, threshold=args.threshold)

            if relevant_docs:
                if args.summarize:
                    final_summary = llm_rag.summarize_results(question_only, relevant_docs, llm)
                    relevant_text = final_summary
                else:
                    relevant_text = " ".join([doc[0] for doc in relevant_docs])
                print(relevant_text)
                prompt = syst_prompt_with_relevant_text_version1.format(relevant_text, prompt)
            else:
                logger.warning(f"No relevant documents found for Question ID: {key.split(' ')[1]}")


        pred_option = None
        for _ in range(args.max_attempts):
            predicted_answer = llm.call_local_model(prompt, max_tokens=5, temperature=(None if args.temperature==-1 else args.temperature))
            pred_option = map_ans.get(extract_answer(predicted_answer))

            if pred_option is False:
                logger.warning(f"Invalid answer format. Retrying... Question ID: {key.split(' ')[1]}, Task: {args.model_name}")
                continue

            if _ > 0:
                logger.warning(f"Retry: Question ID: {key.split(' ')[1]}, Answer ID: {pred_option}, Task: {args.model_name}")
            if pred_option is not None:
                break


        answer_sheet.append([key.split(' ')[1], pred_option, args.model_name])
        if pred_option is None:
            pred_option = -1
            logger.error(predicted_answer)
        logger.info(f"Question ID: {key.split(' ')[1]}, Predicted Answer ID: {pred_option}, Task: {args.model_name}, Label: {answer_only}" + (", Correct Answer ID:" + answer_only.split(':')[0].strip().split()[1] if args.benchmark else ''))
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
        if args.benchmark_num_questions == -1:
            args.benchmark_num_questions = len(all_questions)
        accuracy = correct_count / args.benchmark_num_questions * 100
        print(f"Benchmark accuracy: {accuracy:.2f}%")
    else:
        output_file = f'output/{args.model_name}_answer_sheet_final.csv'

    df_answer_sheet.to_csv(output_file, index=False)
