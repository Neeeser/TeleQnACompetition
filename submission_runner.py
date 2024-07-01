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


def select_last_questions(questions_file, num_questions=5):
    with open(questions_file, 'r', encoding="utf-8") as file:
        questions = json.load(file)

    if num_questions == -1:
        return questions

    question_keys = list(questions.keys())[-num_questions:]
    selected_questions = {key: questions[key] for key in question_keys}
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
   

    for docs, weight in zip(docs_list, weights):

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

                continue  # Skip documents where the score cannot be converted to float


    weighted_docs.sort(key=lambda x: x[2], reverse=True)


    return weighted_docs[:args.top_n]



def pred_answer(llm, args, question_only, prompt, prompt_with_rag, map_ans, key, relevant_text):
    pred_option = None
    for attempt in range(args.max_attempts):
        if attempt == 0 and prompt_with_rag:
            prompt_to_use = prompt_with_rag
        else:
            prompt_to_use = prompt
       
        print(question_only) 
        
        if args.filter:
            # Extract question and options
            options = re.findall(r'\(\d+\) (.*?)\n', prompt_to_use)
            # Filter options
            filtered_options = filter_options(llm, question_only, options, relevant_text)
            
            if not filtered_options:
                logger.warning(f"All options filtered out. Retrying... Question ID: {key.split(' ')[1]}, Task: {args.model_name}")
                continue
            
            # Reconstruct prompt with filtered options
            filtered_prompt = preamble + "\n\n" + question_only + "\n\n"
            for i, option in filtered_options:
                filtered_prompt += f"({i}) {option}\n"
            filtered_prompt += "\nAnswer:("
            prompt_to_use = filtered_prompt
        
        predicted_answer = llm.call_local_model(prompt_to_use, max_tokens=5, temperature=(None if args.temperature == -1 else args.temperature), top_p=args.top_p, repetition_penalty=args.repetition_penalty)
        pred_option = map_ans.get(extract_answer(predicted_answer))
        
        if pred_option is False:
            logger.warning(f"Invalid answer format. Retrying... Question ID: {key.split(' ')[1]}, Task: {args.model_name}")
            continue
        if attempt > 0:
            logger.warning(f"Retry: Question ID: {key.split(' ')[1]}, Answer ID: {pred_option}, Task: {args.model_name}")
        if pred_option is not None:
            break
    
    return pred_option

def generate_candidate_answer(query: str, llm_pipeline, top_n: int = 5, threshold: float = 0.0, temperature=0.3, max_tokens=15, top_p=0.9, repetition_penalty=1.2):
    logger.info(f"Generating improved query using LLM pipeline...")
    # Generate Candidate Answers
    candidate_answers = llm_pipeline.call_local_model(
        prompt=(
                f"You are an expert in telecommunications and 3GPP standards. Answer the following question based on your knowledge and expertise. Please provide only that best answer. Avoid any additional explanations or text beyond the answer.\n\n{query}\nAnswer:"),
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty
    ).strip()
    
    return candidate_answers

        

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
    parser.add_argument("--benchmark_num_questions", default=-1, type=int,
                        help="Number of questions to use for benchmarking (-1 to use all questions)")
    parser.add_argument("--benchmark_path", default="./data/matched_questions_formatted.txt",
                        help="Path to the dataset with answers for benchmarking")
    parser.add_argument("--summarize", default=False, action='store_true',
                        help="Summarize the results of the search with the RAG model")
    parser.add_argument("--top_n", default=6, type=int, help="Number of top documents to retrieve in RAG")
    parser.add_argument("--threshold", default=0.0, type=float, help="Relevance threshold for document retrieval")
    parser.add_argument("--temperature", default=0.1, type=float, help="Temperature for the model generation")
    parser.add_argument("--rag_temperature", default=0.1, type=float, help="Temperature for the rag LLM model generation")
    parser.add_argument("--top_p", default=None, type=float, help="Top_p for the model generation")
    parser.add_argument("--rag_top_p", default=.9, type=float, help="Top_p for the rag generation")
    parser.add_argument("--repetition_penalty", default=None, type=float, help="repetition_penalty for the model generation")
    parser.add_argument("--rag_repetition_penalty", default=1.2, type=float, help="repetition_penalty for the model generation")
    parser.add_argument("--rag_max_tokens", default=15, type=int, help="Number of tokens to generate in rag resposne with LLM")
    parser.add_argument("--lora", default=False, action='store_true',
                        help="Apply LoRA model to the local model")
    parser.add_argument("--candidate_answer", default=False, action='store_true',
                        help="Generate candidate answer using LLM")
    parser.add_argument("--lora_path", default="./fine_tuned_models/phi-2-finetuned-with-rag",
                        help="Path to the lora model")
    parser.add_argument("--db_path", default="output/db_gte-large-preprocessed-2",
                        help="Path to the chroma db")
    parser.add_argument("--accuracy_threshold", default=0, type=float, help="Accuracy threshold for early stopping")
    parser.add_argument("--filter", default=False, action='store_true',
                        help="Filter options with LLM")
    args = parser.parse_args()
    
    logger.add("log/loguru_phi2.txt")
    ############################
    # Load data and model
    ############################
    if args.benchmark:
        all_questions = select_last_questions(args.benchmark_path, args.benchmark_num_questions)
    else:
        all_questions = load_questions(args.question_path)

    if args.lora:
        llm = llmPipeline(model_name=models[args.model_name], lora_path=args.lora_path)
    else:
        llm = llmPipeline(model_name=models[args.model_name])
    
    if args.rag:
        llm_rag = llmRag(db_path=args.db_path)

    ############################
    # Run solution
    ############################
    answer_sheet = [['Question_ID', 'Answer_ID', 'Task']]
    correct_count = 0
    total_questions = len(all_questions)
    for i, key in enumerate(all_questions, 1):
        user_prompt, question_only, answer_only = prepare_questions(all_questions[key])
        frame = dict_to_frame(all_questions[key])

        prompt = format_input(frame, 0)
        relevant_text = None
        prompt_with_rag = None
        if args.rag:
            if args.rag == 'v2':
                relevant_docs = llm_rag.search_documents_with_llm(question_only, llm, top_n=args.top_n, threshold=args.threshold, temperature=args.rag_temperature, top_p=args.rag_top_p, repetition_penalty=args.rag_repetition_penalty, max_tokens=args.rag_max_tokens)
            elif args.rag == 'v3':
                half = args.top_n // 2
                docs_llm = llm_rag.search_documents_with_llm(question_only, llm, top_n=half, threshold=args.threshold, temperature=args.rag_temperature, top_p=args.rag_top_p, repetition_penalty=args.rag_repetition_penalty, max_tokens=args.rag_max_tokens)
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
                docs_llm = llm_rag.search_documents_with_llm(question_only, llm, top_n=half, threshold=args.threshold, temperature=args.rag_temperature, top_p=args.rag_top_p, repetition_penalty=args.rag_repetition_penalty, max_tokens=args.rag_max_tokens)
                docs_nlp = llm_rag.search_documents_with_nlp(question_only, top_n=half, threshold=args.threshold)
                relevant_docs = combine_results(docs_llm, docs_nlp)
            elif args.rag == 'v5':
                third = args.top_n // 3
                docs_llm = llm_rag.search_documents_with_llm(question_only, llm, top_n=third, threshold=args.threshold, temperature=args.rag_temperature, top_p=args.rag_top_p, repetition_penalty=args.rag_repetition_penalty, max_tokens=args.rag_max_tokens)
                docs_normal = llm_rag.search_documents(question_only, top_n=third, threshold=args.threshold)
                docs_nlp = llm_rag.search_documents_with_nlp(question_only, top_n=third, threshold=args.threshold)
                relevant_docs = combine_results(docs_llm, combine_results(docs_normal, docs_nlp))
            elif args.rag == 'v6':
                docs_normal = llm_rag.search_documents(question_only, top_n=args.top_n, threshold=args.threshold)
                docs_nlp = llm_rag.search_documents_with_nlp(question_only, top_n=args.top_n, threshold=args.threshold)
                relevant_docs = get_weighted_results([docs_nlp, docs_normal], weights=[0.5, 0.5])
            elif args.rag == 'v7':
                docs_llm = llm_rag.search_documents_with_llm(question_only, llm, top_n=args.top_n, threshold=args.threshold, temperature=args.rag_temperature, top_p=args.rag_top_p, repetition_penalty=args.rag_repetition_penalty, max_tokens=args.rag_max_tokens)
                docs_nlp = llm_rag.search_documents_with_nlp(question_only, top_n=args.top_n, threshold=args.threshold)
                relevant_docs = get_weighted_results([docs_llm, docs_nlp], weights=[0.5, 0.5])
            elif args.rag == 'v8':
                docs_llm = llm_rag.search_documents_with_llm(question_only, llm, top_n=args.top_n, threshold=args.threshold, temperature=args.rag_temperature, top_p=args.rag_top_p, repetition_penalty=args.rag_repetition_penalty, max_tokens=args.rag_max_tokens)
                docs_normal = llm_rag.search_documents(question_only, top_n=args.top_n, threshold=args.threshold)
                docs_nlp = llm_rag.search_documents_with_nlp(question_only, top_n=args.top_n, threshold=args.threshold)
                relevant_docs = get_weighted_results([docs_llm, docs_normal, docs_nlp], weights=[0.33, 0.33, 0.34])
            elif args.rag == 'v9':
                relevant_docs = llm_rag.search_documents_with_llm_and_nlp(question_only, llm, top_n=args.top_n, threshold=args.threshold, temperature=args.rag_temperature, top_p=args.rag_top_p, repetition_penalty=args.rag_repetition_penalty, max_tokens=args.rag_max_tokens)
            elif args.rag == "v10":
                docs_llm = llm_rag.search_documents_with_llm_and_nlp(question_only, llm, top_n=args.top_n, threshold=args.threshold, temperature=args.rag_temperature, top_p=args.rag_top_p, repetition_penalty=args.rag_repetition_penalty, max_tokens=args.rag_max_tokens)
                docs_nlp = llm_rag.search_documents_with_nlp(question_only, top_n=args.top_n, threshold=args.threshold)
                relevant_docs = get_weighted_results([docs_llm, docs_nlp], weights=[0.5, 0.5])
            elif args.rag == "v11":
                relevant_docs = llm_rag.search_documents_with_ner(question_only, top_n=args.top_n, threshold=args.threshold)
                
            else:
                relevant_docs = llm_rag.search_documents(question_only, top_n=args.top_n, threshold=args.threshold)

            if relevant_docs:
                if args.summarize:
                    final_summary = llm_rag.summarize_results([doc[0] for doc in relevant_docs])
                    relevant_text = final_summary
                else:
                    relevant_text = " ".join([doc[0] for doc in relevant_docs])
                print(relevant_text)
                prompt_with_rag = syst_prompt_with_relevant_text_version1.format(relevant_text, prompt)
            else:
                logger.warning(f"No relevant documents found for Question ID: {key.split(' ')[1]}")
                prompt_with_rag = None  # Ensure it's defined



        pred_option = pred_answer(llm, args, question_only, prompt, prompt_with_rag, map_ans, key, relevant_text)
        
        answer_sheet.append([key.split(' ')[1], pred_option, args.model_name])
        if pred_option is None:
            pred_option = 1
            
        logger.info(f"Question ID: {key.split(' ')[1]}, Predicted Answer ID: {pred_option}, Task: {args.model_name}, Label: {answer_only}" + (", Correct Answer ID:" + answer_only.split(':')[0].strip().split()[1] if args.benchmark else ''))
        
        
        if args.benchmark:
            answer_key = answer_only.split(':')[0].strip().split()[1]

            if pred_option == int(answer_key):
                correct_count += 1
                
            # Calculate current accuracy
            current_accuracy = (correct_count / i) * 100

            # Calculate best possible accuracy assuming all remaining questions are correct
            best_possible_accuracy = ((correct_count + (total_questions - i)) / total_questions) * 100

            logger.info(f"Current accuracy: {current_accuracy:.2f}%, Best possible accuracy: {best_possible_accuracy:.2f}%")

            # Check if best possible accuracy falls below the threshold
            if best_possible_accuracy < args.accuracy_threshold:
                logger.warning(f"Best possible accuracy ({best_possible_accuracy:.2f}%) falls below threshold ({args.accuracy_threshold}%). Stopping early.")
                break

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
        print(f"Benchmark accuracy: {accuracy:.2f}%" + " Correct answers: " + str(correct_count) + " Total questions: " + str(len(all_questions)))
    else:
        output_file = f'output/{args.model_name}_answer_sheet_final.csv'

    df_answer_sheet.to_csv(output_file, index=False)
