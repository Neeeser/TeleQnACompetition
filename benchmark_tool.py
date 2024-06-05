import os
import json
import numpy as np
import pandas as pd
import FreeSimpleGUI as sg
import threading
from copy import deepcopy
from src.llm_utils import prepare_questions, syst_prompt_version1, syst_prompt_with_relevant_text_version1, regex_extraction
from src.llm_pipeline import llmPipeline
from src.llm_rag import llmRag
import openai
from loguru import logger


def check_questions_with_val_output_local(questions_dict, model="phi", use_rag=True):
    pipeline = llmPipeline()  # Initialize llmPipeline
    rag = llmRag()  # Initialize llmRag
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
        user_prompt, question_only, _ = prepare_questions(questions_only[q])

        syst_prompt = syst_prompt_version1
        if use_rag:
            relevant_docs = rag.search_documents(question_only, top_n=1, threshold=1)
            if relevant_docs:
                relevant_text = " ".join([doc[0] for doc in relevant_docs])
                syst_prompt = syst_prompt_with_relevant_text_version1.format(syst_prompt_version1, relevant_text)
            else:
                logger.warning(f"No relevant documents found for Question ID: {q.split(' ')[1]}")

        predicted_answer = pipeline.call_local_model(syst_prompt, user_prompt, model=model).strip()

        parsed_predicted_answers[q] = {
            "question": questions_dict[q]["question"],
            "answer": predicted_answer
        }

        if predicted_answer == answers_only[q].split(":")[0]:
            accepted_questions[q] = questions_dict[q]

    return accepted_questions, parsed_predicted_answers



def update_plot(window, results):
    res = pd.DataFrame.from_dict({
        'categories': [ques.get('category', 'Uncategorized') for ques in results.values()],
        'correct': [ques['correct'] for ques in results.values()]
    })

    if res.empty:
        window['-RESULT-'].update("No data to display.")
        return

    summary = res.groupby('categories').mean()
    summary['counts'] = res.groupby('categories').count()['correct'].values

    if summary.empty:
        window['-RESULT-'].update("No data to display.")
        return

    canvas = window['-CANVAS-']
    canvas.erase()

    graph_width, graph_height = 600, 400  # Increased graph dimensions
    bar_height = graph_height // len(summary)
    max_accuracy = 1.0

    for i in range(21):
        x = i * (graph_width - 20) // 20 + 10
        canvas.draw_line((x, graph_height - 10), (x, graph_height - 15))
        canvas.draw_text(f'{i * 0.05:.2f}', (x, graph_height - 5), font=('Arial', 8), color='black')

    for i, (category, accuracy) in enumerate(summary['correct'].items()):
        bar_width = int(accuracy / max_accuracy * (graph_width - 20))
        x = 10
        y = i * bar_height + 10  # Adjusted y-coordinate to move bars down
        canvas.draw_rectangle((x, y), (x + bar_width, y + bar_height - 5), fill_color='#1f77b4')
        canvas.draw_text(category, (x + bar_width + 5, y + bar_height // 2), font=('Arial', 8), color='black')

    canvas.draw_text('Accuracy', (graph_width // 2, graph_height - 25), font=('Arial', 10), color='black')
    canvas.draw_text('Categories', (10, graph_height // 2), font=('Arial', 10), color='black', angle=90)
    canvas.draw_text('Evaluation Results', (graph_width // 2, 10), font=('Arial', 12, 'bold'), color='black')

    window['-RESULT-'].update(f"Total number of questions answered: {len(results)}\n"
                              f"Final result: {np.mean([q['correct'] for q in results.values()]):.4f}")


def evaluate(model, questions_path, save_path, n_questions, max_attempts, window):
    print("Evaluating {}".format(model))

    with open(questions_path, encoding="utf-8") as f:
        loaded_json = f.read()
    all_questions = json.loads(loaded_json)

    end = len(all_questions)
    shuffled_idx = np.arange(len(all_questions))

    if os.path.exists(save_path):
        with open(save_path) as f:
            loaded_json = f.read()
        results = json.loads(loaded_json)
        start = len(results)
    else:
        results = {}
        start = 0

    print("Start at question: {}".format(start))

    k = 0

    for start_id in range(start, end, n_questions):
        attempts = 0
        end_id = np.minimum(start_id + n_questions, len(all_questions) - 1)

        q_names = ["question {}".format(shuffled_idx[k]) for k in range(start_id, end_id)]
        selected_questions = {q_name: all_questions[q_name] for q_name in q_names}

        while attempts < max_attempts:
            try:
                accepted_questions, parsed_predicted_answers = check_questions_with_val_output_local(selected_questions, model=model)

                for q in selected_questions:
                    parsed_predicted_answers[q]['answer']
                    results[q] = deepcopy(selected_questions[q])
                    results[q]['tested answer'] = parsed_predicted_answers[q]['answer']
                    results[q]['correct'] = q in accepted_questions

                break

            except Exception as e:
                attempts += 1
                print(f"Attempt {attempts} failed. Error: {e}")
                print("Retrying...")

        else:
            print(f"Failed after {max_attempts} attempts.")

        k += 1
        window.write_event_value('-UPDATE-', results)

    with open(save_path, 'w') as f:
        res_str = json.dumps(results)
        f.write(res_str)

    window.write_event_value('-COMPLETE-', results)

model = "phi2"
questions_path = "data/TeleQnA.txt"
save_path = os.path.join(model + "_answers.txt")

n_questions = 1
max_attempts = 5

sg.theme('DefaultNoMoreNagging')

layout = [
    [sg.Text('Evaluation Dashboard', font=('Arial', 16, 'bold'))],
    [sg.Button('Run', key='-RUN-')],
    [sg.Text('', size=(50, 1), key='-PROGRESS-')],
    [sg.Graph((600, 400), (0, 0), (600, 400), key='-CANVAS-', background_color='white')],
    [sg.Text('', size=(50, 2), key='-RESULT-')],
]

window = sg.Window('Evaluation Dashboard', layout, finalize=True, font=('Arial', 12))

while True:
    event, values = window.read(timeout=100)
    if event == sg.WINDOW_CLOSED:
        break
    elif event == '-RUN-':
        window['-RUN-'].update(disabled=True)
        threading.Thread(target=evaluate, args=(model, questions_path, save_path, n_questions, max_attempts, window), daemon=True).start()
    elif event == '-UPDATE-':
        update_plot(window, values['-UPDATE-'])
        window['-PROGRESS-'].update(f"Processed {len(values['-UPDATE-'])} questions")
    elif event == '-COMPLETE-':
        update_plot(window, values['-COMPLETE-'])
        window['-PROGRESS-'].update(f"Evaluation completed. Processed {len(values['-COMPLETE-'])} questions.")

window.close()
