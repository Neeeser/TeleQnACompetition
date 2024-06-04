import os
import json
import numpy as np
import pandas as pd
import FreeSimpleGUI as sg
import threading
from copy import deepcopy
from evaluation_tools import check_questions_with_val_output_local

def update_plot(window, results):
    res = pd.DataFrame.from_dict({
        'categories': [ques['category'] for ques in results.values()],
        'correct': [ques['correct'] for ques in results.values()]
    })
    summary = res.groupby('categories').mean()
    summary['counts'] = res.groupby('categories').count()['correct'].values

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

model = "phi3"
questions_path = "TeleQnA.txt"
save_path = os.path.join(model + "_answers.txt")

n_questions = 1
max_attempts = 5

sg.theme('DefaultNoMoreNagging')

layout = [
    [sg.Text('Evaluation Dashboard', font=('Arial', 16, 'bold'))],
    [sg.Button('Run', key='-RUN-')],
    [sg.Text('', size=(50, 1), key='-PROGRESS-')],
    [sg.Graph((600, 400), (0, 0), (600, 400), key='-CANVAS-', background_color='white')],  # Increased graph dimensions
    [sg.Text('', size=(50, 2), key='-RESULT-')],
]

window = sg.Window('Evaluation Dashboard', layout, finalize=True, font=('Arial', 12))

while True:
    event, values = window.read(timeout=100)
    if event == sg.WINDOW_CLOSED:
        break
    elif event == '-RUN-':
        window['-RUN-'].update(disabled=True)  # Disable the run button
        threading.Thread(target=evaluate, args=(model, questions_path, save_path, n_questions, max_attempts, window), daemon=True).start()
    elif event == '-UPDATE-':
        update_plot(window, values['-UPDATE-'])
        window['-PROGRESS-'].update(f"Processed {len(values['-UPDATE-'])} questions")
    elif event == '-COMPLETE-':
        update_plot(window, values['-COMPLETE-'])
        window['-PROGRESS-'].update(f"Evaluation completed. Processed {len(values['-COMPLETE-'])} questions.")

window.close()
