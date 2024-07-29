import subprocess
import csv
import numpy as np
import time
from tqdm import tqdm
import argparse
import multiprocessing
import os
import json
import optuna

class GPUManager:
    def __init__(self, num_gpus):
        self.available_gpus = multiprocessing.Queue()
        for i in range(num_gpus):
            self.available_gpus.put(i)

    def get_gpu(self):
        return self.available_gpus.get()

    def release_gpu(self, gpu_id):
        self.available_gpus.put(gpu_id)

def run_benchmark(params, print_output, gpu_id):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    command = [
        "python", "submission_runner.py",
        "--model_name", params['model_name'],
        "--benchmark",
        "--benchmark_num_questions", "-1",
        "--rag",  # RAG is always enabled
        "--db_path", params['db_path'],
        "--top_n", str(params['top_n']),
        "--threshold", str(params['threshold']),
        "--rag_temperature", str(params['rag_temperature']),
        "--rag_top_p", str(params['rag_top_p']),
        "--rag_repetition_penalty", str(params['rag_repetition_penalty']),
        "--rag_max_tokens", str(params['rag_max_tokens']),
        "--temperature", str(params['temperature']),
        "--top_p", str(params['top_p']),
        "--repetition_penalty", str(params['repetition_penalty']),
    ]

    if params['lora_path']:
        command.extend(["--lora", "--lora_path", params['lora_path']])
    
    if params['summarize']:
        command.append("--summarize")
    
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, env=env)
        output, error = process.communicate(timeout=3600)  # 1 hour timeout

        if print_output:
            print(f"GPU {gpu_id}: Output: {output}")
            print(f"GPU {gpu_id}: Error: {error}")

        if process.returncode != 0:
            print(f"GPU {gpu_id}: Process returned non-zero exit code: {process.returncode}")
            print(f"GPU {gpu_id}: Error output: {error}")
            return 0.0, command

        accuracy = None
        for line in output.split("\n"):
            if line.startswith("Benchmark accuracy:"):
                accuracy = float(line.split(":")[1].strip().rstrip("%"))
                break

        if accuracy is None:
            print(f"GPU {gpu_id}: Could not find accuracy in output")
            return 0.0, command

        return accuracy, command

    except subprocess.TimeoutExpired:
        print(f"GPU {gpu_id}: Benchmark execution timed out after 1 hour")
        return 0.0, command
    except Exception as e:
        print(f"GPU {gpu_id}: Error occurred during benchmark execution: {str(e)}")
        return 0.0, command

def objective(trial, print_output, gpu_manager, static_params):
    gpu_id = gpu_manager.get_gpu()
    try:
        params = {
            'model_name': static_params['model_name'],
            'temperature': trial.suggest_float('temperature', 0.15, 0.3),
            'top_p': trial.suggest_float('top_p', 0.85, 0.96),
            'repetition_penalty': trial.suggest_float('repetition_penalty', 1.05, 1.15),
            'rag_temperature': trial.suggest_float('rag_temperature', 0.24, 0.28),
            'rag_top_p': trial.suggest_float('rag_top_p', 0.82, 0.98),
            'rag_repetition_penalty': trial.suggest_float('rag_repetition_penalty', 1.05, 1.08),
            'rag_max_tokens': trial.suggest_int('rag_max_tokens', 14, 18),
            'top_n': trial.suggest_int('top_n', 3, 6),
            'threshold': trial.suggest_float('threshold', 0.02, 0.11),
            'db_path': trial.suggest_categorical('db_path', [
                "output/db_gte-large-preprocessed-2"
            ]),
            'lora_path': trial.suggest_categorical('lora_path', [
                "./fine_tuned_models/phi-2-finetuned-with-rag"
            ]),
            'summarize': trial.suggest_categorical('summarize', [False]),
        }
        
        accuracy, command = run_benchmark(params, print_output, gpu_id)
        return accuracy
    finally:
        gpu_manager.release_gpu(gpu_id)

def optimize_parameters(print_output, num_gpus, study_name, storage):
    static_params = {
        'model_name': "phi2",
    }

    results_file = "optimization_results.csv"
    def logger(study, trial):
        with open(results_file, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                static_params['model_name'],
                trial.params['temperature'],
                trial.params['top_p'],
                trial.params['repetition_penalty'],
                trial.params['rag_temperature'],
                trial.params['rag_top_p'],
                trial.params['rag_repetition_penalty'],
                trial.params['rag_max_tokens'],
                trial.params['top_n'],
                trial.params['threshold'],
                trial.params['db_path'],
                trial.params['lora_path'],
                trial.params['summarize'],
                1,  # weight_normal
                1,  # weight_llm
                0,  # weight_nlp
                0,  # weight_ner
                trial.value,
                json.dumps(trial.user_attrs.get('command', []))
            ])

    if not os.path.isfile(results_file):
        with open(results_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "Model", "Temperature", "Top P", "Repetition Penalty", 
                "RAG Temperature", "RAG Top P", "RAG Repetition Penalty", 
                "RAG Max Tokens", "Top N", "Threshold", "DB Path", "LoRA Path", 
                "Summarize", "Weight Normal", "Weight LLM", "Weight NLP", "Weight NER",
                "Accuracy", "Full Command"
            ])

    study = optuna.create_study(study_name=study_name, storage=storage, load_if_exists=True, direction='maximize')
    
    start_time = time.time()
    n_trials = 500  # Adjust as needed

    progress = tqdm(total=n_trials, desc="Running benchmarks")

    def update_progress(study, trial):
        progress.update(1)
        elapsed_time = time.time() - start_time
        avg_time_per_trial = elapsed_time / (trial.number + 1)
        remaining_trials = n_trials - (trial.number + 1)
        estimated_remaining_time = avg_time_per_trial * remaining_trials
        progress.set_postfix_str(f"Elapsed: {elapsed_time:.2f}s, Remaining: {estimated_remaining_time:.2f}s ({estimated_remaining_time / 3600:.2f}h)")

    gpu_manager = GPUManager(num_gpus)

    study.optimize(
        lambda trial: objective(trial, print_output, gpu_manager, static_params),
        n_trials=n_trials,
        n_jobs=num_gpus,
        callbacks=[logger, update_progress]
    )

    print("\nOptimization results:")
    print(f"Best accuracy: {study.best_value:.2f}%")
    print("Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    print("\nParameter importance:")
    importance = optuna.importance.get_param_importances(study)
    for key, value in importance.items():
        print(f"  {key}: {value:.4f}")

    print("\nTop 10 trials:")
    trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:10]
    for trial in trials:
        print(f"  Accuracy: {trial.value:.2f}%")
        print(f"  Parameters: {trial.params}")
        print(f"  Full Command: {json.dumps(trial.user_attrs.get('command', []))}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize model parameters for benchmark tests using multiple GPUs.")
    parser.add_argument("-p", "--print_output", action="store_true", help="Print the output of each subprocess to the console.")
    parser.add_argument("-g", "--num_gpus", type=int, default=1, help="Number of GPUs to use for parallel processing.")
    parser.add_argument("--study_name", type=str, default="benchmark_optimization", help="Name of the Optuna study.")
    parser.add_argument("--storage", type=str, default="sqlite:///optuna_study.db", help="Storage URL for the Optuna study.")
    args = parser.parse_args()

    optimize_parameters(args.print_output, args.num_gpus, args.study_name, args.storage)