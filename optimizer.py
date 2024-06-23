import subprocess
import csv
import itertools
import numpy as np
import time
from tqdm import tqdm
import argparse
import multiprocessing
import os
import json

def run_benchmark(model_name, rag, temperature, top_p, repetition_penalty, rag_temperature, rag_top_p, rag_repetition_penalty, top_n, threshold, db_path, lora_path, print_output, gpu_id):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    command = [
        "python", "submission_runner.py",
        "--model_name", model_name,
        "--benchmark",
        "--benchmark_num_questions", "-1",
        "--db_path", db_path,
        "--top_n", str(top_n),
        "--threshold", str(threshold)
    ]

    if rag:
        command.extend(["--rag", rag])
        if rag_temperature != -1:
            command.extend(["--rag_temperature", str(rag_temperature)])
            if rag_top_p is not None:
                command.extend(["--rag_top_p", str(rag_top_p)])
        if rag_repetition_penalty is not None:
            command.extend(["--rag_repetition_penalty", str(rag_repetition_penalty)])

    if temperature != -1:
        command.extend(["--temperature", str(temperature)])
        if top_p is not None:
            command.extend(["--top_p", str(top_p)])
    
    if repetition_penalty is not None:
        command.extend(["--repetition_penalty", str(repetition_penalty)])

    if lora_path:
        command.extend(["--lora", "--lora_path", lora_path])

    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, env=env)
        output = ""
        for line in process.stdout:
            if print_output:
                print(f"GPU {gpu_id}: {line}", end="")
            output += line

        process.wait()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)

        accuracy = None
        for line in output.split("\n"):
            if line.startswith("Benchmark accuracy:"):
                accuracy = float(line.split(":")[1].strip().rstrip("%"))
                break

    except (subprocess.CalledProcessError, ValueError, IndexError):
        accuracy = 0.0
        if print_output:
            print(f"GPU {gpu_id}: Error occurred during benchmark execution. Setting accuracy to 0.0%")

    return accuracy, command


def format_params(params):
    param_names = ["Model", "RAG", "Temperature", "Top P", "Repetition Penalty", 
                   "RAG Temperature", "RAG Top P", "RAG Repetition Penalty", 
                   "Top N", "Threshold", "DB Path", "LoRA Path"]
    formatted = []
    for name, value in zip(param_names, params):
        if value is not None and value != -1:
            formatted.append(f"{name}: {value}")
    return ", ".join(formatted)


def worker(queue, results, print_output, gpu_id):
    while True:
        params = queue.get()
        if params is None:
            break
        print(f"\nGPU {gpu_id}: Testing parameters: {format_params(params)}")
        accuracy, command = run_benchmark(*params, print_output, gpu_id)
        results.put((params, accuracy, command))

def load_tested_combinations(results_file):
    tested_combinations = set()
    if os.path.isfile(results_file):
        with open(results_file, "r") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for row in reader:
                tested_combinations.add(tuple(row[:-2]))  # Exclude accuracy and command
                
    print(f"Loaded {len(tested_combinations)} previously tested combinations from {results_file}")
    return tested_combinations

def optimize_parameters(print_output, num_gpus):
    models = ["phi2"]
    rag_versions = ["v4"]
    temperatures = [ .1, .2, .3]
    top_ps = [0.9, 0.95]
    repetition_penalties = [1.1, 1.2]
    rag_temperatures = [-1, 0.1, 0.2, .3]
    rag_top_ps = [None, 0.9, 0.95]
    rag_repetition_penalties = [None, 1.0, 1.1, 1.2]
    top_ns = [6]
    thresholds = [0.1]
    db_paths = ["output/db_gte-large-preprocessed-2"]
    lora_paths = ["./fine_tuned_models/phi-2-finetuned-with-rag"]

    results_file = "optimization_results.csv"
    tested_combinations = load_tested_combinations(results_file)

    all_combinations = []
    for combo in itertools.product(models, rag_versions, temperatures, top_ps, repetition_penalties, 
                                   rag_temperatures, rag_top_ps, rag_repetition_penalties, 
                                   top_ns, thresholds, db_paths, lora_paths):
        model, rag, temp, top_p, rep_penalty, rag_temp, rag_top_p, rag_rep_penalty, top_n, threshold, db_path, lora_path = combo
        
        # Skip combinations where RAG is None but RAG-specific parameters are set
        if rag is None and (rag_temp != -1 or rag_top_p is not None or rag_rep_penalty is not None):
            continue
        
        # Skip combinations where temperature is -1 but top_p is set
        if temp == -1 and top_p is not None:
            continue
        
        # Skip combinations where RAG temperature is -1 but RAG top_p is set
        if rag_temp == -1 and rag_top_p is not None:
            continue
        
        # Skip combinations that have already been tested
        if combo in tested_combinations:
            continue
        
        all_combinations.append(combo)

    total_tests = len(all_combinations)

    queue = multiprocessing.Queue()
    results = multiprocessing.Queue()

    for params in all_combinations:
        queue.put(params)

    processes = []
    for i in range(num_gpus):
        p = multiprocessing.Process(target=worker, args=(queue, results, print_output, i))
        processes.append(p)
        p.start()

    for _ in range(num_gpus):
        queue.put(None)

    progress = tqdm(total=total_tests, desc="Running benchmarks")

    start_time = time.time()
    completed_tests = 0
    best_accuracy = 0.0
    best_parameters = None

    file_exists = os.path.isfile(results_file)

    with open(results_file, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        if not file_exists:
            writer.writerow(["Model", "RAG", "Temperature", "Top P", "Repetition Penalty", 
                             "RAG Temperature", "RAG Top P", "RAG Repetition Penalty", 
                             "Top N", "Threshold", "DB Path", "LoRA Path", "Accuracy", "Full Command"])

        while completed_tests < total_tests:
            params, accuracy, command = results.get()
            writer.writerow(list(params) + [accuracy, json.dumps(command)])
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_parameters = params

            completed_tests += 1
            elapsed_time = time.time() - start_time
            avg_time_per_test = elapsed_time / completed_tests
            remaining_tests = total_tests - completed_tests
            estimated_remaining_time = avg_time_per_test * remaining_tests
            progress.set_postfix_str(f"Elapsed: {elapsed_time:.2f}s, Remaining: {estimated_remaining_time:.2f}s ({estimated_remaining_time / 3600:.2f}h)")
            progress.update(1)

            print(f"\nTest Accuracy: {accuracy:.2f}%")
            print(f"Parameters: {format_params(params)}")
            if best_parameters:
                print(f"Current Best Accuracy: {best_accuracy:.2f}%")
                print(f"Best Parameters: {format_params(best_parameters)}")
            csvfile.flush()

    for p in processes:
        p.join()

    print("\nOptimization results:")
    with open(results_file, "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        results = list(reader)

    results.sort(key=lambda x: float(x[-2]) if x[-2] != 'None' else 0, reverse=True)

    for result in results[:10]:  # Print top 10 results
        print(f"Accuracy: {float(result[-2]):.2f}%")
        print(f"Parameters: {result[:-2]}")
        print(f"Full Command: {result[-1]}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize model parameters for benchmark tests using multiple GPUs.")
    parser.add_argument("-p", "--print_output", action="store_true", help="Print the output of each subprocess to the console.")
    parser.add_argument("-g", "--num_gpus", type=int, default=1, help="Number of GPUs to use for parallel processing.")
    args = parser.parse_args()

    optimize_parameters(args.print_output, args.num_gpus)