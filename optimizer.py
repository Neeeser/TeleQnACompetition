import subprocess
import csv
import itertools
import numpy as np
import fcntl
import time
from tqdm import tqdm
import argparse

def run_benchmark(model_name, rag, temperature, top_n, threshold, print_output):
    command = [
        "python", "submission_runner.py",
        "--model_name", model_name,
        "--benchmark",
        "--benchmark_num_questions", "200",
        "--temperature", str(temperature),
        "--lora"
    ]

    if rag:
        command.extend(["--rag", rag])
        command.extend(["--top_n", str(top_n)])
        command.extend(["--threshold", str(threshold)])

    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        output = ""
        for line in process.stdout:
            if print_output:
                print(line, end="")
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
            print("Error occurred during benchmark execution. Setting accuracy to 0.0%")

    return accuracy


def optimize_parameters(print_output):
    models = ["phi2"]
    rag_versions = ["x", "nlp"]
    temperatures = [-1, .1, .2, .3, .4]
    top_ns = [6, 5, 4, 3, 2]
    thresholds = [0.0, 0.1, 0.2, .3]

    total_tests = len(models) * len(rag_versions) * len(temperatures) * len(top_ns) * len(thresholds)
    progress = tqdm(total=total_tests, desc="Running benchmarks")

    start_time = time.time()
    completed_tests = 0

    best_accuracy = 0.0
    best_parameters = None

    with open("optimization_results_lora.csv", "a", newline="") as csvfile:
        fcntl.flock(csvfile, fcntl.LOCK_EX)  # Acquire an exclusive lock on the file
        writer = csv.writer(csvfile)

        # Check if the file is empty and write the header row if necessary
        csvfile.seek(0)
        if csvfile.tell() == 0:
            writer.writerow(["Model", "RAG", "Temperature", "Top N", "Threshold", "Accuracy"])

        for model, rag, temperature in itertools.product(models, rag_versions, temperatures):
            if rag is None:
                print(f"Running benchmark with parameters: model={model}, rag=None, temperature={temperature}")
                accuracy = run_benchmark(model, None, temperature, None, None, print_output)
                writer.writerow((model, None, temperature, None, None, accuracy))
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_parameters = (model, None, temperature, None, None)
            else:
                for top_n, threshold in itertools.product(top_ns, thresholds):
                    print(f"Running benchmark with parameters: model={model}, rag={rag}, temperature={temperature}, top_n={top_n}, threshold={threshold}")
                    test_start_time = time.time()
                    accuracy = run_benchmark(model, rag, temperature, top_n, threshold, print_output)
                    test_end_time = time.time()
                    writer.writerow((model, rag, temperature, top_n, threshold, accuracy))
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_parameters = (model, rag, temperature, top_n, threshold)
                    completed_tests += 1
                    elapsed_time = test_end_time - start_time
                    avg_time_per_test = elapsed_time / completed_tests
                    remaining_tests = total_tests - completed_tests
                    estimated_remaining_time = avg_time_per_test * remaining_tests
                    progress.set_postfix_str(f"Elapsed: {elapsed_time:.2f}s, Remaining: {estimated_remaining_time:.2f}s ({estimated_remaining_time / 3600:.2f}h)")
                    progress.update(1)
                    print(f"\nTest Accuracy: {accuracy:.2f}%")
                    if best_parameters:
                        print(f"Current Best Accuracy: {best_accuracy:.2f}% with parameters: Model: {best_parameters[0]}, RAG: {best_parameters[1]}, Temperature: {best_parameters[2]}, Top N: {best_parameters[3]}, Threshold: {best_parameters[4]}")

            csvfile.flush()  # Flush the file buffer to ensure the results are written immediately
        
        fcntl.flock(csvfile, fcntl.LOCK_UN)  # Release the lock on the file

    with open("optimization_results.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        results = list(reader)

    results = results[1:]  # Exclude the header row
    results.sort(key=lambda x: float(x[5]) if x[5] != 'None' else 0, reverse=True)

    print("Optimization results:")
    for result in results:
        if result[5] != 'None':
            print(f"Model: {result[0]}, RAG: {result[1]}, Temperature: {result[2]}, Top N: {result[3]}, Threshold: {result[4]}, Accuracy: {float(result[5]):.2f}%")
        else:
            print(f"Model: {result[0]}, RAG: {result[1]}, Temperature: {result[2]}, Top N: {result[3]}, Threshold: {result[4]}, Accuracy: N/A")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize model parameters for benchmark tests.")
    parser.add_argument("-p", "--print_output", action="store_true", help="Print the output of each subprocess to the console.")
    args = parser.parse_args()

    optimize_parameters(args.print_output)
