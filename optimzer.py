import subprocess
import csv
import itertools
import numpy as np
import fcntl

def run_benchmark(model_name, rag, temperature, top_n, threshold):
    command = [
        "python", "submission_runner.py",
        "--model_name", model_name,
        "--benchmark",
        "--benchmark_num_questions", "-1",
        "--temperature", str(temperature),
    ]

    if rag:
        command.extend(["--rag", rag])
        command.extend(["--top_n", str(top_n)])
        command.extend(["--threshold", str(threshold)])

    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        output = ""
        for line in process.stdout:
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
        print("Error occurred during benchmark execution. Setting accuracy to 0.0%")

    return accuracy


def optimize_parameters():
    models = ["phi2", "falcon7b"]
    rag_versions = [None, "x", "v2", "v3"]
    temperatures = [-1] + [round(t, 1) for t in list(np.arange(0.1, 1.1, 0.1))]
    top_ns = list(range(1, 5))
    thresholds = [round(t, 1) for t in list(np.arange(0.0, 1.1, 0.1))]

    with open("optimization_results.csv", "a", newline="") as csvfile:
        fcntl.flock(csvfile, fcntl.LOCK_EX)  # Acquire an exclusive lock on the file
        writer = csv.writer(csvfile)

        # Check if the file is empty and write the header row if necessary
        csvfile.seek(0)
        if csvfile.tell() == 0:
            writer.writerow(["Model", "RAG", "Temperature", "Top N", "Threshold", "Accuracy"])

        for model, rag, temperature in itertools.product(models, rag_versions, temperatures):
            if rag is None:
                print(f"Running benchmark with parameters: model={model}, rag=None, temperature={temperature}")
                accuracy = run_benchmark(model, None, temperature, None, None)
                writer.writerow((model, None, temperature, None, None, accuracy))
            else:
                for top_n, threshold in itertools.product(top_ns, thresholds):
                    print(f"Running benchmark with parameters: model={model}, rag={rag}, temperature={temperature}, top_n={top_n}, threshold={threshold}")
                    accuracy = run_benchmark(model, rag, temperature, top_n, threshold)
                    writer.writerow((model, rag, temperature, top_n, threshold, accuracy))

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
    optimize_parameters()