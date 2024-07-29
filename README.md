# TeleQA: Advanced Question-Answering Framework for Telecommunications and 3GPP Standards

## Overview and Objectives

TeleQA is a cutting-edge question-answering framework specifically designed for the telecommunications domain, with a focus on 3GPP standards. This framework leverages state-of-the-art Language Models (LLMs) and Retrieval-Augmented Generation (RAG) techniques to provide accurate and context-aware answers to complex telecommunications questions.

**Objectives:**
1. Provide accurate and context-aware answers to telecommunications and 3GPP standards-related questions.
2. Offer a flexible and customizable framework for various use cases within the telecom industry.
3. Enable efficient benchmarking and performance evaluation of different LLMs and retrieval methods.
4. Facilitate easy integration of the solution into existing telecom knowledge management systems.

## Architecture Diagram

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600">
  <rect x="10" y="10" width="780" height="580" fill="none" stroke="black" stroke-width="2"/>
  
  <!-- Input Layer -->
  <rect x="30" y="30" width="150" height="80" fill="lightblue" stroke="black"/>
  <text x="105" y="75" text-anchor="middle">Input Layer</text>
  <text x="105" y="95" text-anchor="middle" font-size="10">Questions &amp; Documents</text>
  
  <!-- Preprocessing Layer -->
  <rect x="30" y="140" width="150" height="80" fill="lightgreen" stroke="black"/>
  <text x="105" y="185" text-anchor="middle">Preprocessing</text>
  <text x="105" y="205" text-anchor="middle" font-size="10">Chunking &amp; Embedding</text>
  
  <!-- Retrieval Layer -->
  <rect x="210" y="30" width="150" height="190" fill="lightyellow" stroke="black"/>
  <text x="285" y="50" text-anchor="middle">Retrieval Layer</text>
  <text x="285" y="70" text-anchor="middle" font-size="10">Normal Search</text>
  <text x="285" y="90" text-anchor="middle" font-size="10">LLM-based Search</text>
  <text x="285" y="110" text-anchor="middle" font-size="10">NLP-based Search</text>
  <text x="285" y="130" text-anchor="middle" font-size="10">NER-based Search</text>
  <text x="285" y="150" text-anchor="middle" font-size="10">ChromaDB</text>
  
  <!-- Language Model Layer -->
  <rect x="390" y="30" width="150" height="190" fill="lightpink" stroke="black"/>
  <text x="465" y="50" text-anchor="middle">LLM Layer</text>
  <text x="465" y="70" text-anchor="middle" font-size="10">Phi-2</text>
  <text x="465" y="90" text-anchor="middle" font-size="10">Falcon-7B</text>
  <text x="465" y="110" text-anchor="middle" font-size="10">LoRA Fine-tuning</text>
  
  <!-- RAG Integration Layer -->
  <rect x="570" y="30" width="150" height="80" fill="lightgray" stroke="black"/>
  <text x="645" y="75" text-anchor="middle">RAG Integration</text>
  
  <!-- Output Layer -->
  <rect x="570" y="140" width="150" height="80" fill="#FFD700" stroke="black"/>
  <text x="645" y="185" text-anchor="middle">Output Layer</text>
  <text x="645" y="205" text-anchor="middle" font-size="10">Answer Generation</text>
  
  <!-- Benchmarking Layer -->
  <rect x="390" y="250" width="330" height="80" fill="#98FB98" stroke="black"/>
  <text x="555" y="295" text-anchor="middle">Benchmarking &amp; Optimization</text>
  
  <!-- Arrows -->
  <line x1="180" y1="70" x2="210" y2="70" stroke="black" stroke-width="2" marker-end="url(#arrowhead)"/>
  <line x1="180" y1="180" x2="210" y2="180" stroke="black" stroke-width="2" marker-end="url(#arrowhead)"/>
  <line x1="360" y1="125" x2="390" y2="125" stroke="black" stroke-width="2" marker-end="url(#arrowhead)"/>
  <line x1="540" y1="125" x2="570" y2="70" stroke="black" stroke-width="2" marker-end="url(#arrowhead)"/>
  <line x1="645" y1="110" x2="645" y2="140" stroke="black" stroke-width="2" marker-end="url(#arrowhead)"/>
  <line x1="465" y1="220" x2="465" y2="250" stroke="black" stroke-width="2" marker-end="url(#arrowhead)"/>
  
  <!-- Arrowhead definition -->
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="0" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" />
    </marker>
  </defs>
</svg>

This diagram illustrates the main components and data flow of the TeleQA framework.

## Key Features

- **Multiple LLM Support**: Integrates with various language models, including Phi-2 and Falcon-7B.
- **Retrieval-Augmented Generation (RAG)**: Enhances answer accuracy by incorporating relevant document retrieval.
- **Flexible Document Search**: Supports multiple search methods, including normal, LLM-based, NLP-based, and NER-based approaches.
- **Benchmarking Capabilities**: Allows for easy evaluation of model performance against known answers.
- **Dynamic Weighting**: Enables fine-tuning of different search methods for optimal results.
- **Extensive Logging**: Provides detailed logs for tracking progress and debugging.
- **Customizable via CLI**: Offers a wide range of command-line arguments for easy experimentation.
- **Hyperparameter Optimization**: Includes an optimization script for finding the best configuration.

## System Requirements

- **GPU**: Tesla P100-PCIE-16GB (or equivalent)
- **VRAM**: 16GB
- **CUDA Version**: 12.4
- **Driver Version**: 550.90.07

## Installation

1. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Download and install the spaCy model:
   ```
   python -m spacy download en_core_web_lg
   ```

## ETL Process

### Extract
- Data sources: Telecom documentation, 3GPP standards, and user-provided questions.
- Data format: Text files (DOCX) and JSON files containing questions and document corpus.
- Extraction method: File reading and parsing using `docx_preprocess.py`.

### Transform
- Document preprocessing: Tokenization, normalization, and encoding for efficient retrieval.
- Question processing: Parsing and formatting for input to the LLM.
- Chunking: Documents are split into smaller chunks with specified size and overlap.

### Load
- Document storage: Processed and embedded documents are loaded into ChromaDB for efficient retrieval.
- Model loading: LLMs and fine-tuned models (e.g., LoRA) are loaded into memory for inference.

## Data Modeling

### Feature Engineering
- Document embeddings are generated using the Alibaba-NLP/gte-large-en-v1.5 model.
- Named Entity Recognition (NER) is applied for enhanced retrieval using SpaCy.

### Model Selection
- Multiple LLMs supported, including Phi-2 and Falcon-7B.
- Retrieval-Augmented Generation (RAG) for enhanced context awareness.

### Training and Fine-tuning
- LoRA fine-tuning supported for domain adaptation.

### Model Validation
- Benchmarking capabilities with accuracy calculation and early stopping.
- Hyperparameter optimization using Optuna for finding the best configuration.

## Inference

- Deployment: The framework runs on systems with GPU support (Tesla P100-PCIE-16GB or equivalent).
- Input processing: Questions are processed through various search methods (normal, LLM-based, NLP-based, NER-based).
- Output generation: Answers are generated using the selected LLM, enhanced by retrieved context when RAG is enabled.
- Model updates: Supported through LoRA fine-tuning and flexible model selection.

## Run Time

- Average runtime: Approximately 1.5 hours to process all questions in the dataset on a Tesla P100-PCIE-16GB GPU.
- Individual component timings:
  - Document indexing: Varies based on corpus size
  - Question processing: Milliseconds per question
  - Inference: Seconds per question (varies based on LLM and RAG configuration)

## Performance Metrics

- Accuracy: Calculated in benchmark mode against known answers.
- Early stopping threshold: Configurable accuracy threshold for efficient benchmarking.
- Hyperparameter optimization: Optuna-based optimization for finding the best configuration.
- Additional metrics tracked during optimization:
  - Temperature, Top P, Repetition Penalty (for both main LLM and RAG)
  - Number of retrieved documents (Top N)
  - Similarity threshold for document retrieval

## Error Handling and Logging

- Comprehensive logging system using the `loguru` library.
- Logs capture the question-answering process, document retrieval, and any issues encountered during execution.
- Error handling for timeouts, subprocess failures, and unexpected outputs.

## Usage

To run the TeleQA framework, use the following command:

```
python main.py [arguments]
```

For hyperparameter optimization:

```
python optimize_parameters.py [arguments]
```

To run with best found parameters use the runner file
```
./runner.sh
```

### Key Arguments

- `--model_name`: Choose the LLM (e.g., "phi2", "falcon7b")
- `--rag`: Enable Retrieval-Augmented Generation
- `--question_path`: Path to the question dataset
- `--benchmark`: Run in benchmark mode
- `--summarize`: Summarize retrieved documents
- `--top_n`: Number of top documents to retrieve
- `--temperature`: Temperature for LLM generation
- `--lora`: Apply LoRA fine-tuning
- `--db_path`: Path to the ChromaDB database

For a full list of arguments, run:

```
python main.py --help
```

