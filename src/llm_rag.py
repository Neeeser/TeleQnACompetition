import uuid
import os

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from transformers import AutoModel, AutoTokenizer

from loguru import logger
import chromadb

from docx_preprocess import get_header_chunks
from typing import List, Dict


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_API_TOKEN"] = "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"  # Replace with your Hugging Face token

class llmRag:
    def __init__(self, db_path='output/db', collection_name='my_documents', chunk_size=200, overlap=50,
                 batch_size=1) -> None:
        self.db_path = db_path
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.overlap = overlap

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.db_client = chromadb.PersistentClient(path=self.db_path)
        logger.info(f"Connected to ChromaDB at {self.db_path}...")
        self.collection = self.db_client.get_or_create_collection(self.collection_name)
        logger.info(f"Connected to collection: {collection_name}")

        self.model_path = 'Alibaba-NLP/gte-large-en-v1.5'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True).to(self.device)

    def encode_text(self, texts: List[str], instruction_prefix: str = ""):
        max_length = 8192
        inputs = self.tokenizer([instruction_prefix + text for text in texts], return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(self.device)
        with torch.no_grad():
            with autocast():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0]  # Using CLS token
        return F.normalize(embeddings, p=2, dim=1)

    def search_documents(self, query: str, top_n: int = 5, threshold: float = 0.0):
        logger.info(
            f"Searching for the top {top_n} documents similar to the query: '{query}' with threshold {threshold}")
        query_embedding = self.encode_text([query])
        query_embedding = query_embedding.cpu().numpy()

        query_result = self.collection.query(query_embeddings=query_embedding, n_results=top_n)
        documents = query_result["documents"][0]
        metadatas = query_result["metadatas"][0]
        scores = query_result["distances"][0]

        results = [(doc, meta['filename'], score) for doc, meta, score in zip(documents, metadatas, scores) if
                   score >= threshold]
        logger.info(f"Found {len(results)} documents above the similarity threshold.")
        return results

    def store_documents(self, folder_path: str):
        for filename in os.listdir(folder_path):
            if filename.endswith('.docx'):
                file_path = os.path.join(folder_path, filename)
                header_chunks = get_header_chunks(file_path)
                logger.info(f"Extracted {len(header_chunks)} header chunks from {filename}")
                chunks = self.chunkify(header_chunks)
                logger.info(f"Starting to store {len(chunks)} overlapping chunks from {filename}")

                for i in range(0, len(chunks), self.batch_size):
                    batch_docs = [{'text': chunk['text'], 'metadata': {'filename': filename, 'header': chunk['header'], 'text': chunk['text']}} for chunk in chunks[i:i + self.batch_size]]
                    ids = [str(uuid.uuid4()) for _ in batch_docs]
                    embeddings = self.encode_text([doc['text'] for doc in batch_docs])

                    try:
                        self.collection.add(
                            embeddings=embeddings.cpu().tolist(),
                            metadatas=[doc['metadata'] for doc in batch_docs],
                            ids=ids,
                            documents=[doc['text'] for doc in batch_docs]
                        )
                        logger.info(f"Stored batch {i // self.batch_size + 1} for {filename}")
                    except Exception as e:
                        logger.error(f"Failed to store batch {i // self.batch_size + 1} for {filename}: {str(e)}")
                        break
                    
                    # Clear GPU cache
                    torch.cuda.empty_cache()


    def chunkify(self, navigation_dict: Dict[tuple, str]) -> List[Dict[str, str]]:
        """ Takes in a dictionary of chunks from a docx file and returns a list of chunks with the specified chunk size and overlap """
        chunks = []
        for header, text in navigation_dict.items():
            words = text.split()
            for i in range(0, len(words), self.chunk_size - self.overlap):
                chunk = ' '.join(words[i:i + self.chunk_size])
                if len(chunk) > self.overlap:  # Ensure that chunk has meaningful content
                    chunks.append({
                        'text': chunk,
                        'header': header[1]
                    })
        return chunks
    def search_documents_with_llm(self, query: str, llm_pipeline, top_n: int = 5, threshold: float = 0.0):
        logger.info(f"Generating improved query using LLM pipeline...")
        llm_pipeline = llm_pipeline

        improved_query = llm_pipeline.call_local_model(
            prompt=(
                f"You are an expert in telecommunications and document retrieval. "
                f"Your task is to transform the following multiple-choice question into a highly effective search query. "
                f"The question is related to 3GPP standards, telecommunications procedures, or technical definitions. "
                f"Ensure the search query is specific, includes relevant technical terms, and clearly targets the needed information. "
                f"Original question: '{query}' "
                f"Effective search query (without any prefacing text): "
            ),
            temperature=0.1,
            max_tokens=150,
            top_p=0.9,
            repetition_penalty=1.2
        )
        improved_query_cleaned = improved_query.strip().replace("Assistant: ", "")
        logger.info(f"Improved query: {improved_query_cleaned}")

        results = self.search_documents(improved_query_cleaned, top_n, threshold)
        return results


    def summarize_individual_result(self, result: str, query: str, llm_pipeline):
        summary_prompt = (
            f" '{result}'. "
            f"Here is a summary of the text with only the key points: "
        )
        summary = llm_pipeline.call_local_model(
            prompt=summary_prompt,
            temperature=0.3,
            max_tokens=250,
            top_p=0.9,
            repetition_penalty=1.2
        )
        print(f"Generated summary: {summary.strip()}")
        return summary.strip()

    def summarize_results(self, query: str, results: List[str], llm_pipeline):
        logger.info(f"Summarizing the results from {len(results)} documents...")
        individual_summaries = [
            self.summarize_individual_result(result[0], query, llm_pipeline) for result in results
        ]
        combined_summaries = "\n".join(individual_summaries)
        logger.info(f"Generated combined summary: {combined_summaries}")
        return combined_summaries
    
if __name__ == '__main__':
    rag = llmRag(db_path='output/db_gte-large')
    rag.store_documents("../data/Test")
    query = "The Reference Vector file format is used for which purpose in 3GPP standards?"
    results = rag.search_documents(query, top_n=3, threshold=0.1)
    for i, result in enumerate(results):
        print(f"Result {i + 1}: {result[0]}")

