import uuid
import os
import time
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from transformers import AutoModel, AutoTokenizer
import sys
from loguru import logger
import chromadb
import spacy
from .docx_preprocess import get_header_chunks
from typing import List, Dict
import re
import json
import ast

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_API_TOKEN"] = "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"  # Replace with your Hugging Face token

class llmRag:
    def __init__(self, db_path='output/db', collection_name='my_documents', chunk_size=200, overlap=50,
                 batch_size=50) -> None:
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
        
        # Load SpaCy model for keyword extraction
        self.nlp = spacy.load('en_core_web_lg')

    def encode_text(self, texts: List[str], instruction_prefix: str = ""):
        max_length = 1000  # Adjusted max length to match the reduced chunk size
        inputs = self.tokenizer([instruction_prefix + text for text in texts], return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(self.device)
        with torch.no_grad():
            with autocast():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0]  # Using CLS token
        return F.normalize(embeddings, p=2, dim=1)

    def extract_keywords(self, text: str) -> str:
        doc = self.nlp(text)
        keywords = [token.text for token in doc if token.is_alpha and not token.is_stop]
        entities = [ent.text for ent in doc.ents]
        return " ".join(keywords)

    def extract_3gpp_release(self, text):
            # Define the regular expression pattern to match the metadata fields
            pattern = r"\[(3GPP Release \d+)\]"
            
            # Find all matches in the text
            matches = re.findall(pattern, text)
            
            # Extract the first match (assuming there's only one relevant metadata field per document)
            metadata = matches[0] if matches else None
            
            # Remove all metadata fields from the text
            clean_text = re.sub(pattern, '', text).strip()
            
            return metadata, clean_text

    def search_documents(self, query: str, top_n: int = 5, threshold: float = 0.0):
        release, query = self.extract_3gpp_release(query)

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
        doc_files = [filename for filename in os.listdir(folder_path) if filename.endswith('.docx')]
        total_files = len(doc_files)
        start_time = time.time()

        for idx, filename in enumerate(tqdm(doc_files, desc="Processing documents")):
            file_path = os.path.join(folder_path, filename)
            header_chunks = get_header_chunks(file_path)
            if len(header_chunks) == 0:
                sys.exit(f"No header chunks extracted from {filename}")
                
            logger.info(f"Extracted {len(header_chunks)} header chunks from {filename}")
            chunks = self.chunkify(header_chunks)
            #logger.info(f"Starting to store {len(chunks)} overlapping chunks from {filename}")

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
                    #logger.info(f"Stored batch {i // self.batch_size + 1} for {filename}")
                except Exception as e:
                    logger.error(f"Failed to store batch {i // self.batch_size + 1} for {filename}: {str(e)}")
                    break

                # Clear GPU cache
                torch.cuda.empty_cache()
            
            # Calculate elapsed time and estimate remaining time
            elapsed_time = time.time() - start_time
            avg_time_per_file = elapsed_time / (idx + 1)
            remaining_files = total_files - (idx + 1)
            estimated_remaining_time = avg_time_per_file * remaining_files
            
            tqdm.write(f"Elapsed time: {elapsed_time:.2f}s, Estimated remaining time: {estimated_remaining_time:.2f}s ({estimated_remaining_time / 60:.2f} minutes)")



    def store_documents_from_json(self, json_path: str):
        """
        Load preprocessed document chunks from a JSON file and store them in the database.
        """
        with open(json_path, 'r') as json_file:
            preprocessed_data = json.load(json_file)
        
        total_files = len(preprocessed_data)
        start_time = time.time()

        for filename, navigation_dict in preprocessed_data.items():
            # No need to convert keys; use them as they are
            header_chunks = navigation_dict
            if len(header_chunks) == 0:
                sys.exit(f"No header chunks extracted from {filename}")
                
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
            
            # Calculate elapsed time and estimate remaining time
            elapsed_time = time.time() - start_time
            avg_time_per_file = elapsed_time / (list(preprocessed_data.keys()).index(filename) + 1)
            remaining_files = total_files - (list(preprocessed_data.keys()).index(filename) + 1)
            estimated_remaining_time = avg_time_per_file * remaining_files
            
            tqdm.write(f"Elapsed time: {elapsed_time:.2f}s, Estimated remaining time: {estimated_remaining_time:.2f}s ({estimated_remaining_time / 60:.2f} minutes)")



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
    
    def search_documents_with_llm(self, query: str, llm_pipeline, top_n: int = 5, threshold: float = 0.0, temperature=0.3, max_tokens=15, top_p=0.9, repetition_penalty=1.2):
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
        
        refined_query = f"{query} {candidate_answers}"

        
        results = self.search_documents(refined_query, top_n, threshold)
        
        return results
    
    def search_documents_with_nlp(self, query: str, top_n: int = 5, threshold: float = 0.0):
        logger.info(f"Extracting keywords from the query using NLP...")
        keywords = self.extract_keywords(query)
        logger.info(f"Extracted keywords: {keywords}")

        results = self.search_documents(keywords, top_n, threshold)
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
    rag = llmRag(db_path='output/db_gte-large-preprocessed-2')
    rag.store_documents_from_json("processed_dicts.json")
    query = "The Reference Vector file format is used for which purpose in 3GPP standards?"
    results = rag.search_documents(query, top_n=10, threshold=0.1)
    for i, result in enumerate(results):
        print(f"Result {i + 1}: {result[0]}")

