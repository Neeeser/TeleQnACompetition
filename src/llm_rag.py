import uuid
import os
import docx2txt
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from transformers import AutoModel, AutoTokenizer
from typing import List
from loguru import logger
import chromadb
from .llm_pipeline import llmPipeline

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
                chunks = self.read_docx_helper(file_path)
                logger.info(f"Starting to store {len(chunks)} overlapping chunks from {filename}")

                for i in range(0, len(chunks), self.batch_size):
                    batch_docs = [{'text': chunk, 'metadata': {'filename': filename, 'text': chunk}} for chunk in chunks[i:i + self.batch_size]]
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

    def read_docx_helper(self, file_path: str) -> List[str]:
        """ Reads a DOCX file using docx2txt and splits it into overlapping chunks of specified size. """
        logger.info(f"Processing file: {file_path}")
        full_text = docx2txt.process(file_path)
        words = full_text.split()
        chunks = []
        start_index = 0

        while start_index < len(words):
            end_index = min(start_index + self.chunk_size, len(words))
            chunk = " ".join(words[start_index:end_index])
            chunks.append(chunk)
            start_index += self.chunk_size - self.overlap

        logger.info(f"Generated {len(chunks)} overlapping chunks from the document.")
        return chunks

    def search_documents_with_llm(self,query: str, llm_pipeline, top_n: int = 5, threshold: float = 0.0):
        logger.info(f"Generating improved query using LLM pipeline...")
        llm_pipeline = llm_pipeline

        improved_query = llm_pipeline.call_local_model(
            prompt=(
                f"You are an expert in telecommunications and document retrieval. "
                f"Given a query that needs to search through technical documents, improve the following query to make it more effective for retrieving the most relevant documents. "
                f"The query might relate to specific technical standards, procedures, or definitions. "
                f"Original query: '{query}' "
                f"Improved query: "
            ),

            temperature=.1,
            max_tokens=500
        )
        logger.info(f"Improved query: {improved_query}")

        results = self.search_documents(improved_query, top_n, threshold)
        return results


if __name__ == '__main__':
    rag = llmRag(db_path='output/db_gte-large')
    #rag.store_documents("data/rel18")
    
    results = rag.search_documents_with_llm("When can a gNB transmit a DL transmission(s) on a channel after initiating a channel occupancy? [3GPP Release 17]", llmPipeline(),top_n=5, threshold=0.1)
    print(results[0])
