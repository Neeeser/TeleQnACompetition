import uuid
import os
import docx2txt
import chromadb
from typing import List
from loguru import logger

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class llmRag:
    def __init__(self, db_path='output/db', collection_name='my_documents') -> None:
        self.db_path = db_path
        self.collection_name = collection_name
        self.batch_size = 10
        self.chunk_size = 200
        
        self.db_client = chromadb.PersistentClient(path=self.db_path)
        logger.info(f"Connected to ChromaDB at {self.db_path}...")
        self.collection = self.db_client.get_or_create_collection(self.collection_name)
        logger.info(f"Connected to collection: {collection_name}")

    def search_documents(self, query: str, top_n: int = 5, threshold: float = 0.0):
        """ Searches documents based on a query and returns the top N similar chunks. If a threshold is provided, it returns only the chunks with similarity scores above the threshold. """
        logger.info(
            f"Searching for the top {top_n} documents similar to the query: '{query}' with threshold {threshold}")
        query_result = self.collection.query(query_texts=[query], n_results=top_n)

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
                chunks = self.read_docx_helper(file_path, self.chunk_size)
                logger.info(f"Starting to store {len(chunks)} chunks from {filename}")

                for i in range(0, len(chunks), self.batch_size):
                    batch_docs = [{'text': chunk, 'metadata': {'filename': filename}} for chunk in chunks[i:i+self.batch_size]]
                    ids = [str(uuid.uuid4()) for _ in batch_docs]
                    try:
                        self.collection.add(documents=[doc['text'] for doc in batch_docs], metadatas=[doc['metadata'] for doc in batch_docs], ids=ids)
                        logger.info(f"Stored batch {i//self.batch_size + 1} for {filename}")
                    except Exception as e:
                        logger.error(f"Failed to store batch {i//self.batch_size + 1} for {filename}: {str(e)}")
                        break

    def read_docx_helper(self, file_path: str, 
                         chunk_size: int) -> List[str]:
        """ Reads a DOCX file using docx2txt and splits it into chunks of specified size. """
        logger.info(f"Processing file: {file_path}")
        full_text = docx2txt.process(file_path)
        words = full_text.split()
        chunks, current_chunk = [], []
        current_length = 0

        for word in words:
            current_chunk.append(word)
            current_length += 1
            if current_length >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0

        if current_chunk:  # Add the last chunk if any
            chunks.append(" ".join(current_chunk))
        logger.info(f"Generated {len(chunks)} chunks from the document.")
        return chunks