import uuid
import os
import docx2txt
import chromadb
from typing import List

def read_docx(file_path: str, chunk_size: int) -> List[str]:
    """ Reads a DOCX file using docx2txt and splits it into chunks of specified size. """
    print(f"Processing file: {file_path}")
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
    print(f"Generated {len(chunks)} chunks from the document.")
    return chunks

def store_documents(folder_path: str, db_path: str, collection_name: str, batch_size=10):
    print(f"Connecting to ChromaDB at {db_path}...")
    db_client = chromadb.PersistentClient(path=db_path)
    collection = db_client.get_or_create_collection(collection_name)
    print(f"Storing documents in collection: {collection_name}")

    for filename in os.listdir(folder_path):
        if filename.endswith('.docx'):
            file_path = os.path.join(folder_path, filename)
            chunks = read_docx(file_path, 200)
            print(f"Starting to store {len(chunks)} chunks from {filename}")

            for i in range(0, len(chunks), batch_size):
                batch_docs = [{'text': chunk, 'metadata': {'filename': filename}} for chunk in chunks[i:i+batch_size]]
                ids = [str(uuid.uuid4()) for _ in batch_docs]
                try:
                    collection.add(documents=[doc['text'] for doc in batch_docs], metadatas=[doc['metadata'] for doc in batch_docs], ids=ids)
                    print(f"Stored batch {i//batch_size + 1} for {filename}")
                except Exception as e:
                    print(f"Failed to store batch {i//batch_size + 1} for {filename}: {str(e)}")
                    break


def search_documents(query: str, db_path: str, collection_name: str, top_n: int = 5):
    """ Searches documents based on a query and returns the top N similar chunks. """
    print(f"Searching for the top {top_n} documents similar to the query: '{query}'")
    db_client = chromadb.PersistentClient(path=db_path)
    collection = db_client.get_or_create_collection(collection_name)
    query_result = collection.query(query_texts=[query], n_results=top_n)

    documents = query_result["documents"][0]
    metadatas = query_result["metadatas"][0]
    results = [(doc, meta['filename']) for doc, meta in zip(documents, metadatas)]
    print(f"Found {len(results)} documents matching the query.")
    return results



if __name__ == '__main__':

    # Example usage
    folder_path = 'rel18'
    db_path = 'db'
    collection_name = 'my_documents'

    # Store documents
    #store_documents(folder_path, db_path, collection_name)

    # Search for relevant chunks
    query = "According to IEEE Std 802.11-2020, when can an HT STA transmit a frame with LDPC coding? [IEEE 802.11]"
    results = search_documents(query, db_path, collection_name)
    for result in results:
        print(f"Document chunk: '{result[0]}' found in file: '{result[1]}'")
