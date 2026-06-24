import os
import logging
from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

def chunk_text(text: str, chunk_size: int = 250, overlap: int = 50) -> List[str]:
    """
    Splits text into overlapping chunks of words as a proxy for tokens.
    
    Args:
        text: The raw input text.
        chunk_size: Number of words per chunk (~250 words is approx 300 tokens).
        overlap: Number of words to overlap between consecutive chunks.
        
    Returns:
        A list of text chunks.
    """
    words = text.split()
    chunks = []
    
    if not words:
        return chunks
        
    i = 0
    while i < len(words):
        # Create chunk
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))
        
        # Break if we reached the end of the text
        if i + chunk_size >= len(words):
            break
            
        # Move forward, respecting the overlap
        i += (chunk_size - overlap)
        
    return chunks


class ContractRAGEngine:
    """
    RAG (Retrieval-Augmented Generation) Engine for indexing and searching
    the master medical billing legal contract using ChromaDB and Sentence-Transformers.
    """
    def __init__(self, persist_dir: str = "./chroma_db", collection_name: str = "contract_clauses"):
        """
        Initializes the ChromaDB persistent client and sets up the embedding function.
        """
        self.persist_dir = os.path.abspath(persist_dir)
        self.collection_name = collection_name
        
        # Ensure the directory exists
        os.makedirs(self.persist_dir, exist_ok=True)
        
        # Initialize persistent ChromaDB client
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        
        # Set up sentence-transformers embedding function (all-MiniLM-L6-v2)
        # ChromaDB downloads and caches the model automatically
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Retrieve or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function
        )

    def is_indexed(self) -> bool:
        """
        Checks if the contract is already indexed in the collection.
        """
        try:
            return self.collection.count() > 0
        except Exception as e:
            logger.error(f"Error checking index status: {str(e)}")
            return False

    def index_contract(self, text: str, force: bool = False) -> int:
        """
        Chunks and indexes the contract text into ChromaDB.
        
        Args:
            text: Raw text of the legal contract.
            force: If True, deletes any existing collection and rebuilds it.
            
        Returns:
            The number of chunks indexed.
        """
        if not text.strip():
            raise ValueError("Contract text is empty. Cannot index.")

        # Check if already indexed
        if self.is_indexed() and not force:
            logger.info("Contract is already indexed. Skipping re-indexing.")
            return self.collection.count()

        try:
            if force:
                logger.info(f"Forcing re-index. Deleting collection: {self.collection_name}")
                try:
                    self.client.delete_collection(self.collection_name)
                except Exception:
                    pass  # If it doesn't exist, ignore
                
                # Re-create collection
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function
                )

            # Chunk the contract text
            chunks = chunk_text(text, chunk_size=250, overlap=50)
            logger.info(f"Split contract into {len(chunks)} chunks.")

            # Prepare inputs for collection insertion
            ids = [f"clause_{i}" for i in range(len(chunks))]
            metadatas = [{"source": "legal_contract", "chunk_index": i} for i in range(len(chunks))]

            # Insert chunks in batches (to handle large files safely)
            batch_size = 100
            for k in range(0, len(chunks), batch_size):
                end = min(k + batch_size, len(chunks))
                self.collection.add(
                    documents=chunks[k:end],
                    ids=ids[k:end],
                    metadatas=metadatas[k:end]
                )

            logger.info(f"Successfully indexed {len(chunks)} contract clauses in ChromaDB.")
            return len(chunks)

        except Exception as e:
            logger.error(f"Failed to index contract in ChromaDB: {str(e)}")
            raise RuntimeError(f"ChromaDB indexing failed: {str(e)}") from e

    def search_contract(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        Searches the indexed contract for clauses matching the query.

        Args:
            query: Semantic search query.
            n_results: Number of top matches to retrieve (default: 3).

        Returns:
            A list of dictionaries containing:
                - 'text': Content of the clause
                - 'id': ChromaDB identifier
                - 'distance': Distance score (lower is more similar)
                - 'metadata': Context metadata dictionary
        """
        if not self.is_indexed():
            logger.warning("Search query run on empty/unindexed collection.")
            return []

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            formatted_results = []
            if results and "documents" in results and results["documents"]:
                docs = results["documents"][0]
                ids = results["ids"][0]
                metadatas = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)
                distances = results["distances"][0] if "distances" in results else [0.0] * len(docs)

                for doc, id_, meta, dist in zip(docs, ids, metadatas, distances):
                    formatted_results.append({
                        "text": doc,
                        "id": id_,
                        "metadata": meta,
                        "distance": dist
                    })
            
            return formatted_results

        except Exception as e:
            logger.error(f"Failed to query ChromaDB collection: {str(e)}")
            raise RuntimeError(f"ChromaDB search failed: {str(e)}") from e
