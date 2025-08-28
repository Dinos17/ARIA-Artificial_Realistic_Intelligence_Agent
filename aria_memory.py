# aria_memory.py
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import os
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

USERNAME_DOC_ID = "stored_username_doc"

class ARIAMemory:
    def __init__(self, user_name: str = None):
        self.chat_history = []  # Short-term RAM memory
        self.summary = ""
        self.user_name = user_name

        # Initialize embedding model
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        # Vector memory folder
        self.vector_path = os.path.join(os.getcwd(), "vector_memory")
        os.makedirs(self.vector_path, exist_ok=True)

        # Initialize persistent Chroma vector database
        self.client = chromadb.PersistentClient(path=self.vector_path)
        self.collection = self.client.get_or_create_collection(
            name="user_memory",
            embedding_function=self.embedding_fn
        )

        # Regex for filtering personal identifiers
        if self.user_name:
            self.personal_regex = re.compile(re.escape(self.user_name), re.IGNORECASE)
        else:
            self.personal_regex = None

        # If user_name is not provided, try to load from stored username
        if not self.user_name:
            stored = self.get_stored_username()
            if stored:
                self.user_name = stored
                self.personal_regex = re.compile(re.escape(self.user_name), re.IGNORECASE)

    # ----------------------------
    # Short-term + Long-term memory
    # ----------------------------
    def update(self, user_input: str, response: str):
        """Update short-term RAM memory and persistent vector memory."""
        # Short-term memory
        self.chat_history.append({"role": "user", "content": user_input})
        self.chat_history.append({"role": "assistant", "content": response})

        # Long-term memory (vector)
        self.collection.add(
            documents=[user_input, response],
            metadatas=[{"role": "user"}, {"role": "assistant"}],
            ids=[str(len(self.chat_history)-2), str(len(self.chat_history)-1)]
        )

    def get_context(self, top_k: int = 5, filter_personal: bool = True):
        """
        Retrieve combined context for LLM:
        - Short-term RAM memory
        - Long-term vector memory (filtered)
        """
        # Short-term context
        short_term = self.chat_history[-top_k:] if self.chat_history else []

        # Long-term context
        long_term = []
        if os.path.exists(self.vector_path):
            query_text = " ".join([msg["content"] for msg in short_term])
            results = self.collection.query(query_texts=[query_text], n_results=top_k)
            for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                content = doc
                if filter_personal and self.personal_regex:
                    content = self.personal_regex.sub("[REDACTED]", content)
                long_term.append({"role": meta['role'], "content": content})

        # Combine short-term and filtered long-term context
        context = short_term + long_term
        return {"chat_history": context, "summary": self.summary}

    def update_summary(self, summary_text: str):
        """Optional session summary."""
        self.summary = summary_text

    # ----------------------------
    # Persistent username storage
    # ----------------------------
    def store_username(self, username: str):
        """Store username in vector memory for future sessions."""
        self.collection.add(
            documents=[username],
            metadatas=[{"role": "system", "type": "username"}],
            ids=[USERNAME_DOC_ID]
        )

    def get_stored_username(self):
        """Retrieve stored username if available."""
        try:
            result = self.collection.get(ids=[USERNAME_DOC_ID])
            if result and result['documents']:
                stored = result['documents'][0]
                if isinstance(stored, list):
                    return stored[0]  # Full username string
                return stored
        except Exception:
            return None
        return None

    def clear_stored_username(self):
        """Delete stored username from vector memory."""
        try:
            self.collection.delete(ids=[USERNAME_DOC_ID])
        except Exception:
            pass
