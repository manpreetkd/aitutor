from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os
import uuid

class KnowledgeRAG:
    def __init__(self):
        """Initialize LLM and embedding model."""
        self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

    def get_embedding_function(self):
        """Initialize and return HuggingFace embeddings."""
        return HuggingFaceEmbeddings(model_name=self.embedding_model)

    def process_pdf(self, pdf_path):
        """Load and split a PDF into smaller text chunks."""
        try:
            # Load PDF content
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()

            # Split text into chunks with overlap for better retrieval
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " "]
            )
            chunks = text_splitter.split_documents(pages)
            return chunks
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {str(e)}")
            return []

    def create_vectorstore(self, chunks, vectorstore_path):
        """Create or update a Chroma vectorstore with document chunks."""
        try:
            # Initialize embedding function
            embedding_function = self.get_embedding_function()

            # Generate unique IDs for chunks
            unique_ids = set()
            unique_chunks = []
            for doc in chunks:
                doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content))
                if doc_id not in unique_ids:
                    unique_ids.add(doc_id)
                    unique_chunks.append(doc)

            # Create or load Chroma vectorstore with persistence
            vectorstore = Chroma.from_documents(
                documents=unique_chunks,
                ids=list(unique_ids),
                embedding=embedding_function,
                persist_directory=vectorstore_path
            )

            return vectorstore
        except Exception as e:
            print(f"Error creating vectorstore: {str(e)}")
            return None

    def get_relevant_context(self, question, vectorstore_path):
        """Retrieve relevant document context based on the user's question."""
        try:
            # Initialize embedding function
            embedding_function = self.get_embedding_function()

            # Load existing vectorstore
            vectorstore = Chroma(
                persist_directory=vectorstore_path,
                embedding_function=embedding_function
            )

            # Retrieve most relevant chunks
            retriever = vectorstore.as_retriever(search_type="similarity")
            relevant_chunks = retriever.invoke(question)

            # Combine retrieved chunks into a meaningful context
            context = "\n-------------\n".join([doc.page_content for doc in relevant_chunks])

            return context if context else "No relevant information found in the documents."
        except Exception as e:
            print(f"Error retrieving context: {str(e)}")
            return "Error retrieving document context."

    def setup_knowledge_base(self, pdf_path, vectorstore_path):
        """Setup a vector database for document storage and retrieval."""
        try:
            # Extract chunks from PDF
            chunks = self.process_pdf(pdf_path)
            if not chunks:
                print(f"No valid content extracted from {pdf_path}.")
                return False

            # Create vectorstore
            self.create_vectorstore(chunks, vectorstore_path)
            print(f"Knowledge base setup complete for {pdf_path}.")
            return True
        except Exception as e:
            print(f"Error setting up knowledge base: {str(e)}")
            return False

if __name__ == "__main__":
    # Test the RAG implementation
    rag = KnowledgeRAG()
    pdf_path = "knowledge/Lesson-01.pdf"  # Ensure the PDF exists
    vectorstore_path = "vectorstore"  # Define a persistent storage directory
    question = "What are financial statements?"

    # Setup knowledge base
    success = rag.setup_knowledge_base(pdf_path, vectorstore_path)

    # Retrieve context for a sample question
    if success:
        context = rag.get_relevant_context(question, vectorstore_path)
        print("Retrieved context:", context)
