import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma  # Optimized Chroma integration
from langchain.chains import RetrievalQA
from langchain.schema import Document
import tempfile
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain.llms import HuggingFacePipeline
from langchain_ollama.embeddings import OllamaEmbeddings  # Using Mistral for embeddings

# Configuration
DOCUMENT_PATH = "C:\Users\Lenovo\Desktop\Model DistilGPT2-20250224T153649Z-001\Model DistilGPT2\docu.txt"

def load_document(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        exit(1)

# Use Mistral for embeddings
embeddings = OllamaEmbeddings(model="mistral")

# Load DistilGPT-2 model and tokenizer
model_name = 'distilgpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create a text generation pipeline
generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=150,
    temperature=0.7,
    repetition_penalty=1.1,
    pad_token_id=tokenizer.eos_token_id
)

# Initialize LangChain's pipeline wrapper
llm = HuggingFacePipeline(pipeline=generation_pipeline)

# Create a temporary directory for Chroma database
with tempfile.TemporaryDirectory() as temp_dir:
    # Initialize vector database
    vector_db = Chroma(
        collection_name="study-abroad-knowledge",
        embedding_function=embeddings,
        persist_directory=temp_dir
    )

    # Load and process document
    content = load_document(DOCUMENT_PATH)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    doc = Document(page_content=content, metadata={'source': DOCUMENT_PATH})
    chunks = text_splitter.split_documents([doc])

    # Add chunks to vector database
    vector_db.add_documents(chunks)

    # Initialize QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(),
        verbose=False
    )

    # Interactive loop
    try:
        print("\n=== RAG Chatbot ===")
        print(f"Document loaded from: {DOCUMENT_PATH}")
        print("\nType 'exit' to quit.\n")

        while True:
            query = input("Ask a question: ")
            if query.strip().lower() == 'exit':
                print("Exiting...")
                break
            try:
                result = qa_chain.run(query)
                print(f"\nAnswer:\n{result.strip()}\n")
            except Exception as e:
                print(f"Error generating response: {str(e)}")
    except KeyboardInterrupt:
        print("\nScript interrupted by user.")
    except Exception as e:
        print(f"Unrecoverable error: {str(e)}")
