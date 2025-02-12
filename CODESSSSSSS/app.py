from flask import Flask, render_template, request, jsonify
import json
import numpy as np
import faiss
import random
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ======================
# Configuration
# ======================
MODEL_SETTINGS = {
    "retriever": "all-MiniLM-L6-v2",
    "generator": "distilgpt2",
    "max_context_length": 200,
    "max_new_tokens": 80,
    "temperature": 0.5,
    "top_k": 2
}

CONVERSATION_PROMPTS = {
    "greetings": {
        "inputs": ["hi", "hello", "hey", "good morning", "good afternoon"],
        "responses": [
            "Hello! How can I assist you with study abroad information today?",
            "Hi there! Ready to explore study abroad options?",
            "Hey! Ask me anything about studying in Australia, Canada, or the UK!"
        ]
    },
    "farewell": {
        "inputs": ["bye", "goodbye", "exit", "quit", "see you"],
        "responses": [
            "Goodbye! Safe travels in your study abroad journey!",
            "Farewell! Feel free to come back with more questions!",
            "Have a great day! Good luck with your applications!"
        ]
    },
    "thanks": {
        "inputs": ["thanks", "thank you", "appreciate it"],
        "responses": [
            "You're welcome! Happy to help!",
            "My pleasure! Let me know if you need anything else!",
            "Glad I could assist! 😊"
        ]
    },
    "casual": {
        "inputs": ["how are you", "what's up", "how's it going"],
        "responses": [
            "I'm doing well, thanks! Excited to help with your study plans!",
            "All systems go! Ready to answer your study abroad questions!",
            "I'm here and ready to assist! What can I help you with today?"
        ]
    }
}

# ======================
# Load Knowledge Base
# ======================
def load_knowledge_base(file_path: str):
    """Load knowledge base from a .txt or .json file"""
    if file_path.endswith(".json"):
        with open(file_path, "r") as f:
            return json.load(f)
    elif file_path.endswith(".txt"):
        with open(file_path, "r") as f:
            # Treat each line as a separate document
            return [{"id": i + 1, "text": line.strip()} for i, line in enumerate(f) if line.strip()]
    else:
        raise ValueError("Unsupported file format. Use .txt or .json.")

# ======================
# Chatbot Class
# ======================
class AbroadStudyAssistant:
    def __init__(self, knowledge_base):
        # Initialize components
        self.retriever = SentenceTransformer(MODEL_SETTINGS["retriever"])
        self.generator = GPT2LMHeadModel.from_pretrained(MODEL_SETTINGS["generator"])
        self.tokenizer = GPT2Tokenizer.from_pretrained(MODEL_SETTINGS["generator"])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Prepare FAISS index
        texts = [doc['text'] for doc in knowledge_base]
        embeddings = self.retriever.encode(texts)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings))

    def detect_intent(self, query: str) -> str:
        """Detect conversation intent"""
        query_lower = query.lower()
        for intent, data in CONVERSATION_PROMPTS.items():
            if any(keyword in query_lower for keyword in data["inputs"]):
                return intent
        return "study_query"

    def get_canned_response(self, intent: str) -> str:
        """Get pre-defined response"""
        return random.choice(CONVERSATION_PROMPTS[intent]["responses"])

    def retrieve_info(self, query: str) -> list:
        """Retrieve relevant study information"""
        query_embedding = self.retriever.encode([query])
        _, indices = self.index.search(query_embedding, MODEL_SETTINGS["top_k"])
        return [
            KNOWLEDGE_BASE[i]['text'][:MODEL_SETTINGS["max_context_length"]]
            for i in indices[0]
        ]

    def generate_response(self, query: str, contexts: list) -> str:
        """Generate friendly study-related response"""
        prompt = (
            "You're a friendly study abroad assistant. Answer naturally using this context:\n"
            f"Context: {' '.join(contexts)}\n"
            f"Question: {query}\n"
            "Friendly Answer:"
        )
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        
        outputs = self.generator.generate(
            **inputs,
            max_new_tokens=MODEL_SETTINGS["max_new_tokens"],
            temperature=MODEL_SETTINGS["temperature"],
            repetition_penalty=1.5,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True
        )
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full_response.split("Friendly Answer:")[-1].strip()

    def chat(self, query: str) -> str:
        """Handle complete conversation flow"""
        intent = self.detect_intent(query)
        
        if intent != "study_query":
            return self.get_canned_response(intent)
        
        contexts = self.retrieve_info(query)
        if not contexts:
            return "I'm not sure about that, but I can help with questions about studying in Australia, Canada, or the UK!"
            
        try:
            return self.generate_response(query, contexts)
        except Exception as e:
            return f"Oops! I encountered an error: {str(e)}"

# ======================
# Flask App
# ======================
app = Flask(__name__)

# Load knowledge base
KNOWLEDGE_BASE = load_knowledge_base("knowledge_base.txt")

# Initialize chatbot
assistant = AbroadStudyAssistant(KNOWLEDGE_BASE)

@app.route("/")
def home():
    """Render the chatbot interface"""
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """Handle chatbot requests"""
    user_input = request.json.get("message")
    response = assistant.chat(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)