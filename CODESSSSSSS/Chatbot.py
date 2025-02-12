# Step 1: Install dependencies (run once)
# pip install torch transformers sentence-transformers faiss-cpu

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
    "max_context_length": 200,  # Truncate context to avoid overload
    "max_new_tokens": 80,       # Shorter responses
    "temperature": 0.5,         # Balanced randomness
    "top_k": 2                  # Retrieve top 2 most relevant docs
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
# Knowledge Base
# ======================
KNOWLEDGE_BASE = [
    {
        "id": 1,
        "text": "Study Abroad Entry Requirements for Australia: - Bachelor's Degree: Good grades in 9-12th grade. - English Proficiency Requirements: IELTS: 6.5 or above, PTE: 42 minimum, TOEFL: 90+. - Master's Degree: Bachelor's Degree (minimum three years), GMAT: 580-600+. - Visa Requirements: Ensure passport validity throughout the study period, provide proof of financial resources, etc."
    },
    {
        "id": 2,
        "text": "Study Abroad Entry Requirements for Canada: - Bachelor's Degree/Diploma/Certificate: Good grades in 9-12th grade. - English Proficiency Requirements: IELTS: 6.5, PTE: 42 minimum, TOEFL: 110. - Master's Degree/Postgraduate: Bachelor's Degree (four years), GRE: 300-320+. - Visa Requirements: Proof of financial resources, academic transcripts, etc."
    },
    {
        "id": 3,
        "text": "Study Abroad Entry Requirements for the United Kingdom (UK): - Eligibility Requirements: Official acceptance letter from a recognized UK educational institution. - Proof of English language proficiency (IELTS or SELT scores). - Documentation verifying financial capability to cover tuition and living costs. - Fulfillment of specific admission criteria set by the chosen institution."
    }
]

# ======================
# Chatbot Class
# ======================
class StudyBuddy:
    def __init__(self):
        # Initialize components
        self.retriever = SentenceTransformer(MODEL_SETTINGS["retriever"])
        self.generator = GPT2LMHeadModel.from_pretrained(MODEL_SETTINGS["generator"])
        self.tokenizer = GPT2Tokenizer.from_pretrained(MODEL_SETTINGS["generator"])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Prepare FAISS index
        texts = [doc['text'] for doc in KNOWLEDGE_BASE]
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
# Run Chatbot
# ======================
if __name__ == "__main__":
    buddy = StudyBuddy()
    print("🌟 StudyBuddy: Hi! I'm here to help with study abroad questions. Type 'exit' to quit.")
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("🌟 StudyBuddy: Good luck with your studies! 🎓")
                break
                
            response = buddy.chat(user_input)
            print(f"🌟 StudyBuddy: {response}")
            
        except KeyboardInterrupt:
            print("\n🌟 StudyBuddy: Goodbye! 👋")
            break