    import sys
    import io
    import os
    import json
    import numpy as np
    import faiss
    import random
    from sentence_transformers import SentenceTransformer
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    MODEL_SETTINGS = {
        "retriever": "all-MiniLM-L6-v2",
        "generator": "distilgpt2",
        "max_context_length": 200, 
        "max_new_tokens": 80,       
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

    class Abroad_Study_Assistant:
        def __init__(self, knowledge_file="docu.txt"):
            # Load knowledge base from file
            self.knowledge_file = knowledge_file
            self.KNOWLEDGE_BASE = self.load_knowledge_base()

            # Initialize components
            self.retriever = SentenceTransformer(MODEL_SETTINGS["retriever"])
            self.generator = GPT2LMHeadModel.from_pretrained(MODEL_SETTINGS["generator"])
            self.tokenizer = GPT2Tokenizer.from_pretrained(MODEL_SETTINGS["generator"])
            self.tokenizer.pad_token = self.tokenizer.eos_token

            # Prepare FAISS index
            texts = [doc["text"] for doc in self.KNOWLEDGE_BASE]
            embeddings = self.retriever.encode(texts)
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(np.array(embeddings))

        def load_knowledge_base(self):
            if not os.path.exists(self.knowledge_file):
                print(f"Warning: {self.knowledge_file} not found! Using an empty knowledge base.")
                return []

            knowledge_base = []
            with open(self.knowledge_file, "r", encoding="utf-8") as file:
                content = file.read().strip()
                entries = content.split("\n---\n")  # Splitting each entry using "---"
                
                for idx, entry in enumerate(entries):
                    knowledge_base.append({"id": idx + 1, "text": entry.strip()})
            
            print(f"Loaded {len(knowledge_base)} knowledge entries from {self.knowledge_file}.")
            return knowledge_base

        def detect_intent(self, query: str) -> str:
            query_lower = query.lower()
            for intent, data in CONVERSATION_PROMPTS.items():
                if any(keyword in query_lower for keyword in data["inputs"]):
                    return intent
            return "study_query"

        def get_canned_response(self, intent: str) -> str:
            return random.choice(CONVERSATION_PROMPTS[intent]["responses"])

        def retrieve_info(self, query: str) -> list:
            query_embedding = self.retriever.encode([query])
            _, indices = self.index.search(query_embedding, MODEL_SETTINGS["top_k"])
            return [
                self.KNOWLEDGE_BASE[i]['text'][:MODEL_SETTINGS["max_context_length"]]
                for i in indices[0]
            ]

        def generate_response(self, query: str, contexts: list) -> str:
            prompt = (
                "You're a friendly study abroad assistant is here:\n"
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
                temperature=1.0,
                repetition_penalty=1.5,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True
            )
            
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return full_response.split("Friendly Answer:")[-1].strip()

        def chat(self, query: str) -> str:
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


    if __name__ == "__main__":
        buddy = Abroad_Study_Assistant()
        print(" Abroad Study Assistant: Hi! I'm here to help with study abroad questions. Type 'exit' to quit.")
        
        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() in ["exit", "quit"]:
                    print(" Abroad Study Assistant: Good luck with your studies! 🎓")
                    break
                    
                response = buddy.chat(user_input)
                print(f" Abroad Study Assistant: {response}")
                
            except KeyboardInterrupt:
                print("\n Abroad Study Assistant: Goodbye! ")
                break