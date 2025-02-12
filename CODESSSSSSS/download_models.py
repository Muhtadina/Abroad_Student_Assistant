from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Save retriever model
retriever_model = SentenceTransformer('all-MiniLM-L6-v2')
retriever_model.save('retriever_model')  # Saves to a folder named "retriever_model"

# Save generator model
generator_model = GPT2LMHeadModel.from_pretrained('distilgpt2')
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
generator_model.save_pretrained('generator_model')
tokenizer.save_pretrained('generator_model')