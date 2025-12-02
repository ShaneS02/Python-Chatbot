import os
import numpy as np
from huggingface_hub import InferenceClient
from astradb import vector_search



HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

client = InferenceClient(token=HF_API_KEY)
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLAMA_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

def embed_query(text):
    
    #Get embedding for a given text using Hugging Face Hub API.
    #Returns a list of floats suitable for vector_search.
    
    embedding_array = client.feature_extraction(text, model=EMBEDDING_MODEL)
    embedding_array = np.array(embedding_array)
    embedding_list = embedding_array.flatten().astype(float).tolist()  # ensures list of Python floats

    return embedding_list

def get_relevant_docs(question):
    #Compute embedding for question and retrieve relevant documents from your vector DB.

    embedding = embed_query(question)
    docs = vector_search(embedding)
    return docs



def generate_answer(question, context):
    answer = ""
    prompt = f"""
    You are an AI assistant for animals. You may use the following context 
    to help answer questions, but you do not need to mention it explicitly. 
    Answer naturally and directly. do not start answers with "According to the provided context". 
    Use context silently if it helps. If listing multiple items or points, use bullet points (-) 
    or numbered lists (1., 2., 3.). Don't return images.

    Context:
    {context}

    Question: {question}

    Answer:
    """

    response = client.chat_completion(
        model=LLAMA_MODEL, 
        messages=[
            {"role": "system", "content": "You are an AI assistant for animals."},
            {"role": "user", "content": prompt}
        ])
    
    # Access the text answer from the huggingface hub response
    answer = response["choices"][0]["message"]["content"]        
    
    return answer


def rag_answer(question):
    docs = get_relevant_docs(question)
    context = "\n".join(docs)
    return generate_answer(question, context)