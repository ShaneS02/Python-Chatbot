import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from rag import rag_answer

def main():
    st.set_page_config(page_title="Python RAG Chatbot")
    st.header("Python RAG Chatbot")
    
    user_query = st.text_input("Ask a Question about animals")

    answer = "No Answer"
    if user_query:
        with st.spinner("Thinking..."):
            answer = rag_answer(user_query)
        st.write(answer)

    

if __name__ == '__main__':
    main()