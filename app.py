import os
import streamlit as st
import pickle
import time
import faiss
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer


st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_huggingface.pkl"

main_placeholder = st.empty()

model_name = "facebook/bart-large-cnn"  # You can choose another model available on Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    # create embeddings and save it to FAISS index
    embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embeddings_model.encode([doc.page_content for doc in docs])
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump((index, docs), f)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            index, docs = pickle.load(f)
            embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
            query_embedding = embeddings_model.encode([query])
            D, I = index.search(query_embedding, k=5)  # Get the top 5 results
            results = [docs[i] for i in I[0]]

            # Generate the answer using the summarizer pipeline
            context = " ".join([res.page_content for res in results])
            answer = summarizer(context, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
            st.header("Answer")
            st.write(answer)

            # Display sources
            st.subheader("Sources:")
            for res in results:
                st.write(res.metadata['source'])
