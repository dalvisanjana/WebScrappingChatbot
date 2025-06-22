import streamlit as st
import requests
from bs4 import BeautifulSoup
import os

from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub

# Set HuggingFace API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "Your_hugging token"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def scrape_website(url):
    st.write(f"🔗 Scraping: {url}")
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(separator="\n")
        st.success("✅ Scrape complete.")
        return [Document(page_content=text)]
    except Exception as e:
        st.error(f"❌ Failed to scrape website: {e}")
        return []


def build_qa_chain(documents):
    try:
        st.write("🔧 Chunking text...")
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = splitter.split_documents(documents)
        st.write(f"📄 Chunks created: {len(docs)}")

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectordb = FAISS.from_documents(docs, embeddings)

        # ✅ Use a text2text-generation model that works with Inference API
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-base",
            model_kwargs={"temperature": 0.5, "max_length": 512}
        )

        chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())
        return chain
    except Exception as e:
        st.error(f"❌ Error building QA chain: {e}")
        return None


def main():
    st.set_page_config(page_title="🕷 Web Scraping Q&A Bot", layout="centered")
    st.title("🕷 Web Scraping Q&A Bot")

    url = st.text_input("Enter website URL:", placeholder="https://www.python.org/about/")
    qa_chain = None

    if url:
        with st.spinner("⏳ Scraping and processing..."):
            documents = scrape_website(url)
            if not documents:
                return
            qa_chain = build_qa_chain(documents)

    if qa_chain:
        question = st.text_input("Ask about this page:")
        if question:
            with st.spinner("💭 Thinking..."):
                try:
                    answer = qa_chain.run(question)
                    st.success("💬 Answer:")
                    st.write(answer)
                except Exception as e:
                    st.error(f"❌ Failed to generate answer:\n\n{e}")


if __name__ == "__main__":
    main()
