# Import necessary libraries
import pandas as pd
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS as LangchainFAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain import HuggingFacePipeline
import torch
import os
from docx import Document
import PyPDF2
import textract
import re

# Set display options
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)

# General function to load files
def load_files(file_paths):
    text_data = []
    for file_path in file_paths:
        if not os.path.isfile(file_path):
            print(f"File not found: {file_path}")
            continue
        extension = os.path.splitext(file_path)[1].lower()
        try:
            if extension == '.csv':
                # Load CSV
                df = pd.read_csv(file_path)
                text = df.astype(str).agg(' '.join, axis=1).tolist()
                text_data.extend(text)
            elif extension == '.txt':
                # Load txt
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    text_data.append(text)
            elif extension == '.pdf':
                # Load PDF
                text = ''
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        text += page.extract_text()
                text_data.append(text)
            elif extension == '.docx':
                # Load DOCX
                doc = Document(file_path)
                fullText = [para.text for para in doc.paragraphs]
                text = '\n'.join(fullText)
                text_data.append(text)
            elif extension == '.doc':
                # Load DOC
                text = textract.process(file_path).decode('utf-8')
                text_data.append(text)
            else:
                print(f"Unsupported file type: {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    return text_data

def extract_controls(text):
    """Extract control-related information from text."""
    control_patterns = [
        r'control\s*:\s*([^.!?\n]+)',
        r'mitigation\s*:\s*([^.!?\n]+)',
        r'safeguard\s*:\s*([^.!?\n]+)',
        r'security measure\s*:\s*([^.!?\n]+)',
    ]
    controls = []
    for pattern in control_patterns:
        controls.extend(re.findall(pattern, text, re.IGNORECASE))
    return controls

# Split text function
def split_text(text_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.create_documents(text_data)
    
    # Extract and add control information to chunk metadata
    for chunk in chunks:
        chunk.metadata['controls'] = extract_controls(chunk.page_content)
    
    return chunks

# Create vector store function
def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = LangchainFAISS.from_documents(chunks, embeddings)
    return vectorstore

# Load LLM function
def load_llm():
    model_id = "meta-llama/Llama-2-13b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        quantization_config=nf4_config,
        device_map="auto"
    )

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, temperature=0.1)
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

# Create re-ranker function
def create_reranker(llm):
    compressor = LLMChainExtractor.from_llm(llm)
    return compressor

# Get conversation chain function
def get_conversation_chain(vectorstore, llm, reranker):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    prompt_template = """You are an AI assistant specializing in encryption and cybersecurity risk findings, with a focus on control measures. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know—don't try to make up an answer.

    Context:
    {context}

    Question: {question}

    Provide a detailed answer, including:
    1. Relevant control measures identified in the context
    2. An assessment of the effectiveness of these controls
    3. Recommendations for additional or improved controls if necessary
    4. Any potential gaps in the control framework

    Detailed answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=vectorstore.as_retriever(search_kwargs={"k": 5})
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=compression_retriever, 
        memory=memory, 
        prompt=PROMPT,
        return_source_documents=True  # This will return the source documents for each answer
    )
    return conversation_chain

# Handle user input function
def handle_userinput(user_question, conversation):
    response = conversation({'question': user_question})
    chat_history = response['chat_history']
    source_documents = response['source_documents']
    
    for i, message in enumerate(chat_history):
        if i % 2 == 0:
            print(f"User: {message.content}")
        else:
            print(f"Control Bot: {message.content}")
            
    print("\nSource Documents:")
    for doc in source_documents:
        print(f"- {doc.metadata.get('source', 'Unknown source')}")
        if 'controls' in doc.metadata:
            print("  Controls mentioned:")
            for control in doc.metadata['controls']:
                print(f"    - {control}")
    print("\n")

def analyze_control_coverage(vectorstore):
    """Analyze the coverage of controls in the dataset."""
    all_controls = []
    for doc in vectorstore.docstore._dict.values():
        all_controls.extend(doc.metadata.get('controls', []))
    
    control_counts = pd.Series(all_controls).value_counts()
    print("Control Coverage Analysis:")
    print(control_counts)
    print("\nTotal unique controls:", len(control_counts))
    print("Top 10 most mentioned controls:")
    print(control_counts.head(10))

# Main function
def main():
    print("Executing RAG system for Cybersecurity Control Analysis...")
    # Directory containing data files
    data_dir = 'data'
    # Supported file extensions
    supported_extensions = ['.csv', '.txt', '.pdf', '.docx', '.doc']
    # Get list of all files in data_dir with supported extensions
    file_paths = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in supported_extensions:
                file_paths.append(os.path.join(root, file))
    # Load data from all files
    text_data = load_files(file_paths)
    # Split and process text
    chunks = split_text(text_data)
    vectorstore = create_vectorstore(chunks)
    llm = load_llm()
    reranker = create_reranker(llm)
    conversation = get_conversation_chain(vectorstore, llm, reranker)

    # Analyze control coverage
    analyze_control_coverage(vectorstore)

    print("RAG System Ready. Type 'exit' to quit.")
    while True:
        user_question = input("Ask a question about cybersecurity risks and controls (or type 'exit' to quit): ")
        if user_question.lower() == 'exit':
            break
        handle_userinput(user_question, conversation)

if __name__ == "__main__":
    main()
