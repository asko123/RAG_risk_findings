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

# Static data dumps with focus on access control and remediation
ACCESS_CONTROL_PRINCIPLES = """
1. Principle of Least Privilege (PoLP): Users should be given the minimum levels of access needed to perform their job functions.
2. Separation of Duties (SoD): Critical tasks should be divided among multiple people to prevent fraud and errors.
3. Need-to-Know: Access to information should be limited to those who need it for their job functions.
4. Defense in Depth: Multiple layers of security controls should be used to protect resources.
5. Zero Trust: Trust nothing and verify everything, regardless of whether it's inside or outside the network perimeter.
6. Role-Based Access Control (RBAC): Access rights are assigned based on roles users have within an organization.
7. Attribute-Based Access Control (ABAC): Access decisions are based on attributes associated with users, resources, and environmental conditions.
8. Mandatory Access Control (MAC): Access is controlled by the operating system based on security labels.
9. Discretionary Access Control (DAC): The owner of a resource specifies who can access it and what privileges they have.
10. Time-based Access Control: Access is granted or restricted based on time of day or day of the week.
"""

OWASP_ACCESS_CONTROL = """
A01:2021-Broken Access Control
Common access control vulnerabilities:
1. Violation of the principle of least privilege or deny by default
2. Bypassing access control checks by modifying the URL or HTML page
3. Permitting viewing or editing someone else's account by providing its unique identifier
4. Accessing API with missing access controls for POST, PUT and DELETE
5. Elevation of privilege through metadata manipulation (e.g., JWT)
6. CORS misconfiguration allowing unauthorized API access
7. Force browsing to authenticated pages as an unauthenticated user or to privileged pages as a standard user
"""

NIST_ACCESS_CONTROL = """
NIST SP 800-53 Access Control (AC) Family:
AC-1: Access Control Policy and Procedures
AC-2: Account Management
AC-3: Access Enforcement
AC-4: Information Flow Enforcement
AC-5: Separation of Duties
AC-6: Least Privilege
AC-7: Unsuccessful Logon Attempts
AC-8: System Use Notification
AC-9: Previous Logon Notification
AC-10: Concurrent Session Control
AC-11: Session Lock
AC-12: Session Termination
AC-14: Permitted Actions without Identification or Authentication
AC-17: Remote Access
AC-18: Wireless Access
AC-19: Access Control for Mobile Devices
AC-20: Use of External Information Systems
AC-21: Information Sharing
AC-22: Publicly Accessible Content
AC-23: Data Mining Protection
AC-24: Access Control Decisions
AC-25: Reference Monitor
"""

ACCESS_CONTROL_REMEDIATION = """
Common Access Control Remediation Strategies:
1. Implement and enforce the principle of least privilege (PoLP).
   - Regularly review and update user access rights.
   - Use role-based access control (RBAC) to manage permissions.

2. Strengthen authentication mechanisms.
   - Implement multi-factor authentication (MFA) for all user accounts.
   - Use strong password policies and consider password managers.

3. Improve authorization processes.
   - Implement proper session management techniques.
   - Use token-based authentication for APIs (e.g., JWT with proper signing).

4. Enhance access control checks.
   - Implement server-side validation for all access control checks.
   - Use parameterized queries to prevent SQL injection attacks.

5. Secure API endpoints.
   - Implement proper access controls for all HTTP methods (GET, POST, PUT, DELETE).
   - Use API gateways to centralize access control enforcement.

6. Address CORS (Cross-Origin Resource Sharing) issues.
   - Configure CORS policies correctly to restrict access to trusted domains only.

7. Implement proper error handling and logging.
   - Avoid exposing sensitive information in error messages.
   - Implement comprehensive logging and monitoring for access control events.

8. Conduct regular security audits and penetration testing.
   - Perform automated and manual testing to identify access control weaknesses.
   - Conduct code reviews focusing on access control implementation.

9. Implement Zero Trust Architecture.
   - Verify every access request as if it originates from an untrusted network.
   - Use micro-segmentation to limit lateral movement within the network.

10. Educate developers and users.
    - Provide regular training on secure coding practices and access control best practices.
    - Raise awareness about the importance of access control among all users.
"""

def load_files(file_paths):
    text_data = []
    for file_path in file_paths:
        if not os.path.isfile(file_path):
            print(f"File not found: {file_path}")
            continue
        extension = os.path.splitext(file_path)[1].lower()
        try:
            if extension == '.csv':
                df = pd.read_csv(file_path)
                text = df.astype(str).agg(' '.join, axis=1).tolist()
                text_data.extend(text)
            elif extension == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    text_data.append(text)
            elif extension == '.pdf':
                text = ''
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        text += page.extract_text()
                text_data.append(text)
            elif extension == '.docx':
                doc = Document(file_path)
                fullText = [para.text for para in doc.paragraphs]
                text = '\n'.join(fullText)
                text_data.append(text)
            elif extension == '.doc':
                text = textract.process(file_path).decode('utf-8')
                text_data.append(text)
            else:
                print(f"Unsupported file type: {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    return text_data

def extract_access_control_info(text):
    """Extract access control-related information and remediation strategies from text."""
    access_control_patterns = [
        r'access control\s*:\s*([^.!?\n]+)',
        r'authentication\s*:\s*([^.!?\n]+)',
        r'authorization\s*:\s*([^.!?\n]+)',
        r'principle of least privilege\s*:\s*([^.!?\n]+)',
        r'separation of duties\s*:\s*([^.!?\n]+)',
        r'role-based access control\s*:\s*([^.!?\n]+)',
        r'RBAC\s*:\s*([^.!?\n]+)',
        r'mandatory access control\s*:\s*([^.!?\n]+)',
        r'MAC\s*:\s*([^.!?\n]+)',
        r'discretionary access control\s*:\s*([^.!?\n]+)',
        r'DAC\s*:\s*([^.!?\n]+)',
        r'zero trust\s*:\s*([^.!?\n]+)',
        r'multi-factor authentication\s*:\s*([^.!?\n]+)',
        r'MFA\s*:\s*([^.!?\n]+)',
        r'single sign-on\s*:\s*([^.!?\n]+)',
        r'SSO\s*:\s*([^.!?\n]+)',
        r'AC-\d+\s*:\s*([^.!?\n]+)',  # For NIST SP 800-53 AC controls
        r'A01:2021\s*-\s*([^.!?\n]+)',  # For OWASP Broken Access Control
        r'remediation\s*:\s*([^.!?\n]+)',  # For remediation strategies
        r'mitigation\s*:\s*([^.!?\n]+)',  # Alternative term for remediation
    ]
    access_control_info = []
    for pattern in access_control_patterns:
        access_control_info.extend(re.findall(pattern, text, re.IGNORECASE))
    return access_control_info

def split_text(text_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.create_documents(text_data)
    
    for chunk in chunks:
        chunk.metadata['access_control_info'] = extract_access_control_info(chunk.page_content)
    
    return chunks

def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = LangchainFAISS.from_documents(chunks, embeddings)
    return vectorstore

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

def create_reranker(llm):
    compressor = LLMChainExtractor.from_llm(llm)
    return compressor

def get_conversation_chain(vectorstore, llm, reranker):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    prompt_template = """You are an AI assistant specializing in cybersecurity with a focus on access control and remediation strategies. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know—don't try to make up an answer.

    Context:
    {context}

    Question: {question}

    Provide a detailed answer, including:
    1. Relevant access control measures identified in the context
    2. An assessment of the effectiveness of these access control measures
    3. Specific vulnerabilities or gaps in the access control framework
    4. Detailed remediation strategies for identified issues
    5. How the identified access control measures and remediation strategies align with industry standards (e.g., NIST SP 800-53, OWASP Top 10)
    6. Best practices for implementing and maintaining strong access control
    7. Any additional recommendations for improving overall access control posture

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
        return_source_documents=True
    )
    return conversation_chain

def handle_userinput(user_question, conversation):
    response = conversation({'question': user_question})
    chat_history = response['chat_history']
    source_documents = response['source_documents']
    
    for i, message in enumerate(chat_history):
        if i % 2 == 0:
            print(f"User: {message.content}")
        else:
            print(f"Access Control & Remediation Bot: {message.content}")
            
    print("\nSource Documents:")
    for doc in source_documents:
        print(f"- {doc.metadata.get('source', 'Unknown source')}")
        if 'access_control_info' in doc.metadata:
            print("  Access Control and Remediation Information:")
            for info in doc.metadata['access_control_info']:
                print(f"    - {info}")
    print("\n")

def analyze_access_control_coverage(vectorstore):
    """Analyze the coverage of access control measures and remediation strategies in the dataset."""
    all_access_control_info = []
    for doc in vectorstore.docstore._dict.values():
        all_access_control_info.extend(doc.metadata.get('access_control_info', []))
    
    access_control_counts = pd.Series(all_access_control_info).value_counts()
    print("Access Control and Remediation Coverage Analysis:")
    print(access_control_counts)
    print("\nTotal unique access control measures and remediation strategies:", len(access_control_counts))
    print("Top 10 most mentioned access control measures and remediation strategies:")
    print(access_control_counts.head(10))

def main():
    print("Executing RAG system for Access Control and Remediation Analysis...")
    
    # Add static data
    text_data = [ACCESS_CONTROL_PRINCIPLES, OWASP_ACCESS_CONTROL, NIST_ACCESS_CONTROL, ACCESS_CONTROL_REMEDIATION]
    
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
    text_data.extend(load_files(file_paths))
    
    # Split and process text
    chunks = split_text(text_data)
    vectorstore = create_vectorstore(chunks)
    llm = load_llm()
    reranker = create_reranker(llm)
    conversation = get_conversation_chain(vectorstore, llm, reranker)

    # Analyze access control and remediation coverage
    analyze_access_control_coverage(vectorstore)

    print("Access Control and Remediation RAG System Ready. Type 'exit' to quit.")
    while True:
        user_question = input("Ask a question about access control and remediation in cybersecurity (or type 'exit' to quit): ")
        if user_question.lower() == 'exit':
            break
        handle_userinput(user_question, conversation)

if __name__ == "__main__":
    main()
