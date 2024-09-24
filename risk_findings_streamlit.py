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
import pdfplumber
import re
import streamlit as st

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

OWASP_TOP_10_2021 = """
OWASP Top 10 2021:

1. A01:2021-Broken Access Control
   - Moves up from the fifth position; 94% of applications were tested for some form of broken access control.

2. A02:2021-Cryptographic Failures
   - Previously known as 'Sensitive Data Exposure,' which is more of a broad symptom rather than a root cause.

3. A03:2021-Injection
   - Drops to the third position. 94% of the applications were tested for some form of injection, and the 33 CWEs mapped into this category have the second-highest number of occurrences in applications.

4. A04:2021-Insecure Design
   - A new category for 2021, with a focus on risks related to design flaws.

5. A05:2021-Security Misconfiguration
   - Moves up from #6 in the previous edition; 90% of applications were tested for some form of misconfiguration.

6. A06:2021-Vulnerable and Outdated Components
   - Previously titled 'Using Components with Known Vulnerabilities' and is #2 in the Top 10 community survey, but also had enough data to make the Top 10 via data analysis.

7. A07:2021-Identification and Authentication Failures
   - Previously was 'Broken Authentication' and sliding down from the second position, and now includes CWEs that are more related to identification failures.

8. A08:2021-Software and Data Integrity Failures
   - A new category for 2021, focusing on making assumptions related to software updates, critical data, and CI/CD pipelines without verifying integrity.

9. A09:2021-Security Logging and Monitoring Failures
   - Previously 'Insufficient Logging & Monitoring,' this category is expanded to include more types of failures, is challenging to test for, and isn't well represented in the CVE/CVSS data.

10. A10:2021-Server-Side Request Forgery
    - A new addition to the Top 10 for 2021, this category represents the scenario where the security community members are telling us this is important, even though it's not illustrated in the data at this time.
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
            st.warning(f"File not found: {file_path}")
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
                tables = []
                current_table = None
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        # Extract text
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + '\n'
                        
                        # Extract tables
                        page_tables = page.extract_tables()
                        for table in page_tables:
                            if current_table is None:
                                current_table = table
                            else:
                                # Check if this table is a continuation of the previous one
                                if len(table) == len(current_table[0]):  # Same number of columns
                                    current_table.extend(table[1:])  # Append rows, skipping header
                                else:
                                    # This is a new table, save the current one and start a new one
                                    tables.append(current_table)
                                    current_table = table
                    
                    # Add the last table if there is one
                    if current_table:
                        tables.append(current_table)
                
                # Process all extracted tables
                for table in tables:
                    table_text = '\n'.join([', '.join(filter(None, row)) for row in table if any(row)])
                    text += f"\nTABLE:\n{table_text}\nEND TABLE\n"
                
                text_data.append(text)
            elif extension == '.docx':
                doc = Document(file_path)
                fullText = [para.text for para in doc.paragraphs]
                text = '\n'.join(fullText)
                text_data.append(text)
            else:
                st.warning(f"Unsupported file type: {file_path}")
        except Exception as e:
            st.error(f"Error processing {file_path}: {e}")
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
        chunk_size=1024,
        chunk_overlap=100,
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

    prompt_template = """You are an AI assistant specializing in cybersecurity with a focus on access control and remediation strategies. If you don't know the answer, just say that you don't know—don't try to make up an answer.

    Question: {question}

    Provide a detailed answer, including:
    1. Relevant access control measures
    2. An assessment of the effectiveness of these access control measures
    3. Potential vulnerabilities or gaps in the access control framework
    4. Detailed remediation strategies for potential issues
    5. How the identified access control measures and remediation strategies align with industry standards (e.g., NIST SP 800-53, OWASP Top 10)
    6. Best practices for implementing and maintaining strong access control
    7. Any additional recommendations for improving overall access control posture

    Detailed answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["question"]
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
            st.write(f"**User:** {message.content}")
        else:
            st.write(f"**Access Control & Remediation Bot:** {message.content}")
            
    st.write("\n**Source Documents:**")
    for doc in source_documents:
        st.write(f"- {doc.metadata.get('source', 'Unknown source')}")
        if 'access_control_info' in doc.metadata:
            st.write("  **Access Control and Remediation Information:**")
            for info in doc.metadata['access_control_info']:
                st.write(f"    - {info}")
    st.write("\n")

def analyze_access_control_coverage(vectorstore):
    """Analyze the coverage of access control measures and remediation strategies in the dataset."""
    all_access_control_info = []
    for doc in vectorstore.docstore._dict.values():
        all_access_control_info.extend(doc.metadata.get('access_control_info', []))
    
    access_control_counts = pd.Series(all_access_control_info).value_counts()
    st.write("**Access Control and Remediation Coverage Analysis:**")
    st.write(access_control_counts)
    st.write("\n**Total unique access control measures and remediation strategies:**", len(access_control_counts))
    st.write("**Top 10 most mentioned access control measures and remediation strategies:**")
    st.write(access_control_counts.head(10))

def main():
    st.title("Access Control and Remediation Analysis")
    st.write("Executing RAG system for Access Control and Remediation Analysis...")

    # Add static data
    text_data = [ACCESS_CONTROL_PRINCIPLES, OWASP_TOP_10_2021, NIST_ACCESS_CONTROL, ACCESS_CONTROL_REMEDIATION]
    
    # Directory containing data files
    data_dir = 'data'
    # Supported file extensions
    supported_extensions = ['.csv', '.txt', '.pdf', '.docx']
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

    st.write("Access Control and Remediation RAG System Ready.")
    user_question = st.text_input("Ask a question about access control and remediation in cybersecurity:")
    if user_question:
        handle_userinput(user_question, conversation)

if __name__ == "__main__":
    main()
