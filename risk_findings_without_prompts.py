import pandas as pd
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS as LangchainFAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseRetriever
from langchain import HuggingFacePipeline
import torch
import os
from docx import Document
import pdfplumber
import re
import logging
from langchain.retrievers import VectorStoreRetriever

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set display options
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)

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

AC3_ACCESS_ENFORCEMENT = """
AC-3 Access Enforcement:

Definition: The information system enforces approved authorizations for logical access to information and system resources in accordance with applicable access control policies.

Key points:
1. Implement access control policies (e.g., identity-based policies, role-based policies, attribute-based policies) and associated access enforcement mechanisms (e.g., access control lists, access control matrices, cryptography).
2. Ensure that access enforcement occurs immediately before or as part of access to objects.
3. Employ automated mechanisms to support the management of information system accounts.
4. Enforce the principle of least privilege, allowing only authorized accesses for users which are necessary to accomplish assigned tasks.
5. Implement separation of duties through assigned information system access authorizations.
6. Enforce approved authorizations for controlling the flow of information within the system and between interconnected systems.
7. Employ dual authorization for highly sensitive operations or access to critical resources.

Common implementation strategies:
- Use of access control lists (ACLs)
- Role-based access control (RBAC)
- Attribute-based access control (ABAC)
- Mandatory access control (MAC)
- Discretionary access control (DAC)
"""

def load_files(file_paths):
    text_data = []
    for file_path in file_paths:
        if not os.path.isfile(file_path):
            logging.warning(f"File not found: {file_path}")
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
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + '\n'
                        
                        page_tables = page.extract_tables()
                        for table in page_tables:
                            tables.append(table)

                for table in tables:
                    table_text = '\n'.join([', '.join(filter(None, row)) for row in table if any(row)])
                    text += f"\nTable:\n{table_text}\nEND TABLE\n"

                text_data.append(text)
            elif extension == '.docx':
                doc = Document(file_path)
                fullText = [para.text for para in doc.paragraphs]
                text = '\n'.join(fullText)
                text_data.append(text)
            else:
                logging.warning(f"Unsupported file type: {file_path}")
        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}")
    return text_data

def extract_access_control_info(text):
    access_control_patterns = [
        r'access control\s*:\s*([^.!?\n]+)',
        r'authentication\s*:\s*([^.!?\n]+)',
        r'authorization\s*:\s*([^.!?\n]+)',
        r'least privilege\s*:\s*([^.!?\n]+)',
        r'separation of duties\s*:\s*([^.!?\n]+)',
        r'need-to-know\s*:\s*([^.!?\n]+)',
        r'role-based access\s*:\s*([^.!?\n]+)',
        r'multi-factor authentication\s*:\s*([^.!?\n]+)',
        r'access enforcement\s*:\s*([^.!?\n]+)',
        r'AC-3\s*:\s*([^.!?\n]+)',
        r'Access Enforcement\s*:\s*([^.!?\n]+)',
        r'logical access\s*:\s*([^.!?\n]+)',
        r'access control policies\s*:\s*([^.!?\n]+)',
        r'access enforcement mechanisms\s*:\s*([^.!?\n]+)',
    ]
    
    access_control_info = []
    for pattern in access_control_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        access_control_info.extend(matches)
    
    return access_control_info

def split_text(text_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1020,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.create_documents(text_data)
    
    for chunk in chunks:
        chunk.metadata['access_control_info'] = extract_access_control_info(chunk.page_content)
    
    return chunks

def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = LangchainFAISS.from_documents(chunks, embeddings)
    return vectorstore

def load_llm():
    model_id = "meta-llama/Meta-Llama-3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Assuming the model uses bfloat16 precision and requires device mapping
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        temperature=0.1,
        max_length=1024,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

def create_reranker():
    # Load a model for sequence classification (e.g., cross-encoder for re-ranking)
    reranker_model_id = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_pipeline = pipeline("text-classification", model=reranker_model_id, device_map="auto")
    return reranker_pipeline

def rerank_documents(question, docs, reranker):
    # Prepare inputs for the reranker
    inputs = [f"{question} [SEP] {doc.page_content}" for doc in docs]
    # Get relevance scores
    scores = reranker(inputs, truncation=True)
    # Attach scores to docs and sort
    for doc, score in zip(docs, scores):
        doc.metadata["score"] = score["score"]
    reranked_docs = sorted(docs, key=lambda x: x.metadata["score"], reverse=True)
    return reranked_docs

class RerankRetriever(VectorStoreRetriever):
    def __init__(self, vectorstore, reranker):
        # Initialize the base class
        super().__init__(vectorstore=vectorstore)
        # Ensure reranker is set after base class initialization
        self.reranker = reranker

    def get_relevant_documents(self, query):
        # Retrieve documents using the vectorstore
        docs = self.vectorstore.similarity_search(query, k=10)
        # Re-rank the documents
        reranked_docs = rerank_documents(query, docs, self.reranker)
        return reranked_docs[:5]  # Return top N reranked docs

    async def aget_relevant_documents(self, query):
        return self.get_relevant_documents(query)

def get_conversation_chain(vectorstore, llm, reranker):
    memory = ConversationBufferWindowMemory(
        k=1, memory_key="chat_history", return_messages=True
    )
    retriever = RerankRetriever(vectorstore, reranker)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        get_chat_history=lambda x: "",
        verbose=False,
    )
    return conversation_chain

def handle_userinput(user_question, conversation):
    response = conversation({'question': user_question})
    latest_response = response['answer'].strip()
    
    print("\n--- Answer ---")
    print(latest_response)
    
    return latest_response

def analyze_access_control_coverage(vectorstore):
    all_access_control_info = []
    ac3_related_info = []
    for doc in vectorstore.docstore._dict.values():
        info = doc.metadata.get('access_control_info', [])
        all_access_control_info.extend(info)
        ac3_related_info.extend([item for item in info if 'AC-3' in item or 'Access Enforcement' in item])
    
    access_control_counts = pd.Series(all_access_control_info).value_counts()
    ac3_counts = pd.Series(ac3_related_info).value_counts()

    print("Access Control and Remediation Coverage Analysis:")
    print(access_control_counts)
    print("\nTotal unique access control measures and remediation strategies:", len(access_control_counts))
    print("Top 10 most mentioned access control measures and remediation strategies:")
    print(access_control_counts.head(10))
    
    print("\nAC-3 Access Enforcement Specific Analysis:")
    print(ac3_counts)
    print("\nTotal unique AC-3 related measures:", len(ac3_counts))
    print("Top 5 most mentioned AC-3 related measures:")
    print(ac3_counts.head(5))

def main():
    try:
        logging.info("Executing RAG system for Access Control and Remediation Analysis...")
        
        # Add static data
        text_data = [ACCESS_CONTROL_PRINCIPLES, OWASP_TOP_10_2021, NIST_ACCESS_CONTROL, ACCESS_CONTROL_REMEDIATION, AC3_ACCESS_ENFORCEMENT]
        
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
        reranker = create_reranker()
        conversation = get_conversation_chain(vectorstore, llm, reranker)

        # Analyze access control and remediation coverage
        analyze_access_control_coverage(vectorstore)

        logging.info("Access Control and Remediation RAG System Ready.")
        print("Access Control and Remediation RAG System Ready. Type 'exit' to quit.")

        while True:
            user_question = input("Ask a question about access control and remediation in cybersecurity (or type 'exit' to quit): ")
            if user_question.strip().lower() == 'exit':
                break
            handle_userinput(user_question, conversation)

    except Exception as e:
        logging.error(f"An error occurred in the main function: {str(e)}")
        print("An error occurred. Please check the log for details.")

if __name__ == "__main__":
    main()
