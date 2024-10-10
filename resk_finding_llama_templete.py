from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.llms import HuggingFacePipeline

def load_llm():
    # Load the LLM using the Hugging Face pipeline
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device="cuda")
    
    # Wrap the Hugging Face pipeline with HuggingFacePipeline for Langchain compatibility
    llm = HuggingFacePipeline(pipeline=llm_pipeline)
    return llm

def create_reranker():
    try:
        reranker_model_id = "cross-encoder/ms-marco-MiniLM-L-12-v2"
        tokenizer = AutoTokenizer.from_pretrained(reranker_model_id)
        model = AutoModelForSequenceClassification.from_pretrained(
            reranker_model_id, trust_remote_code=True, use_safetensors=True
        )
        reranker_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer, device="cuda")
        logging.info("Reranker created successfully.")
        return reranker_pipeline
    except Exception as e:
        logging.error(f"Error creating reranker: {e}")
        raise

def rerank_documents(question, docs, reranker):
    inputs = [f"{question} [SEP] {doc.page_content}" for doc in docs]
    scores = reranker(inputs, truncation=True)
    for doc, score in zip(docs, scores):
        doc.metadata["score"] = score["score"]
    return sorted(docs, key=lambda x: x.metadata["score"], reverse=True)

def get_conversation_chain(vectorstore, llm, reranker):
    memory = ConversationBufferWindowMemory(k=1, memory_key="chat_history", return_messages=True)

    # Custom retriever function that reranks results
    def retriever_with_reranking(question):
        retrieved_docs = vectorstore.similarity_search(question)
        reranked_docs = rerank_documents(question, retrieved_docs, reranker)
        return reranked_docs

    # Conversational chain that uses reranked retriever
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever_with_reranking,  # Custom retriever
        memory=memory,
        verbose=False
    )
    return conversation_chain

def handle_userinput(user_question, conversation):
    response = conversation({'question': user_question})
    latest_response = response['answer'].strip()
    print("\n--- Answer ---")
    print(latest_response)
    return latest_response

def main():
    try:
        logging.info("Executing RAG system for Access Control and Remediation Analysis...")

        # Create vector store, LLM, and reranker
        text_data = [ACCESS_CONTROL_PRINCIPLES, OWASP_TOP_10_2021, NIST_ACCESS_CONTROL, ACCESS_CONTROL_REMEDIATION, AC3_ACCESS_ENFORCEMENT]
        chunks = split_text(text_data)
        vectorstore = create_vectorstore(chunks)
        llm = load_llm()
        reranker = create_reranker()
        conversation = get_conversation_chain(vectorstore, llm, reranker)

        logging.info("Access Control and Remediation RAG System Ready.")
        print("Access Control and Remediation RAG System Ready. Type 'exit' to quit.")

        while True:
            user_question = input("Ask a question (or type 'exit' to quit): ")
            if user_question.strip().lower() == 'exit':
                break
            handle_userinput(user_question, conversation)

    except Exception as e:
        logging.error(f"An error occurred in the main function: {str(e)}")
        print("An error occurred. Please check the log for details.")

if __name__ == "__main__":
    main()
