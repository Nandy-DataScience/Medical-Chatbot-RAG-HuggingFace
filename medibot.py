import os
import streamlit as st

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint


DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

HF_TOKEN=os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id, HF_TOKEN):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.7,
        # model_kwargs={"max_length": 512}
    )
    return llm


def main():
    st.title("Hi, I am your Medibot(Medical Chatbot)!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt=st.chat_input("Write your Query here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
                Use the pieces of information provided in the context to answer user's question.
                If you dont know the answer, just say that you dont know, dont try to make up an answer. 
                Dont provide anything out of the given context.
                As a diabetes expert, please provide information about diabetes symptoms and management. Avoid discussing unrelated topics.
                You are a health assistant specializing in diabetes. 
                Your role is to provide accurate information about diabetes management, symptoms, and treatment options. 
                If a user asks about unrelated topics, respond with: 
                "I'm here to help with diabetes-related questions. Please ask me something about diabetes."
                "Here are some examples of questions I can answer: 
                1. What are the symptoms of diabetes?
                2. How can I manage my blood sugar levels?
                3. What foods should I avoid if I have diabetes?"

                Context: {context}
                Question: {question}

                Start the answer directly. No small talk please.
                """
        
        HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN=os.environ.get("HF_TOKEN")

        try: 
            vectorstore=get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            qa_chain=RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response=qa_chain.invoke({'query':prompt})

            diabetes_links = ["\n",
        "https://www.diabetes.org/","\n"
        "https://www.cdc.gov/diabetes/basics/index.html","\n"
        "https://www.who.int/health-topics/diabetes"
    ]

            # Format the links for output
            links_message = "For more information, check out these resources:\n" + "\n".join(diabetes_links)

            result=response["result"]
            # result_to_show=result +"\nSource Docs:\n"+str(diabetes_links)
            result_to_show = result + "\n\n" + links_message
            #response="Hi, I am MediBot!"
            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role':'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")


# Streamlit app
# Remove whitespace from the top of the page and sidebar
st.markdown(
        """
            <style>
                .appview-container .main .block-container {{
                    padding-top: {padding_top}rem;
                    padding-bottom: {padding_bottom}rem;
                    }}

            </style>""".format(
            padding_top=1, padding_bottom=1
        ),
        unsafe_allow_html=True,
    )


st.markdown("""
    <h3 style='text-align: left; color: red; padding-top: 30px; border-bottom: 3px solid green;'>
        Discover the AI Powered Healthcare Service 🤖🩺
    </h3>""", unsafe_allow_html=True)


side_bar_message = """
Hi! 👋 I'm here to help you with your Diabetes Queries. What would you like to know or explore?
\nHere are some areas you might be interested in:
1. **Diabetes Insights** 📊✅
2. **Customized Meal Tips** 🍽️👩‍🍳
3. **Diabetes Healing Journey** 🌅💪
4. **Doctor Recommendations** 🩺✨


Feel free to ask me anything about Diabetes!
"""

with st.sidebar:
    st.title('🤖MediBot: Your Diabetes Specialist')
    st.markdown(side_bar_message)


# Store LLM generated responses
# if "messages" not in st.session_state.keys():
#     st.session_state.messages = [{"role": "assistant", "content": initial_message}]

# Display or clear chat messages
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# def clear_chat_history():
#     st.session_state.messages = [{"role": "assistant", "content": initial_message}]
# st.button('Clear Chat', on_click=clear_chat_history)

# User-provided prompt
# if prompt := st.chat_input():
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

# Generate a new response if last message is not from assistant
# if st.session_state.messages[-1]["role"] != "assistant":
#     with st.chat_message("assistant"):
#         with st.spinner("Hold on, I'm fetching the latest medical advice for you..."):
#             response = get_response(prompt)
#             placeholder = st.empty()
#             full_response = response  # Directly use the response
#             placeholder.markdown(full_response)
#     message = {"role": "assistant", "content": full_response}
#     st.session_state.messages.append(message)

if __name__ == "__main__":
    main()