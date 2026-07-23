import os
import time
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent, tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS
from langchain_core.globals import set_debug, set_verbose
from langchain_core.stores import InMemoryStore
from pydantic import SecretStr
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from tavily import TavilyClient

# set_debug(True)
# set_verbose(True)

st.title("FAISS RAG Agent")
db_folder_path="faiss"
pdf_path="pdf.pdf"
open_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
@tool
def initialise_rag(query:str="",num_results:int=0):
    """use this to fetch results about user_query from vectorDB."""
    if not os.path.exists(db_folder_path):
       os.makedirs(db_folder_path)
    faiss_index_path=os.path.join(db_folder_path,"index.faiss")
    pkl_file_path=os.path.join(db_folder_path,"index.pkl")
    loader=PyPDFLoader(file_path=pdf_path)
    docs=loader.load()
    if os.path.exists(faiss_index_path) and os.path.exists(pkl_file_path):
       DB=FAISS.load_local(
            folder_path=db_folder_path,
            embeddings=open_embeddings,
            allow_dangerous_deserialization=True
       )   
    else:
        DB=FAISS.from_documents(
            documents=docs,
            embedding=open_embeddings
        ) 
        DB.save_local(folder_path=db_folder_path)
    return DB.similarity_search(query, k=num_results)


@tool
def web_search(query:str):
    """
    use this tool to search the web to refine user inputs along with vectorDB results.
    """
    tavily_client=TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
    return tavily_client.search(query=query)


system_prompt=()

if "messages" not in st.session_state:
    st.session_state["messages"]=[]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def main():
   
    
    with st.spinner("loading resources..."):
        try:
            api=os.environ.get("OPENAI_API_KEY")
            if not api:
                st.error("OPENAI_API_KEY not found in environment variables.")
                st.stop()
            
            api_secret = SecretStr(api)
            tools=[web_search,initialise_rag]
            llm=ChatOpenAI(
                model="gpt-4o",
                api_key=api_secret,
                temperature=0,
                verbose=True,
            )

            prompt=ChatPromptTemplate.from_messages([
                        ("system",system_prompt),
                        MessagesPlaceholder(variable_name="chat_history"),
                        ("human","{input}"),
                        MessagesPlaceholder(variable_name="agent_scratchpad")
                    ])

            agent=create_tool_calling_agent(llm,tools,prompt)
            agent_executor=AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True
            )
        except Exception as e:
            st.error(f"Failed to initialize the agent: {e}")
            st.stop()

    if user_query := st.chat_input("please enter your question...."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
            
        with st.chat_message("assistant"):
            with st.spinner("searching and thinking for answers..."):
                response = agent_executor.invoke({"input": user_query, "chat_history": st.session_state.messages})
                answer = response["output"]
                st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

            
if __name__=="__main__":
    main()
