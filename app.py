import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchResults
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler

# Creating Tools
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=5, doc_content_chars_max=500)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=5, doc_content_chars_max=500)
arxiv_tool = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

search = DuckDuckGoSearchResults(name="Search")

# Streamlit app
st.title("Chat with Search!")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API key:", type="password")

if not api_key:
    st.info("Please add your GROQ API key to continue!")
    st.stop9()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I am a chatbot who can search the web. How can I help you today?"}
    ]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle user input
if prompt := st.chat_input(placeholder="What is Machine Learning?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Initialize LLM
    if api_key:
        llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)

        # Define tools
        tools = [search, arxiv_tool, wiki_tool]

        # Initialize agent
        search_agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True  # Useful for debugging
        )

        # Run search and display response
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = search_agent.run(prompt, callbacks=[st_cb])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
    else:
        st.error("Please enter your Groq API key in the sidebar.")
