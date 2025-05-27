from typing import Set

from backend.core import run_llm
import streamlit as st

st.set_page_config(
    page_title="Document-Helper",
    page_icon="https://agile-systems.de/wp-content/uploads/2024/03/LangChain-Logo.png"
)

# Sidebar with user profile
with st.sidebar:
    st.title("User Profile")
    
    # Profile picture
    profile_pic = st.sidebar.image(
        "https://cdn-icons-png.flaticon.com/512/1077/1077114.png",
        width=100
    )
    
    # User information
    st.sidebar.text_input("Name", placeholder="Enter your name")
    st.sidebar.text_input("Email", placeholder="Enter your email")
    
    # Add some spacing
    st.sidebar.markdown("---")

# LangChain logo, heading, and link
st.markdown(
    '''
    <div style="display: flex; align-items: center; gap: 12px;">
        <img src="https://agile-systems.de/wp-content/uploads/2024/03/LangChain-Logo.png" alt="LangChain Logo" width="40"/>
        <a href="https://python.langchain.com/" target="_blank" style="text-decoration: none; color: inherit; font-size: 2rem; font-weight: 700;">
            Langchain-Documentation Helper Bot
        </a>
    </div>
    ''',
    unsafe_allow_html=True
)

# Inject custom CSS to make the submit button always filled green with white text
st.markdown(
    """
    <style>
    .stButton > button, .stButton button[kind="primary"] {
        background-color: #00B16B !important;
        color: #fff !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
        border-radius: 8px !important;
        transition: background 0.2s;
    }
    .stButton > button:hover, .stButton button[kind="primary"]:hover {
        background-color: #009e5c !important;
        color: #fff !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Prompt input with submit button
with st.form(key="prompt_form", clear_on_submit=False):
    prompt = st.text_input("Prompt", placeholder="Enter your prompt here..", key="prompt_input")
    submit = st.form_submit_button("Submit")

if (
    "chat_answers_history" not in st.session_state
    and "user_prompt_history" not in st.session_state
    and "chat_history" not in st.session_state
):
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []


def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i}. {source}\n"
    return sources_string


if submit and prompt:
    with st.spinner("Generating Response"):
        generated_response = run_llm(query=prompt, chat_history=st.session_state["chat_history"])
        sources = set([doc.metadata["source"] for doc in generated_response["source_documents"]])

        formatted_response = f"{generated_response['result']} \n\n {create_sources_string(sources)}"

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", generated_response["result"]))

if st.session_state["chat_answers_history"]:
    for generated_response, user_query in zip(st.session_state["chat_answers_history"], st.session_state["user_prompt_history"]):
        st.chat_message("user").write(user_query)
        st.chat_message("assistant").write(generated_response)