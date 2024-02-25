import os
import random
import streamlit as st

def enable_chat_history(func):
    if os.environ.get("FIREWORK_API_KEY"):

        current_page = func.__qualname__
        if "current_page" not in st.session_state:
            st.session_state["current_page"] = current_page
        if st.session_state["current_page"] != current_page:
            try:
                st.cache_resource.clear()
                del st.session_state["current_page"]
                del st.session_state["messages"]
            except:
                pass

        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
        for msg in st.session_state["messages"]:
            st.chat_message(msg["role"]).write(msg["content"])

    def execute(*args, **kwargs):
        func(*args, **kwargs)
    return execute

def display_msg(msg, author):

    st.session_state.messages.append({"role": author, "content": msg})
    st.chat_message(author).write(msg)

def configure_firework_api_key():
    OPENAI_api_key = st.sidebar.text_input(
        label="OPENAIs API Key",
        type="password",
        value=st.session_state['OPENAI_API_KEY'] if 'OPENAI_API_KEY' in st.session_state else '',
        placeholder="AzV..."
        )
    if OPENAI_api_key:
        st.session_state['OPENAI_API_KEY'] = OPENAI_api_key
        os.environ['OPENAI_API_KEY'] = OPENAI_api_key
    else:
        st.error("Please add your OPENAI API key to continue.")
        st.stop()
    return OPENAI_api_key