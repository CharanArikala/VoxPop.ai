import os
import utils
import streamlit as st
from chat_effect import StreamHandler
from langchain_community.llms.fireworks import Fireworks 
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI


st.set_page_config(page_title="ChatwithVopox.ai", page_icon="📄")
st.header('Know about your customer voice')

class CustomDataChatbot:
    def __init__(self):
        self.OPENAI_API_KEY = utils.configure_openai_api_key()

    def save_file(self, file):
        folder = 'tmp'
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        file_path = f'./{folder}/{file.name}'
        with open(file_path, 'wb') as f:
            f.write(file.getvalue())
        return file_path

    @st.spinner('Analyzing documents..')
    def setup_qa_chain(self):

        embeddings = HuggingFaceEmbeddings()
        vectordb = FAISS.load_local("faiss_db", embeddings)

        retriever = vectordb.as_retriever(
            search_type='mmr',
            search_kwargs={'k':2, 'fetch_k':4}
        )

        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
        )

        # llm = Fireworks(
        #     fireworks_api_key='',
        #     model="accounts/fireworks/models/mistral-7B-instruct-v0.1",
        #     max_tokens=256
        #     )

        llm = OpenAI(openai_api_key=self.OPENAI_API_KEY)
        
        qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory, verbose=True)
        return qa_chain

    @utils.enable_chat_history
    def main(self):

        user_query = st.chat_input(placeholder="Type here !!!")

        if user_query:
            qa_chain = self.setup_qa_chain()

            utils.display_msg(user_query, 'user')

            with st.chat_message("assistant"):
                st_cb = StreamHandler(st.empty())
                response = qa_chain.run(user_query, callbacks=[st_cb])
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    obj = CustomDataChatbot()
    obj.main()