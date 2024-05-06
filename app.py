import io
import logging
import uuid
from pathlib import Path
from typing import Dict

import matplotlib
import pandas as pd
import streamlit as st

from pandasai import SmartDataframe, Agent, Config
from pandasai.callbacks import StdoutCallback
from pandasai.helpers import Logger

from middleware.base import CustomChartsMiddleware
from parser.response_parser import CustomResponseParser
from util import get_open_ai_model, get_ollama_model, get_baidu_as_model, get_prompt_template, get_baidu_qianfan_model

logger = Logger()

matplotlib.rc_file("./.matplotlib/.matplotlibrc");

# page settings
st.set_page_config(page_title="Mutual Fund Performance Insights Application", layout="wide")
st.header("Mutual Fund Performance Insights Application")

class AgentWrapper:
    id: str
    agent: Agent

    def __init__(self) -> None:
        self.agent = None
        self.id = str(uuid.uuid4())

    def get_llm(self):
        op = st.session_state.last_option
        llm = None
        if op == "Ollama":
            llm = get_ollama_model(st.session_state.ollama_model, st.session_state.ollama_base_url)
        elif op == "Baidu/AIStudio-Ernie-Bot":
            if st.session_state.access_token != "":
                llm = get_baidu_as_model(st.session_state.access_token)
        elif op == "Baidu/Qianfan-Ernie-Bot":
            if st.session_state.client_id != "" and st.session_state.client_secret != "":
                llm = get_baidu_qianfan_model(st.session_state.client_id, st.session_state.client_secret)
        if llm is None:
            st.error("LLM initialization failed, check LLM configuration")
        return llm

    def set_file_data(self, df):
        llm = self.get_llm()
        if llm is not None:
            print("llm.type", llm.type)
            config = Config(
                llm=llm,
                callback=StdoutCallback(),
                # middlewares=[CustomChartsMiddleware()],
                response_parser=CustomResponseParser,
                custom_prompts={
                    "generate_python_code": get_prompt_template()
                },
                enable_cache=False,
                verbose=True
            )
            self.agent = Agent(df, config=config, memory_size=memory_size)
            self.agent._lake.add_middlewares(CustomChartsMiddleware())
            st.session_state.llm_ready = True

    def chat(self, prompt):
        if self.agent is None:
            st.error("LLM initialization failed, check LLM configuration")
            st.stop()
        else:
            return self.agent.chat(prompt)

    def start_new_conversation(self):
        self.agent.start_new_conversation()
        st.session_state.chat_history = []

@st.cache_resource
def get_agent(agent_id) -> AgentWrapper:
    agent = AgentWrapper()
    return agent

chat_history_key = "chat_history"
if chat_history_key not in st.session_state:
    st.session_state[chat_history_key] = []

if "llm_ready" not in st.session_state:
    st.session_state.llm_ready = False

# Description
tab1= st.tabs(["Workspace"])

# DataGrid
with st.expander("DataGrid Content") as ep:
    grid = st.dataframe(pd.DataFrame(), use_container_width=True)
counter = st.markdown("")

# Sidebar layout
with st.sidebar:
    option = st.selectbox("Choose LLM", ["Baidu/AIStudio-Ernie-Bot", "Baidu/Qianfan-Ernie-Bot", "Ollama"])

    # Initialize session keys
    if "api_token" not in st.session_state:
        st.session_state.api_token = ""
    if "access_token" not in st.session_state:
        st.session_state.access_token = ""
    if "ollama_model" not in st.session_state:
        st.session_state.ollama_model = ""
    if "ollama_base_url" not in st.session_state:
        st.session_state.ollama_base_url = ""
    if "client_id" not in st.session_state:
        st.session_state.client_id = ""
    if "client_secret" not in st.session_state:
        st.session_state.client_secret = ""

    # Initialize model configuration panel
    if option == "Baidu/AIStudio-Ernie-Bot":
        access_token = st.text_input("Access Token", st.session_state.access_token, type="password",
                                     placeholder="Access token")
    elif option == "Baidu/Qianfan-Ernie-Bot":
        client_id = st.text_input("Client ID", st.session_state.client_id, placeholder="Client ID")
        client_secret = st.text_input("Client Secret", st.session_state.client_secret, type="password",
                                      placeholder="Client Secret")
    elif option == "Ollama":
        ollama_model = st.selectbox(
            "Choose Ollama Model",
            ["starcoder:7b", "codellama:7b-instruct-q8_0", "zephyr:7b-alpha-q8_0"]
        )
        ollama_base_url = st.text_input("Ollama BaseURL", st.session_state.ollama_base_url,
                                        placeholder="http://localhost:11434")

    memory_size = st.selectbox("Memory Size", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=9)

    if st.button("+ New Chat"):
        st.session_state.llm_ready = False
        st.session_state[chat_history_key] = []

    # Validation
    info = st.markdown("")

    if option == "Baidu/AIStudio-Ernie-Bot":
        if not access_token:
            info.error("Invalid Access Token")
        if access_token != st.session_state.access_token:
            st.session_state.access_token = access_token
            st.session_state.llm_ready = False
    elif option == "Baidu/Qianfan-Ernie-Bot":
        if client_id != st.session_state.client_id:
            st.session_state.client_id = client_id
            st.session_state.llm_ready = False
        if client_secret != st.session_state.client_secret:
            st.session_state.client_secret = client_secret
            st.session_state.llm_ready = False
    elif option == "Ollama":
        if ollama_model != st.session_state.ollama_model:
            st.session_state.ollama_model = ollama_model
            st.session_state.llm_ready = False
        if ollama_base_url != st.session_state.ollama_base_url:
            st.session_state.ollama_base_url = ollama_base_url
            st.session_state.llm_ready = False

    if "last_option" not in st.session_state:
        st.session_state.last_option = None

    if option != st.session_state.last_option:
        st.session_state.last_option = option
        st.session_state.llm_ready = False

    if "last_memory_size" not in st.session_state:
        st.session_state.last_memory_size = None

    if memory_size != st.session_state.last_memory_size:
        st.session_state.last_memory_size = memory_size
        st.session_state.llm_ready = False

logger.log(f"st.session_state.llm_ready={st.session_state.llm_ready}", level=logging.INFO)

if not st.session_state.llm_ready:
    st.session_state.agent_id = str(uuid.uuid4())

with st.sidebar:
    st.divider()
    file = st.file_uploader("Upload File", type=["xlsx", "csv"])
    if file is None:
        st.session_state.uploaded = False
        if st.session_state.llm_ready:
            get_agent(st.session_state.agent_id).start_new_conversation()

    if "last_file" not in st.session_state:
        st.session_state.last_file = None

    if file is not None:
        file_obj = io.BytesIO(file.getvalue())
        file_ext = Path(file.name).suffix.lower()
        if file_ext == ".csv":
            df = pd.read_csv(file_obj)
        else:
            df = pd.read_excel(file_obj)
        grid.dataframe(df)
        counter.info("Total: **%s** records" % len(df))

        if file != st.session_state.last_file or st.session_state.llm_ready is False:
            st.session_state.agent_id = str(uuid.uuid4())
            get_agent(st.session_state.agent_id).set_file_data(df)

        st.session_state.last_file = file

with st.sidebar:
    st.markdown("""
    <style>
        .tw_share {
            display: inline-block;
            cursor: pointer;
        }
        
        .tw_share a {
            text-decoration: none;
        }
        
        .tw_share span {
            color: white;
        }
        
        .tw_share span {
            margin-left: 2px;
        }
        
        .tw_share:hover svg path {
            fill: #1da1f2;
        }
        
        .tw_share:hover span {
            color: #1da1f2;
        }
    </style>
    <div class="tw_share">
    </div>
    """, unsafe_allow_html=True)

# Alert box for PDF to Excel conversion
if not st.session_state.llm_ready:
    st.warning("To proceed, please convert the PDF data into Excel format.")

# ChatBox layout
for item in st.session_state.chat_history:
    with st.chat_message(item["role"]):
        if "type" in item and item["type"] == "plot":
            tmp = st.image(item['content'])
        elif "type" in item and item["type"] == "dataframe":
            tmp = st.dataframe(item['content'])
        else:
            st.markdown(item["content"])

prompt = st.chat_input("Input the question here")
if prompt is not None:
    st.chat_message("user").markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        if not st.session_state.llm_ready:
            response = "Please upload the file and configure the LLM well first"
            st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        else:
            tmp = st.markdown(f"Analyzing, hold on please...")

            response = get_agent(st.session_state.agent_id).chat(prompt)

            if isinstance(response, SmartDataframe):
                tmp.dataframe(response.dataframe)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": response.dataframe, "type": "dataframe"})
            elif isinstance(response, Dict) and "type" in response and response["type"] == "plot":
                tmp.image(f"{response['value']}")
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": response["value"], "type": "plot"})
            else:
                tmp.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})

