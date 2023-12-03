# cheak out streamlit.io!
# https://streamlit.io

from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts.chat import ChatPromptTemplate
from langchain.tools import DuckDuckGoSearchRun
from langchain.tools import tool
import streamlit as st
import datetime
import pytz

st.set_page_config(page_title="LangChain: Chat with search", page_icon="🦜")
st.title("🦜 LangChain: Chat with search")

# the definition of my custom function so it knows that needs to be done to convert
# to a time zone before sending it through.
functions = [
    {
        "name": "get_current_time",
        "description": "Get the current time given a location",
        "parameters": {
            "type": "object",
            "properties": {
                "time_zone": {
                    "type": "string",
                    "description": "The time zone of the location we want to find the time for",
                },
            },
            "required": ["time_zone"],
        },
    }
]

msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output"
)
if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")
    st.session_state.steps = {}

@tool
def get_current_time(time_zone):
    """Get the current time in a given location"""
    time = datetime.datetime.now(pytz.timezone(time_zone))
    formatted_time = time.strftime("%H:%M:%S")
    time_info = {
        "time_zone": time_zone,
        "time": formatted_time,
    }
    return f'The current time is {time_info["time"]} in {time_info["time_zone"]}'

avatars = {"human": "user", "ai": "assistant"}
for idx, msg in enumerate(msgs.messages):
    with st.chat_message(avatars[msg.type]):
        # Render intermediate steps if any were saved
        for step in st.session_state.steps.get(str(idx), []):
            if step[0].tool == "_Exception":
                continue
            with st.status(f"**{step[0].tool}**: {step[0].tool_input}", state="complete"):
                st.write(step[0].log)
                st.write(step[1])
        st.write(msg.content)

if prompt := st.chat_input(placeholder="Who won the Women's U.S. Open in 2018?"):
    st.chat_message("user").write(prompt)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", streaming=True)
    
    tools = [DuckDuckGoSearchRun(name="Search"), get_current_time]
    chat_agent = ConversationalChatAgent.from_llm_and_tools(llm=llm, tools=tools)
    executor = AgentExecutor.from_agent_and_tools(
        agent=chat_agent,
        tools=tools,
        memory=memory,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = executor(prompt, callbacks=[st_cb])
        st.write(response["output"])
        st.session_state.steps[str(len(msgs.messages) - 1)] = response["intermediate_steps"]