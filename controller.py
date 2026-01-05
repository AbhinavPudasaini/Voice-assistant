from langchain_groq import ChatGroq
from langchain.messages import HumanMessage, SystemMessage, trim_messages
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os

# API Keys
GROQ_API_KEY = "groq_api_key"

# Main Brain LLM
llm = ChatGroq(model="openai/gpt-oss-120b", api_key=GROQ_API_KEY)

# Fast Decider LLM
decider_llm = ChatGroq(model="moonshotai/kimi-k2-instruct-0905", api_key=GROQ_API_KEY)

# Store for chat histories
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# --- Main Agent Setup ---
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an Agent with expertise in Customer care support assistance.
You will be mimic the customer support care to provide assistance to the user.
     You are customer support of an dental hopsital, so you will mimic as so. Your name is rita and hospital name is Jf kennedy Hospital.
     FOr now just make up a conversation on whatever the customer ask, you don't have to be accurate. this is just for demo, so whatever the customer says, answer.
The conversation starts with user asking questions, and you have to respectfully greet and make a followup answer or question.
     
NOTE: Don't output any unnecessary text except the conversation answers. Your output will be directly passed to text to speech model, so be carefull on this."""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

agent_chain = agent_prompt | llm

agent_with_history = RunnableWithMessageHistory(
    agent_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

def call_agent(text: str, session_id: str = "default"):
    response = agent_with_history.invoke(
        {"input": text},
        config={"configurable": {"session_id": session_id}}
    )
    return response.content

decider_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a turn‑taking and intent‑completion classifier for a real‑time voice assistant. 
Analyze the user's latest input in the context of the conversation history to determine if the assistant should respond.

Labels:
1. READY_FOR_RESPONSE:
   - The user has finished a complete sentence, question, or request.
   - The user has provided a logical answer to the assistant's last question (e.g., Assistant: "What is your name?" -> User: "Abhinav").
   - The user's intent is clear and requires a reply.

2. USER_NOT_DONE:
   - The user is pausing, thinking, or using filler words (e.g., "uhm", "well...", "so I was thinking", "let me see").
   - The input is a partial sentence that doesn't yet form a complete thought or answer.
   - The user is trailing off or mid-sentence.

3. USER_INTERRUPTING:
   - The user is explicitly interrupting the assistant (e.g., "wait", "stop", "no, actually...", "hold on").
     
4. END_Call
    - The user has got what it has asked for and is satisfied with the call, then output this.

Decision Rule:
Compare the 'input' with the LAST message in the 'history'. If the assistant just asked a question and the user's input satisfies that question (even with a single word or with a unstructure format of grammer), classify as READY_FOR_RESPONSE.

Output ONLY the label. No punctuation or extra text."""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# decider_chain = decider_prompt | decider_llm
# ...existing code...# filepath: c:\Users\Abhinav\OneDrive\Desktop\Devpost\AgenticAI\Practices\VoiceAgent\controller.py
# ...existing code...
# --- Decider Setup ---
# decider_prompt = ChatPromptTemplate.from_messages([
#     ("system", """You are a turn‑taking and intent‑completion classifier for a real‑time voice assistant. 
# Analyze the user's latest input in the context of the conversation history to determine if the assistant should respond.

# Labels:
# 1. READY_FOR_RESPONSE:
#    - The user has finished a complete sentence, question, or request.
#    - The user has provided a logical answer to the assistant's last question (e.g., Assistant: "What is your name?" -> User: "Abhinav").
#    - The user's intent is clear and requires a reply.

# 2. USER_NOT_DONE:
#    - The user is pausing, thinking, or using filler words (e.g., "uhm", "well...", "so I was thinking", "let me see").
#    - The input is a partial sentence that doesn't yet form a complete thought or answer.
#    - The user is trailing off or mid-sentence.

# 3. USER_INTERRUPTING:
#    - The user is explicitly interrupting the assistant (e.g., "wait", "stop", "no, actually...", "hold on").

# Decision Rule:
# Compare the 'input' with the LAST message in the 'history'. If the assistant just asked a question and the user's input satisfies that question (even with a single word), classify as READY_FOR_RESPONSE.

# Output ONLY the label. No punctuation or extra text."""),
#     MessagesPlaceholder(variable_name="history"),
#     ("human", "{input}"),
# ])

decider_chain = decider_prompt | decider_llm
# ...existing code...

# ...existing code...
decider_chain = decider_prompt | decider_llm

def decider(text: str, session_id: str = "default"):
    # 1. Retrieve the main conversation history (Read-Only for context)
    chat_history = get_session_history(session_id)
    
    # 2. Invoke the chain directly (without RunnableWithMessageHistory)
    # We pass the existing messages so the LLM knows what the Bot just asked.
    # We pass the current accumulated 'text' as input.
    response = decider_chain.invoke(
        {
            "history": chat_history.messages, 
            "input": text
        })
        # filepath: c:\Users\Abhinav\OneDrive\Desktop\Devpost\AgenticAI\Practices\VoiceAgent\controller.py
# ...existing code...
decider_chain = decider_prompt | decider_llm

# DELETE or COMMENT OUT the decider_with_history block
# We do not want to automatically save decider interactions to the main history
# decider_with_history = RunnableWithMessageHistory(
#     decider_chain,
#     get_session_history,
#     input_messages_key="input",
#     history_messages_key="history",
# )

def decider(text: str, session_id: str = "default"):
    # 1. Retrieve the main conversation history (Read-Only for context)
    chat_history = get_session_history(session_id)
    
    # 2. Invoke the chain directly (without RunnableWithMessageHistory)
    # We pass the existing messages so the LLM knows what the Bot just asked.
    # We pass the current accumulated 'text' as input.
    response = decider_chain.invoke(
        {
            "history": chat_history.messages, 
            "input": text
        })
    return response.content.strip()

