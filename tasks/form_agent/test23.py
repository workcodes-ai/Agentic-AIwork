# import asyncio
# import json
# import logging
# from dataclasses import dataclass, field
# from datetime import datetime, timezone
# from pathlib import Path
# from typing import Optional, TypedDict as dict
# import nltk
# # nltk.download('names')
# # from nltk.corpus import names
# from dateutil import parser as date_parser
# from dotenv import load_dotenv
# from pydantic import BaseModel, constr, ValidationError, field_validator
# from langchain.schema import AIMessage, HumanMessage, SystemMessage
# from langchain_groq import ChatGroq
# from langgraph.graph import StateGraph
# import httpx 


# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# OUTPUT_FILE = Path("final_payload.json")
# MODEL_NAME = "llama3-70b-8192"
# TEMPERATURE = 0
# SUBMIT_URL = "http://127.0.0.1:8000/receive_data"

# # all_names = names.words()
# # print(all_names[:7944])  # Sample output, e.g. ['Abel', 'Aaron', 'Adrian', ...]
# # print(len(all_names))  # Number of names in total



# load_dotenv()
# import os
# import getpass

# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# if not GROQ_API_KEY:
#     GROQ_API_KEY = getpass.getpass("Enter GROQ_API_KEY: ").strip()
# if not GROQ_API_KEY:
#     raise RuntimeError("GROQ_API_KEY not provided.")

# llm = ChatGroq(api_key=GROQ_API_KEY, model_name=MODEL_NAME, temperature=TEMPERATURE)


# try:
#     nltk.data.find("corpora/names")
# except LookupError:
#     nltk.download("names")




# class UserData(BaseModel):
#     username: constr(strip_whitespace=True, min_length=1, max_length=80)
#     dob: str
#     address: constr(strip_whitespace=True, min_length=6)
#     phone: constr(strip_whitespace=True)


#     @field_validator("username")
#     def validate_username(cls, v):
#         if not all(ch.isalpha() or ch.isspace() for ch in v):
#             raise ValueError("Username must contain letters and spaces only")
#         words = v.lower().split()
       
#         return v

#     @field_validator("dob")
#     def validate_dob(cls, v):
#         try:
#             dt = date_parser.parse(v)
#         except Exception:
#             raise ValueError("Invalid date format")
#         now = datetime.now(timezone.utc)
#         age = (now - dt).days / 365.25
#         if not (5 <= age <= 120):
#             raise ValueError("Age must be between 5 and 120")
#         return dt.strftime("%Y-%m-%d")

# @dataclass
# class AgentState:
#     messages: list = field(default_factory=list)
#     final: bool = False
#     parsed_data: Optional[UserData] = None


# SYSTEM_PROMPT = """
# You are a careful data-collection agent. Your job is to collect and validate the following fields:

# - username
# - dob (date of birth)
# - address
# - phone

# RULES:
# 1. Validate each field using the rules below.
# 2. If a field is invalid or missing, ask ONE clarifying question.
# 3. Accept any valid date format for DOB.
# 4. Once all fields are valid, return only the final JSON — no extra text.

# SCHEMA:
# {
#   "username": string,
#   "dob": "YYYY-MM-DD",
#   "address": string,
#   "phone": string
# }

# VALIDATION RULES:
# - username: letters and spaces only, max 80 chars, must be real name (not gibberish)
# - dob: valid date, age between 5 and 120
# - address: minimum 6 characters
# - phone: exactly 10 digits
# """



# def extract_json_from_text(text: str) -> Optional[dict]:
#     import re
#     match = re.search(r"\{.*\}", text, re.DOTALL)
#     if not match:
#         return None
#     try:
#         data = json.loads(match.group(0))
#         return data
#     except Exception as e:
#         logger.warning(f"Failed to parse JSON: {e}")
#         return None

# async def llm_invoke(state: AgentState) -> AgentState:
#     logger.info("Calling LLM...")
    
#     resp = llm.invoke(state.messages)
#     state.messages.append(AIMessage(content=resp.content))
#     data = extract_json_from_text(resp.content)
#     if data:
#         try:
#             user_data = UserData(**data)
#             state.parsed_data = user_data
#             state.final = True
#             logger.info("Valid data received from LLM")
#         except ValidationError as e:
#             logger.info(f"Validation error in LLM output: {e}")
#             state.final = False
#     else:
#         state.final = False
#     return state


# async def get_user_input(state: AgentState) -> AgentState:
#     last_msg = state.messages[-1].content.strip()
#     print(f"\nLLM → {last_msg}")
#     while True:
#         user_input = input("Your reply (or 'exit'): ").strip()
#         if user_input.lower() == "exit":
#             logger.info("User requested exit.")
#             exit(0)
#         if user_input.lower() == "submit":
#             print("Cannot manually submit. Please answer the question.")
#             continue
#         break
#     state.messages.append(HumanMessage(content=user_input))
#     return state

# async def submit_final(state: AgentState) -> AgentState:
#     user_data = state.parsed_data.model_dump()
#     user_data["validated_at"] = datetime.UTC().strftime("%Y-%m-%dT%H:%M:%SZ")

#     OUTPUT_FILE.write_text(json.dumps(user_data, indent=2), encoding="utf-8")
#     logger.info(f"Final JSON saved to {OUTPUT_FILE}")
#     print(json.dumps(user_data, indent=2))

#     print(f"Sending POST request to {SUBMIT_URL}...")
#     try:
#         async with httpx.AsyncClient() as client:
#             response = await client.post(SUBMIT_URL, json=user_data, timeout=10)
#         print(f"POST response status: {response.status_code}")
#         print(f"Response body: {response.text}")
#     except Exception as e:
#         logger.error(f"Failed to submit data: {e}")

#     state.final = True
#     return state



# async def main():
#     initial_state = AgentState(messages=[SystemMessage(content=SYSTEM_PROMPT)])

#     builder = StateGraph(AgentState)
#     builder.add_node("llm", llm_invoke)
#     builder.add_node("user_input", get_user_input)
#     builder.add_node("submit", submit_final)

#     builder.set_entry_point("llm")
#     builder.add_conditional_edges("llm", lambda s: "submit" if s.final else "user_input", {
#         "submit": "submit",
#         "user_input": "user_input"
#     })
#     builder.add_edge("user_input", "llm")
#     builder.set_finish_point("submit")

#     graph = builder.compile()

#     await graph.ainvoke(initial_state, config={"recursion_limit": 50})

# if __name__ == "__main__":
#     asyncio.run(main())


import os
import json
import getpass
import re
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv
from dateutil import parser as date_parser
from typing import TypedDict, Optional, Annotated
 
 
# LangChain & LangGraph
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain import ChatGroq
 
# Pydantic v2
from pydantic import BaseModel, constr, field_validator
 
# PDF & embeddings for RAG
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pypdf.errors import PdfReadError
 
 
# ----------- Data Collection Agent ------------
 
class UserData(BaseModel):
    username: Annotated[str, constr(strip_whitespace=True, min_length=1, max_length=80)]
    dob: str
    address: Annotated[str, constr(strip_whitespace=True, min_length=6)]
    phone: str
 
    @field_validator('username')  
    @classmethod
    def validate_username(cls, v):
       if not re.fullmatch(r"[A-Za-z ]+", v):
           
           raise ValueError("Only letters and spaces allowed")
       return v
 
 
    @field_validator('dob')
    @classmethod
    def dob_valid(cls, v):
        try:
            dt = date_parser.parse(v)
        except Exception:
            raise ValueError("Invalid date format")
        age = (datetime.now() - dt).days // 365
        if not (5 <= age <= 120):
            raise ValueError("Age must be between 5 and 120")
        return dt.strftime("%Y-%m-%d")
 
    @field_validator('phone')
    @classmethod
    def phone_digits_only(cls, v):
        if not re.fullmatch(r"\d{10}", v.strip()):
            raise ValueError("Phone must be 10 digits")
        return v.strip()
 
 
class DataCollectionAgent:
    def __init__(self, llm, output_file="payload.json", submit_url="http://127.0.0.1:8000/docs#/default/create_form_data_form_post"):
        self.llm = llm
        self.output_file = output_file
        self.submit_url = submit_url
 
        self.system_text = """
You are a careful data-collection agent. Your job is to collect and validate the following fields:
 
- username
- dob (date of birth)
- address
- phone
 
RULES:
1. Validate each field using the rules below.
2. If a field is invalid or missing, ask ONE clarifying question.
3. Accept any valid date format for DOB.
4. Once all fields are valid, return only the final JSON — no extra text.
 
SCHEMA:
{
  "username": string,
  "dob": "YYYY-MM-DD",
  "address": string,
  "phone": string
}
 
VALIDATION RULES:
- username: letters and spaces only, max 80 chars, must be real name (not gibberish)
- dob: valid date, age between 5 and 120
- address: minimum 6 characters
- phone: exactly 10 digits
"""
 
        self.messages = [SystemMessage(content=self.system_text)]
        self.final_data = None
        self.finished = False
 
    def parse_output(self, text):
        try:
            return self.parser.parse(text).model_dump()
        except Exception as e:
            return None
 
    def step(self, user_input: str):
        if self.finished:
            return "Data collection is already completed. You can ask questions or type 'exit' to quit."
 
        self.messages.append(HumanMessage(content=user_input))
        resp = self.llm.invoke(self.messages)
        self.messages.append(AIMessage(content=resp.content))
 
        parsed = self.parse_output(resp.content)
        if parsed:
            self.final_data = parsed
            self.finished = True
            self._save_and_submit(parsed)
            return f" Data collection completed:\n{json.dumps(parsed, indent=2)}"
 
        return resp.content
 
    def _save_and_submit(self, data):
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        data["validated_at"] = now
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
 
        print(f"[DataCollectionAgent] Final JSON saved to {self.output_file}")
        try:
            res = requests.post(self.submit_url, json=data, timeout=10)
            print(f"[DataCollectionAgent] POST sent, status: {res.status_code}")
        except Exception as e:
            print(f"[DataCollectionAgent] POST failed: {e}")
 
 
# ----------- PDF RAG Agent ------------
 
PDF_PATH = r"C:\Users\SMBXL GUEST\AgentiAi\tasks\adaptive_rag\knowledge_base\latest_doc.pdf"
persistent_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db", "faiss_index")
 
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"trust_remote_code": True}
)
 
def load_pdf_documents(pdf_path):
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        return docs
    except PdfReadError as e:
        print(f"Error loading PDF file: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error loading PDF: {e}")
        return None
 
if os.path.exists(persistent_directory):
    vectorstore = FAISS.load_local(
        persistent_directory,
        embedding,
        allow_dangerous_deserialization=True
    )
else:
    docs = load_pdf_documents(PDF_PATH)
    if docs is None or len(docs) == 0:
        raise RuntimeError("Failed to load or parse the PDF. Please check the PDF file integrity.")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embedding)
    vectorstore.save_local(persistent_directory)
 
retriever = vectorstore.as_retriever()
 
pdf_llm = ChatGroq(
    api_key="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",
    model="deepseek-r1-distill-llama-70b",
    temperature=0,
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
)
 
pdf_llm.with_structured_output(schema={
    "title": "RAGOutput",
    "description": "Output schema for RAG system with question, context and answer.",
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "context": {"type": "string"},
        "answer": {"type": "string"},
    },
    "required": ["question", "context", "answer"],
})
 
def pdf_rag_single_question(question: str) -> str:
    docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in docs])
 
    if context.strip():
        prompt = f"""You are a helpful assistant. Use the following context to answer.
 
Context:
 
{context}
 
Question: {question}
 
Answer:"""
    else:
        prompt = f"""You are a helpful assistant. Answer the following question:
 
Question: {question}
 
Answer:"""
 
    answer_obj = pdf_llm.invoke(prompt)
 
    if isinstance(answer_obj, dict) and "content" in answer_obj:
        return answer_obj["content"]
    elif hasattr(answer_obj, "content"):
        return answer_obj.content
    else:
        return str(answer_obj)
 
 
# ----------- Main Agent (Router) ------------
 
def route_input(user_input: str, data_agent: DataCollectionAgent) -> str:
    # Keywords to detect data collection intent
    keywords_data = {"username", "dob", "date of birth", "phone", "address", "fill", "update"}
 
    # Keywords to detect RAG intent (basic heuristic)
    keywords_rag = {"who", "what", "where", "when", "how", "tell me", "info about", "information", "explain"}
 
    lower_input = user_input.lower()
 
    # If data collection unfinished, prioritize data collection agent unless user explicitly asks a RAG-style question
    if not data_agent.finished:
        if any(k in lower_input for k in keywords_rag):
            return "RAG"
        else:
            return "DATA_COLLECTION"
 
    # If data collection finished, check if user wants to start new data collection
    if any(k in lower_input for k in keywords_data):
        return "DATA_COLLECTION"
 
    # Otherwise, route to RAG
    return "RAG"
 
 
def main_dispatch_agent():
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        GROQ_API_KEY = getpass.getpass("Enter GROQ_API_KEY: ").strip()
 
    data_llm = ChatGroq(api_key=GROQ_API_KEY, temperature=0, model_name="llama3-70b-8192")
    data_agent = DataCollectionAgent(data_llm)
 
    print("Welcome to the Unified Agent! Type 'exit' to quit.")
    print("You can provide data fields or ask general questions anytime.\n")
 
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
 
        route = route_input(user_input, data_agent)
 
        if route == "DATA_COLLECTION":
            if data_agent.finished:
                print("\n[Starting new data collection session...]\n")
                data_agent = DataCollectionAgent(data_llm)
 
            response = data_agent.step(user_input)
            print(f"\nAgent (Data Collection):\n{response}")
 
        else:  # RAG
            answer = pdf_rag_single_question(user_input)
            print(f"\nAgent:\n{answer}")
 
 
if __name__ == "__main__":
    main_dispatch_agent()
 