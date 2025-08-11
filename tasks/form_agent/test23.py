import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TypedDict as dict
import nltk
nltk.download('names')
from nltk.corpus import names
from dateutil import parser as date_parser
from dotenv import load_dotenv
from pydantic import BaseModel, constr, ValidationError, field_validator
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph
import httpx 


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


OUTPUT_FILE = Path("final_payload.json")
MODEL_NAME = "llama3-70b-8192"
TEMPERATURE = 0
SUBMIT_URL = "http://127.0.0.1:8000/receive_data"

# all_names = names.words()
# print(all_names[:7944])  # Sample output, e.g. ['Abel', 'Aaron', 'Adrian', ...]
# print(len(all_names))  # Number of names in total



load_dotenv()
import os
import getpass

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    GROQ_API_KEY = getpass.getpass("Enter GROQ_API_KEY: ").strip()
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not provided.")

llm = ChatGroq(api_key=GROQ_API_KEY, model_name=MODEL_NAME, temperature=TEMPERATURE)


try:
    nltk.data.find("corpora/names")
except LookupError:
    nltk.download("names")

KNOWN_NAMES = set(n.lower() for n in nltk.corpus.names.words())
KNOWN_NAMES.update({
    "kumar", "akula", "leela", "ravi", "ram", "sita", "lakshmi",
    "priya", "sai", "mahesh", "rajesh", "suresh", "anil", "harsha", "naveen", "venkat"
})


class UserData(BaseModel):
    username: constr(strip_whitespace=True, min_length=1, max_length=80)
    dob: str
    address: constr(strip_whitespace=True, min_length=6)
    phone: constr(strip_whitespace=True)


    @field_validator("username")
    def validate_username(cls, v):
        if not all(ch.isalpha() or ch.isspace() for ch in v):
            raise ValueError("Username must contain letters and spaces only")
        words = v.lower().split()
       
        return v

    @field_validator("dob")
    def validate_dob(cls, v):
        try:
            dt = date_parser.parse(v)
        except Exception:
            raise ValueError("Invalid date format")
        now = datetime.now(timezone.utc)
        age = (now - dt).days / 365.25
        if not (5 <= age <= 120):
            raise ValueError("Age must be between 5 and 120")
        return dt.strftime("%Y-%m-%d")

@dataclass
class AgentState:
    messages: list = field(default_factory=list)
    final: bool = False
    parsed_data: Optional[UserData] = None


SYSTEM_PROMPT = """
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



def extract_json_from_text(text: str) -> Optional[dict]:
    import re
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        return data
    except Exception as e:
        logger.warning(f"Failed to parse JSON: {e}")
        return None

async def llm_invoke(state: AgentState) -> AgentState:
    logger.info("Calling LLM...")
    
    resp = llm.invoke(state.messages)
    state.messages.append(AIMessage(content=resp.content))
    data = extract_json_from_text(resp.content)
    if data:
        try:
            user_data = UserData(**data)
            state.parsed_data = user_data
            state.final = True
            logger.info("Valid data received from LLM")
        except ValidationError as e:
            logger.info(f"Validation error in LLM output: {e}")
            state.final = False
    else:
        state.final = False
    return state


async def get_user_input(state: AgentState) -> AgentState:
    last_msg = state.messages[-1].content.strip()
    print(f"\nLLM → {last_msg}")
    while True:
        user_input = input("Your reply (or 'exit'): ").strip()
        if user_input.lower() == "exit":
            logger.info("User requested exit.")
            exit(0)
        if user_input.lower() == "submit":
            print("Cannot manually submit. Please answer the question.")
            continue
        break
    state.messages.append(HumanMessage(content=user_input))
    return state

async def submit_final(state: AgentState) -> AgentState:
    user_data = state.parsed_data.model_dump()
    user_data["validated_at"] = datetime.UTC().strftime("%Y-%m-%dT%H:%M:%SZ")

    OUTPUT_FILE.write_text(json.dumps(user_data, indent=2), encoding="utf-8")
    logger.info(f"Final JSON saved to {OUTPUT_FILE}")
    print(json.dumps(user_data, indent=2))

    print(f"Sending POST request to {SUBMIT_URL}...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(SUBMIT_URL, json=user_data, timeout=10)
        print(f"POST response status: {response.status_code}")
        print(f"Response body: {response.text}")
    except Exception as e:
        logger.error(f"Failed to submit data: {e}")

    state.final = True
    return state



async def main():
    initial_state = AgentState(messages=[SystemMessage(content=SYSTEM_PROMPT)])

    builder = StateGraph(AgentState)
    builder.add_node("llm", llm_invoke)
    builder.add_node("user_input", get_user_input)
    builder.add_node("submit", submit_final)

    builder.set_entry_point("llm")
    builder.add_conditional_edges("llm", lambda s: "submit" if s.final else "user_input", {
        "submit": "submit",
        "user_input": "user_input"
    })
    builder.add_edge("user_input", "llm")
    builder.set_finish_point("submit")

    graph = builder.compile()

    await graph.ainvoke(initial_state, config={"recursion_limit": 50})

if __name__ == "__main__":
    asyncio.run(main())
