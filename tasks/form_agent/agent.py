# from pydantic import BaseModel, ValidationError
# import json
# import os

# DATA_FILE = "data.json"

# # ✅ Updated model with address, no email
# class User(BaseModel):
#     first_name: str
#     last_name: str
#     age: int
#     occupation: str
#     address: str

# # Order of fields in the form
# FIELDS = ["first_name", "last_name", "age", "occupation", "address"]


# def load_data():
#     if not os.path.exists(DATA_FILE) or os.path.getsize(DATA_FILE) == 0:
#         return []  # Start with empty list
#     with open(DATA_FILE, "r") as f:
#         return json.load(f)

# def save_data(data):
#     with open(DATA_FILE, "w") as f:
#         json.dump(data, f, indent=2)

# def main():
#     all_users = load_data()
#     form_data = {}
#     field_index = 0

#     print("Fill in the form. You can overwrite previous fields anytime.")

#     while field_index < len(FIELDS):
#         field = FIELDS[field_index]
#         print(f"\nCurrent form data:")
#         for f in FIELDS:
#             print(f"  {f}: {form_data.get(f, '')}")

#         value = input(f"Enter value for '{field}': ").strip()

#         # If the input looks like "field value", treat it as an update to that field
#         if " " in value and value.split()[0] in FIELDS:
#             update_field = value.split()[0]
#             new_value = " ".join(value.split()[1:])
#             form_data[update_field] = new_value
#             print(f"✅ Updated '{update_field}' to '{new_value}'.")
#             continue

#         # Normal entry
#         form_data[field] = value
#         field_index += 1

#     # ✅ Validation and save
#     try:
#         user = User(**form_data)
#         all_users.append(user.dict())
#         save_data(all_users)
#         print("\n✅ Form submitted successfully!")
#     except ValidationError as e:
#         print("❌ Validation error:", e)

# if __name__ == "__main__":
#     main()

import requests

API_BASE = "http://127.0.0.1:8000"

def create_user():
    user_data = {}
    user_data["first_name"] = input("First name: ")
    user_data["last_name"] = input("Last name: ")
    user_data["age"] = int(input("Age: "))
    user_data["occupation"] = input("Occupation: ")
    user_data["address"] = input("Address: ")

    res = requests.post(f"{API_BASE}/users", json=user_data)
    print(res.json())

def view_users():
    res = requests.get(f"{API_BASE}/users")
    users = res.json()
    for i, user in enumerate(users):
        print(f"{i}: {user}")
    return users

def update_user():
    users = view_users()
    if not users:
        print("No users found.")
        return

    idx = int(input("Enter index of user to update: "))
    if idx < 0 or idx >= len(users):
        print("Invalid index.")
        return

    updated_data = {}
    for field, value in users[idx].items():
        new_value = input(f"{field} [{value}]: ") or value
        if field == "age":
            new_value = int(new_value)
        updated_data[field] = new_value

    res = requests.put(f"{API_BASE}/users/{idx}", json=updated_data)
    print(res.json())

def main():
    while True:
        print("\n1. Create User")
        print("2. View Users")
        print("3. Update User")
        print("4. Exit")
        choice = input("Choose: ")
        if choice == "1":
            create_user()
        elif choice == "2":
            view_users()
        elif choice == "3":
            update_user()
        elif choice == "4":
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()


# agent.py
# 

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Annotated
from httpx import HTTPStatusError
from pydantic import BaseModel, ValidationError, constr, field_validator
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain import ChatGroq
import httpx


# ===== CONFIG =====
GROQ_API_KEY = "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"  # Your Groq API key here
MODEL_NAME = "llama-3.3-70b-versatile"
TEMPERATURE = 0.3

DATA_FILE = Path("data.json")
OUTPUT_FILE = Path("final_payload.json")
SUBMIT_URL = "http://127.0.0.1:8000/docs#/default/create_form_data_form_post"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== Pydantic model for validation =====
class FormData(BaseModel):
    name: Annotated[str, constr(strip_whitespace=True, min_length=1)]
    age: int
    email: Annotated[str, constr(strip_whitespace=True, min_length=5)]

    @field_validator("age")
    def validate_age(cls, v):
        if not (5 <= v <= 120):
            raise ValueError("Age must be between 5 and 120")
        return v

    @field_validator("email")
    def validate_email(cls, v):
        if "@" not in v or "." not in v:
            raise ValueError("Invalid email format")
        return v

# ===== Dataclass for Agent State =====
@dataclass
class AgentState:
    messages: list = field(default_factory=list)
    final: bool = False
    parsed_data: Optional[FormData] = None
    form_data: dict = field(default_factory=dict)
    current_field_index: int = 0

# ===== Initialize Groq LLM =====
llm = ChatGroq(api_key=GROQ_API_KEY, model_name=MODEL_NAME, temperature=TEMPERATURE)

# ===== Fields to collect =====
FIELDS = ["name", "age", "email"]

# ===== System prompt for LLM =====
SYSTEM_PROMPT = f"""
You are a helpful assistant that collects user information in a form.
You will ask the user for these fields in order: {', '.join(FIELDS)}.
For each user response, extract the answer for the requested field ONLY.
Validate the input according to these rules:
- name: at least 1 character
- age: integer between 5 and 120
- email: must contain '@' and '.'

If the input is invalid, ask again politely.
When all fields are collected and valid, say "All done."

Respond ONLY with JSON: {{"field": <field_name>, "value": <value>, "error": <error_message or null>}}.
"""

# ===== Utility function to save data =====
def save_data(data):
    existing = []
    if DATA_FILE.exists():
        try:
            existing = json.loads(DATA_FILE.read_text())
        except Exception:
            existing = []
    existing.append(data)
    DATA_FILE.write_text(json.dumps(existing, indent=2))
    logger.info("Saved data to file.")

# ===== Parse LLM response JSON =====
def parse_llm_response(text):
    import re
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        return json.loads(match.group(0))
    except Exception as e:
        logger.warning(f"Failed to parse JSON from LLM: {e}")
        return None
    
async def invoke_llm_with_retry(messages, max_retries=5, base_delay=1):
    retries = 0
    while retries < max_retries:
        try:
            response = llm.invoke(messages)
            return response
        except HTTPStatusError as e:
            if e.response.status_code == 429:
                wait_time = base_delay * (2 ** retries)
                logger.warning(f"Rate limited by Groq API, retrying after {wait_time}s...")
                await asyncio.sleep(wait_time)
                retries += 1
            else:
                raise
    raise RuntimeError("Exceeded max retries for Groq API due to rate limits.")

# ===== Async function to interact with LLM =====
async def ask_field(state: AgentState):
    # If first message, send system prompt
    if not state.messages:
        state.messages.append(SystemMessage(content=SYSTEM_PROMPT))

    # Get the current field to ask
    if state.current_field_index >= len(FIELDS):
        state.final = True
        return state

    field = FIELDS[state.current_field_index]

    # Add user prompt requesting the field
    prompt = f"Please provide your {field}:"
    state.messages.append(HumanMessage(content=prompt))

    # Call the LLM with conversation history
    resp = await invoke_llm_with_retry(state.messages)
    state.messages.append(AIMessage(content=resp.content))

    # Parse response JSON
    parsed = parse_llm_response(resp.content)
    if not parsed:
        print("Sorry, I couldn't understand your response. Let's try again.")
        return state

    error = parsed.get("error")
    value = parsed.get("value")
    resp_field = parsed.get("field")

    if resp_field != field:
        print(f"Unexpected field returned ({resp_field}), expected {field}. Please try again.")
        return state

    if error:
        print(f"Input error: {error}. Please try again.")
        return state

    # Validation with pydantic model
    temp_data = dict(state.form_data)
    temp_data[field] = value
    try:
        FormData(**temp_data)
    except ValidationError as e:
        print(f"Validation failed: {e.errors()[0]['msg']}. Please try again.")
        return state

    # Save the valid input
    state.form_data[field] = value
    print(f"{field} set to: {value}")

    state.current_field_index += 1
    return state

# ===== Async main loop =====
async def main():
    print("Welcome! Let's fill the form step by step.")
    state = AgentState()

    while not state.final:
        state = await ask_field(state)

    print("\nAll data collected successfully:")
    print(json.dumps(state.form_data, indent=2))

    # Save data to file
    save_data(state.form_data)

    # Submit data to API
    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(SUBMIT_URL, json=state.form_data, timeout=10)
            print(f"Submitted data to API. Response status: {r.status_code}")
            print("Response:", r.text)
        except Exception as e:
            print(f"Failed to submit data: {e}")

    print("Thank you!")

if __name__ == "__main__":
    asyncio.run(main())
