import os
import requests 
import json
import uuid
from langchain_groq import ChatGroq
from pydantic import BaseModel, constr, field_validator, Field
from langchain.output_parsers import PydanticOutputParser
from  typing import TypedDict, Annotated, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pypdf.errors import PdfReadError
from langchain.prompts import PromptTemplate

class Users(BaseModel):
    id : int
    name: str = Field(..., description="Full name of the user")
    age: int = Field(..., description="Age in years")
    address: str = Field(..., description="Address of the user")

class FormCollectionAgent:
    def __init__(self, llm):
        self.llm = llm
        self.final_data = {}
        self.finished = False


# class FormColletionAgent:
#     def __init__(self, llm, output_file = "data.json", endpoint = "http://127.0.0.1:8000/docs#/default/create_form_data_form_post"):
#         self.llm = llm
#         self.output_file = output_file
#         self.endpoint = endpoint
#         self.Finished = False
#         self.final_data = None
#         self.parser = PydanticOutputParser(pydantic_object=Users)
#         self.prompt = PromptTemplate(
#             template = (
#                 "You are a form-filling assistant. Ask questions to collect all required fields.\n"
#                 "When all data is collected, return ONLY a JSON object that matches this schema:\n"
#                 "{format_instructions}\n"
#                 "Conversation so far:\n{history}\n"
#                 "User: {input}"
#             ),
#             input_variables=["history", "input"],
#             partial_variables={"format_instructions" : self.parser.get_format_instructions()},
#         )
#         self.history = ""

#     def step(self, input : str):
#         if self.Finished:
#             return "Data Collection completed."
        
#         self.history += f"User: {input}\n"
#         prompt_text = self.prompt.format(history=self.history, input = input)
#         output = self.llm.invoke(prompt_text)
#         text = output.content.strip()
#         if text.startswith("{") and text.endswith("}"):
#             try:
#                 parsed_data = self.parser.parse(text)
#                 self.final_data = parsed_data.model_dump()
#                 self.Finished = True
#                 return f"Collected data: {self.final_data}"
#             except Exception:
#                 return text
#         else:
#             return text

#     def save_and_send(self):
#         if self.final_data:
#             self.final_data["id"] = str(uuid.uuid4())
#             try:
#                 with open(self.output_file, "r") as f:
#                     existing_data = json.load(f)
#                     if not isinstance(existing_data, list):
#                         existing_data = [existing_data]
                    
#             except (FileNotFoundError, json.JSONDecodeError):
#                 existing_data = []

#             existing_data.append(self.final_data)

#             with open(self.output_file, "w") as f:
#                 json.dump(existing_data, f, indent = 4)

#                 try:
#                     response = requests.post(self.endpoint, json=self.final_data)
#                     print(f"API Response [{response.status_code}: {response.text}]")

#                 except requests.exceptions.RequestException as e:
#                     print(f"Failed to post payload: {e}")

# if __name__ == "__main__":
#     llm = ChatGroq(
#         model = "llama-3.3-70b-versatile",
#         temperature=0,
#         groq_api_key = os.getenv("GROQ_API_KEY")
#     )

#     agent = FormColletionAgent(llm)

#     print("User Data collection chat (type 'exit' to quit)")

#     while not agent.Finished:
#         user_input = input("You: ")
#         if user_input.lower() in ["exit", "quit"]:
#             break

#         response = agent.step(user_input)
#         print(f"Bot: {response}")

#     if agent.Finished:
#         agent.save_and_send()
#         print("Data Saved.")
        

    parser = PydanticOutputParser(pydantic_object=Users)

    # LLM setup
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )


    crud_template = """
    You are a JSON CRUD assistant. Based on the user input, return ONLY a JSON object in this format:
    {{
    "action": "CREATE" | "UPDATE" | "DELETE" | "LIST",
    "data": {{
        ... fields ...
    }}
    }}

    Examples:
    User: Add a new user named Riya, age 21, from Delhi
    Response:
    {{ "action": "CREATE", "data": {{ "name": "Riya", "age": 21, "city": "Delhi" }} }}

    User: Delete the user Riya
    Response:
    {{ "action": "DELETE", "data": {{ "name": "Riya" }} }}

    Now process this request:
    {user_input}
    """

    crud_prompt = PromptTemplate(
        template=crud_template,
        input_variables=["user_input"]
    )

    # File path
    json_file = "data.json"



# Ensure file exists
    try:
        with open(json_file, "x") as f:
            json.dump([], f)
    except FileExistsError:
        pass

    # Function to process LLM response
    def process_user_input(user_input, self):
        # Ask LLM for action
        llm_response = llm.invoke(crud_prompt.format(user_input=user_input)).content
        
        # Try parsing as JSON
        try:
            action_data = json.loads(llm_response)
        except json.JSONDecodeError:
            return f"⚠️ Could not parse: {llm_response}"

        action = action_data.get("action")
        data = action_data.get("data", {})

        # CREATE
        if action == "CREATE":
            # Assign unique id
            data["id"] = str(uuid.uuid4())
            with open(json_file, "r+") as f:
                records = json.load(f)
                records.append(data)
                f.seek(0)
                json.dump(records, f, indent=2)
                f.truncate()
            return f"✅ Added: {data}"

        # UPDATE
        elif action == "UPDATE":
            with open(json_file, "r+") as f:
                records = json.load(f)
                updated = False
                for rec in records:
                    if rec.get("id") == data.get("id") or rec.get("name") == data.get("name"):
                        rec.update({k: v for k, v in data.items() if k != "id"})
                        updated = True
                f.seek(0)
                json.dump(records, f, indent=2)
                f.truncate()
            return "✅ Updated" if updated else "❌ No record found"

        # DELETE
        elif action == "DELETE":
            with open(json_file, "r+") as f:
                records = json.load(f)
                before = len(records)
                records = [r for r in records if r.get("id") != data.get("id") and r.get("name") != data.get("name")]
                f.seek(0)
                json.dump(records, f, indent=2)
                f.truncate()
            return "✅ Deleted" if len(records) < before else "❌ No record found"

        # LIST
        elif action == "LIST":
            with open(json_file, "r") as f:
                records = json.load(f)
            return json.dumps(records, indent=2)

        return f"❌ Unknown action: {action}"
    print("💬 JSON CRUD Chatbot (type 'exit' to quit)")
    

    # ========================
    # Chat Loop
    # ========================
    


class RAGBOT:
    def __init__(self, llm, pdf_path):
        self.llm = llm
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()


        text_Splitter = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 50)
        chunks = text_Splitter.split_documents(docs)


        embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = FAISS.from_documents(chunks, embeddings)


    def answer(self, query):
        retriever = self.vectorstore.as_retriever(search_kwargs={"k" : 3})
        docs = retriever.get_relevant_documents(query)


        context = "\n\n".join([d.page_content for d in docs])
        prompt = f"Use the context below to answer the question. \n\nContext:\n{context}\n\n Question: {query}"
        return self.llm.invoke(prompt)
    

class MainAgent:
    def __init__(self, form_agent, rag_agent, llm):
        self.form_agent = form_agent
        self.rag_agent = rag_agent
    
    def route(self, user_input):
        lower_input = user_input.lower()
        if any(word in lower_input for word in ["create", "add", "update", "delete", "list", "entry", "user", "record", "collect"]):
            return self.form_agent.chat(user_input)
        elif any(word in lower_input for word in ["document", " pdf", "context", "report", "information", "content", "based on document", "knowledge"]):
            query = user_input.replace("document", "").replace("pdf", "").strip()
            return self.rag_agent.answer(query)
        
        else:
            decision_prompt = f"""
You are a router for two AI agents:
1. Form Agent - handles JSON CRUD for user records.
2. RAG Agent - answers questions based on a PDF document.

User query: "{user_input}"

Decide which agent should handle this query.
Respond in JSON as:
{{"agent": "form"}} or {{"agent": "rag"}} only.
"""
            try:
                decision = self.llm.invoke(decision_prompt).content
                if "from" in decision.lower():
                    return self.form_agent.chat(user_input)
                elif "rag" in decision.lower():
                    return self.rag_agent.answer(user_input)
                else:
                    return self.llm.invoke(f"The user said: '{user_input}'. Please respond directly")
                
            except Exception as e:
                return f"Error deciding route: {str(e)}"
            
if __name__ == "__main__":
    from langchain_groq import ChatGroq
    llm = ChatGroq(temperature=0, model_name = "llama-3.3-70b-versatile")
    pdf_path = os.path.abspath(r"C:\Users\SMBXL GUEST\AgentiAi\tasks\form_agent\knowledge_base\latest_doc.pdf")
    form_agent = FormCollectionAgent(llm)
    rag_agent = RAGBOT(llm, pdf_path )
    main_agent = MainAgent(form_agent, rag_agent)

    print("Multi Agent Bot (type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower == "exit":
            break
        response = main_agent.route(user_input)
        print(response)

            





