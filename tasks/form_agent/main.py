from fastapi import FastAPI, HTTPException
import json
import uuid
import os

app = FastAPI()

DATA_FILE = "data.json"

def read_data():
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE, "r") as f:
        return json.load(f)

def write_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

@app.get("/form")
def get_form_data():
    return read_data()

@app.post("/form")
def create_form_data(entry: dict):
    data = read_data()
    entry["uuid"] =str(uuid.uuid4())
    data.append(entry)
    write_data(data)
    return {"message": "Form data added successfully", "data": entry}

@app.put("/form/{id}")
def update_form_data(id: int, updated_entry: dict):
    data = read_data()
    for i, item in enumerate(data):
        if item.get("uuid") == id:
            updated_entry["uuid"] = id
            data[i] = updated_entry
            write_data(data)
            return{"message" : "Object Field updated successfully", "data" : updated_entry}
    raise HTTPException(status_code=404, detail="Entry not found")

@app.delete("/form/{id}")
def delete_form_data(id: int):
    data = read_data()
    for i, item in enumerate(data):
        if item.get["uuid"] == id:
            removed_entry = data_pop(i)
            write_data(data)
            return {"message" : "Object deleted successfully", "deleted" : removed_entry}
        
    raise HTTPException(status_code=404, detail="Object not found")