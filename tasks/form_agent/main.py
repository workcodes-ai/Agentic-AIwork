from fastapi import FastAPI, HTTPException
import json
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
    data.append(entry)
    write_data(data)
    return {"message": "Form data added successfully", "data": entry}

@app.put("/form/{index}")
def update_form_data(index: int, updated_entry: dict):
    data = read_data()
    if index < 0 or index >= len(data):
        raise HTTPException(status_code=404, detail="Entry not found")
    data[index] = updated_entry
    write_data(data)
    return {"message": "Form data updated successfully", "data": updated_entry}

@app.delete("/form/{index}")
def delete_form_data(index: int):
    data = read_data()
    if index < 0 or index >= len(data):
        raise HTTPException(status_code=404, detail="Entry not found")
    removed_entry = data.pop(index)
    write_data(data)
    return {"message": "Form data deleted successfully", "deleted": removed_entry}
