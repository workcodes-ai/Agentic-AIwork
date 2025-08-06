from fastapi import FastAPI
app = FastAPI()

@app.get("/")
def get_const():
    return {"Message" : "Reading root"}