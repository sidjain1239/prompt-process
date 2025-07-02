from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

classifier = pipeline("text-classification", model="./distilbert_ai_router", tokenizer="./distilbert_ai_router")

class PromptInput(BaseModel):
    prompt: str

@app.get("/")
def read_root():
    return {"message": "FastAPI app is running"}

@app.post("/predict")
def predict(data: PromptInput):
    return classifier(data.prompt)
