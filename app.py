from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = FastAPI()

# Load label encoder
df = pd.read_csv("ai_service_all_classes_540.csv")
le = LabelEncoder()
le.fit(df["service"])

# Load pipeline
classifier = pipeline(
    "text-classification",
    model="./distilbert_ai_router",
    tokenizer="./distilbert_ai_router",
    framework="pt",
    device=-1  # CPU
)

class Prompt(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "AI Prompt Classifier API is up."}

@app.post("/predict")
def predict_service(prompt: Prompt):
    result = classifier(prompt.text)[0]
    predicted_id = int(result["label"].split("_")[-1])
    predicted_label = le.inverse_transform([predicted_id])[0]
    return {"service": predicted_label}
