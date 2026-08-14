import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from predict import MalayalamNERPredictor

# Initialize FastAPI App
app = FastAPI(
    title="Malayalam Named Entity Recognition (NER) API",
    description="REST API to extract named entities (PER, LOC, ORG, etc.) from Malayalam text using fine-tuned IndicBERTv2/MuRIL.",
    version="1.0.0"
)

# Initialize NER Predictor
predictor = MalayalamNERPredictor()

class TextRequest(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Malayalam NER API is running. Send a POST request to /predict with Malayalam text."}

@app.post("/predict")
def predict_ner(payload: TextRequest):
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
    
    try:
        word_map, entities = predictor.predict(payload.text)
        return {
            "text": payload.text,
            "entities": entities,
            "word_level_tags": word_map
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
