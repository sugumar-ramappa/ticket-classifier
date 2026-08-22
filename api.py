"""FastAPI server for ticket classification."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.classifier import load_model

app = FastAPI(title="Ticket Classifier API")


class TicketRequest(BaseModel):
    text: str


class TicketResponse(BaseModel):
    text: str
    category: str
    priority: str
    sentiment: str
    category_confidence: float
    priority_confidence: float
    sentiment_confidence: float


@app.on_event("startup")
def load_models():
    global cat_model, pri_model, sent_model
    try:
        cat_model = load_model("category_classifier")
        pri_model = load_model("priority_classifier")
        sent_model = load_model("sentiment_classifier")
    except FileNotFoundError:
        cat_model = None
        pri_model = None
        sent_model = None


@app.post("/predict", response_model=TicketResponse)
def predict(req: TicketRequest):
    if cat_model is None:
        raise HTTPException(status_code=503, detail="Models not trained. Run: python -m scripts.train")

    cat_pred = cat_model.predict([req.text])[0]
    cat_proba = max(cat_model.predict_proba([req.text])[0])
    pri_pred = pri_model.predict([req.text])[0]
    pri_proba = max(pri_model.predict_proba([req.text])[0])
    sent_pred = sent_model.predict([req.text])[0]
    sent_proba = max(sent_model.predict_proba([req.text])[0])

    return TicketResponse(
        text=req.text,
        category=cat_pred,
        priority=pri_pred,
        sentiment=sent_pred,
        category_confidence=round(cat_proba, 3),
        priority_confidence=round(pri_proba, 3),
        sentiment_confidence=round(sent_proba, 3),
    )


@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": cat_model is not None}
