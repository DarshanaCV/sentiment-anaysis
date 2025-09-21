from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"]
)
model = load_model("sentiment_model.h5")

word_index = imdb.get_word_index()

class Review(BaseModel):
    text: str

def preprocess(text):
    words = text.lower().split()  # simple tokenization
    seq = []
    for w in words:
        idx = word_index.get(w, 2) + 3  # 0-3 reserved in Keras IMDB
        if idx < 10000:  # only top 10k words
            seq.append(idx)
    padded = pad_sequences([seq], maxlen=200)
    return padded

@app.post("/predict")
def predict_sentiment(review: Review):
    seq = preprocess(review.text)
    prediction = model.predict(seq)[0][0]
    sentiment = "positive" if prediction > 0.5 else "negative"
    return {"review": review.text, "sentiment": sentiment, "confidence": float(prediction)}
