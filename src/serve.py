from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
import joblib
import os

app = FastAPI()

GCS_BUCKET = os.environ["GCS_BUCKET"]
GCS_MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")


def download_model():
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(GCS_MODEL_KEY)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    blob.download_to_filename(MODEL_PATH)
    print(f"Model downloaded from gs://{GCS_BUCKET}/{GCS_MODEL_KEY}")


download_model()
model = joblib.load(MODEL_PATH)

LABELS = {0: "thap", 1: "trung_binh", 2: "cao"}


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    if len(req.features) != 12:
        raise HTTPException(status_code=400, detail="Expected 12 features (wine quality)")
    pred = int(model.predict([req.features])[0])
    return {"prediction": pred, "label": LABELS[pred]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
