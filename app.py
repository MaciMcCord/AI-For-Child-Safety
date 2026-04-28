import sys
import json
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

from transcript import get_transcript   # ← your transcript.py

# ── Constants (must match python_main.py) ─────────────────────────────────────
MAX_LEN = 100
LABELS  = {0: "Hate Speech", 1: "Offensive Language", 2: "Neutral"}

# ── Load model + tokenizer ────────────────────────────────────────────────────
model = keras.models.load_model("hatespeech_model.keras")

with open("tokenizer.json", "r") as f:
    tokenizer = tokenizer_from_json(json.load(f))

# ── Predict ───────────────────────────────────────────────────────────────────
def predict(video_id: str) -> dict:
    lines = get_transcript(video_id)

    if not lines:
        return {"error": "Transcript was empty after cleaning."}

    sequences = tokenizer.texts_to_sequences(lines)
    padded    = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    probs     = model.predict(padded, verbose=0)
    class_ids = np.argmax(probs, axis=1)

    results = [
        {
            "text":           line,
            "prediction":     LABELS[int(cid)],
            "confidence":     round(float(np.max(prob)), 3),
            "hate_prob":      round(float(prob[0]), 3),
            "offensive_prob": round(float(prob[1]), 3),
            "neutral_prob":   round(float(prob[2]), 3),
        }
        for line, cid, prob in zip(lines, class_ids, probs)
    ]

    counts = {label: 0 for label in LABELS.values()}
    for r in results:
        counts[r["prediction"]] += 1

    return {
        "results": results,
        "summary": counts,
        "total":   len(results)
    }

# ── Entry point (called by server.js) ────────────────────────────────────────
if __name__ == "__main__":
    video_id = sys.argv[1]          # ← server.js passes the video ID here
    output   = predict(video_id)
    print(json.dumps(output))       # ← server.js reads this