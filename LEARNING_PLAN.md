# Learning Plan: Ticket Classifier

## Prerequisites

- Python 3.11+
- Java/Spring Boot background (we'll map ML concepts to Java)
- No API keys needed — everything runs locally

---

## Phase 0: ML Basics for Java Developers (Day 1)

| Step | What to learn | File |
|------|-------------|------|
| 1 | Run Lesson 0 | `learn/00_ml_for_java_devs.py` |
| 2 | What is ML? Training vs coding | Lesson 0 |
| 3 | scikit-learn basics | Lesson 0 |
| 4 | TF-IDF — turning text into numbers | Lesson 0 |

---

## Phase 1: Build the Pipeline (Day 2)

| Step | What to learn | File |
|------|-------------|------|
| 1 | Load and explore data | `src/data_loader.py` |
| 2 | Preprocess text (clean, tokenize) | `src/preprocessor.py` |
| 3 | Train the model | `src/classifier.py` |
| 4 | Run training script | `python -m scripts.train` |
| 5 | Understand metrics (accuracy, precision, recall) | Training output |

---

## Phase 2: Serve and Test (Day 3)

| Step | What to learn | File |
|------|-------------|------|
| 1 | FastAPI endpoint | `api.py` |
| 2 | Start server | `uvicorn api:app --reload` |
| 3 | Test predictions | `curl` / browser |
| 4 | Run unit tests | `pytest` |
| 5 | Understand confusion matrix | Training output |

---

## Commands Cheat Sheet

```bash
# Setup
cd ticket-classifier
source .venv/bin/activate

# Train
python -m scripts.train

# API
uvicorn api:app --reload

# Test
pytest

# Predict
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "My payment failed"}'
```
