"""Tests for the ML pipeline."""

from src.classifier import create_pipeline, evaluate_model, train_model
from src.data_loader import load_data, split_data
from src.preprocessor import clean_text


def test_clean_text():
    assert clean_text("My Payment FAILED!!!") == "payment failed"


def test_load_data():
    df = load_data()
    assert len(df) > 0
    assert "text" in df.columns
    assert "category" in df.columns
    assert "priority" in df.columns


def test_split_data():
    df = load_data()
    X_train, X_test, y_cat_train, y_cat_test, _, _, _, _ = split_data(df)
    assert len(X_train) > len(X_test)


def test_train_and_predict():
    df = load_data()
    X_train, X_test, y_cat_train, y_cat_test, _, _, _, _ = split_data(df)

    model = train_model(X_train, y_cat_train)
    prediction = model.predict(["my payment failed"])[0]
    assert prediction in ["billing", "account", "technical", "shipping"]


def test_evaluate():
    df = load_data()
    X_train, X_test, y_cat_train, y_cat_test, _, _, _, _ = split_data(df)

    model = train_model(X_train, y_cat_train)
    metrics = evaluate_model(model, X_test, y_cat_test)
    assert "accuracy" in metrics
    assert 0 <= metrics["accuracy"] <= 1


def test_sentiment():
    df = load_data()
    X_train, X_test, _, _, _, _, y_sent_train, y_sent_test = split_data(df)

    model = train_model(X_train, y_sent_train)
    prediction = model.predict(["this product is terrible"])[0]
    assert prediction in ["positive", "negative", "neutral"]
