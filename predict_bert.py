from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

def predict_news(text):
    result = classifier(text)[0]
    label = result['label']
    score = result['score']

    if label == "NEGATIVE":
        return "Fake News", score
    else:
        return "Real News", score
