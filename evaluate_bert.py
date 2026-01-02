import pandas as pd
from transformers import pipeline 
from sklearn.metrics import classification_report, accuracy_score

model_path = "bert_model"
classifier = pipeline("text-classification", model=model_path)

df = pd.read_csv("dataset.csv")

predictions = []
for text in df["text"]:
    result = classifier(text)[0]
    label = int(result["label"].split("_")[-1])
    predictions.append(label)

print("Accuracy:", accuracy_score(df["label"], predictions))
print(classification_report(df["label"], predictions))
