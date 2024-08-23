from transformers import pipeline

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
classifier = pipeline("sentiment-analysis", model=model_name)

text = "I love using Hugging Face models!"
result = classifier(text)
print(result)
