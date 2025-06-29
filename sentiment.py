from transformers import pipeline

# Load the model
model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

while True:
    text = input("\nEnter a sentence (or type 'exit' to quit): ")
    if text.lower() == "exit":
        print("Goodbye!")
        break
    result = model(text)
    print(result)
