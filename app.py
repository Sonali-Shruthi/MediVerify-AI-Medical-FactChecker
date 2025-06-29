from flask import Flask, render_template, request, jsonify




# Load spaCy model
import spacy
import requests
from bs4 import BeautifulSoup
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import asyncio
from scholarly import scholarly
from transformers import pipeline


app = Flask(__name__)
# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load Hugging Face summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Load BioBERT tokenizer & model
biobert_model = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(biobert_model)
model = AutoModel.from_pretrained(biobert_model)

def summarize_text(text):
    summary = summarizer(text, max_length=80, min_length=40, do_sample=False)
    summarized_text = summary[0]['summary_text']
    print("\n[STEP 0] Summarized Input:", summarized_text)
    return summarized_text

def get_embedding(text):
    tokens = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)

    with torch.no_grad():
        output = model(**tokens)

    sentence_embedding = output.last_hidden_state.mean(dim=1).numpy()
    return sentence_embedding

def fetch_pubmed_articles(query):
    url = f"https://pubmed.ncbi.nlm.nih.gov/?term={query.replace(' ', '+')}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    articles = []
    for result in soup.find_all("div", class_="docsum-content")[:5]:
        title = result.find("a", class_="docsum-title").text.strip()
        link = "https://pubmed.ncbi.nlm.nih.gov" + result.find("a", class_="docsum-title")["href"]
        summary = result.find("div", class_="full-view-snippet").text.strip() if result.find("div", class_="full-view-snippet") else "No summary available."
        articles.append({"title": title, "link": link, "summary": summary})

    return articles

def fetch_scholar_articles(query):
    articles = []
    search_query = scholarly.search_pubs(query)

    for paper in search_query:
        title = paper['bib'].get('title', 'No title available')
        link = paper.get('pub_url', 'No link available')
        summary = paper['bib'].get('abstract', 'No summary available.')
        articles.append({"title": title, "link": link, "summary": summary})

        if len(articles) >= 5:
            break

    return articles



def fetch_articles(query):
    articles = fetch_pubmed_articles(query)

    if not articles:
        print("\n[INFO] No PubMed results found. Fetching from Google Scholar...")
        articles = fetch_scholar_articles(query)  # Call the function synchronously

    return articles


def extract_keywords(text):
    doc = nlp(text)
    keywords = " ".join([token.lemma_ for token in doc if token.pos_ in ["NOUN", "PROPN", "ADJ"]])
    print("\n[STEP 5] Extracted Keywords:", keywords)
    return keywords

def evaluate_response(ai_response):
    summarized_text = summarize_text(ai_response)
    keywords = extract_keywords(summarized_text)

    articles = fetch_articles(keywords)
    if not articles:
        return {"verdict": "Insufficient Data", "score": 0, "articles": []}

    response_embedding = get_embedding(summarized_text)
    article_embeddings = [get_embedding(article["summary"]) for article in articles]

    similarities = [cosine_similarity(response_embedding, emb.reshape(1, -1))[0][0] for emb in article_embeddings]
    print("\n[STEP 6] Similarity Scores:", similarities)

    sorted_articles = sorted(zip(articles, similarities), key=lambda x: x[1], reverse=True)[:3]
    avg_similarity = np.mean([sim for _, sim in sorted_articles])

    negation_present = any(word in summarized_text.lower() for word in ["not", "no", "n't", "don't", "doesn't", "didn't"])
    print("[STEP 7] Negation Present:", negation_present)

    if avg_similarity > 0.85:
        verdict = "False" if negation_present else "True"
    elif avg_similarity <= 0.70:
        verdict = "Likely False" if negation_present else "Likely True"
    elif avg_similarity > 0.50:
        verdict = "Likely False" if negation_present else "Likely False"
    else:
        verdict = "True" if negation_present else "False"

    top_articles = [{"title": art["title"], "link": art["link"], "summary": art["summary"], "similarity": sim} for art, sim in sorted_articles]

    return {"verdict": verdict, "score": round(avg_similarity * 100, 2), "articles": top_articles}

# Example usage
ai_generated_text = "Penicillin is an effective antibiotic used to treat strep throat, a bacterial infection caused by Streptococcus pyogenes. It works by targeting the bacterial cell wall, ultimately killing the bacteria and stopping the infection. Doctors often prescribe a 10-day course of penicillin or amoxicillin to fully eliminate the bacteria and prevent complications like rheumatic fever. Penicillin remains the first-line treatment for strep throat due to its effectiveness, safety, and low cost."
result = evaluate_response(ai_generated_text)

print("\nVerdict:", result["verdict"])
print("Confidence Score:", result["score"])
print("\nTop 3 Articles:")
for i, article in enumerate(result["articles"], 1):
    print(f"{i}. {article['title']} (Similarity: {article['similarity']:.2f})")
    print(f"   {article['summary']}")
    print(f"   Link: {article['link']}\n")


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input = request.form["claim"]
        result = evaluate_response(user_input)
        return render_template("index.html", claim=user_input, result=result)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
