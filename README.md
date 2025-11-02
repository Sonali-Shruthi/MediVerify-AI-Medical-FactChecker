# 🩺 MediVerify — AI-Powered Medical Claim Fact-Checking System

MediVerify is an **AI/NLP-based medical fact-checking system** designed to combat online health misinformation. It automatically verifies medical claims using trusted biomedical sources such as **PubMed**, **WHO**, and **ClinicalTrials**, producing concise, evidence-based reports with multilingual support.

This project was developed as part of a research initiative to enhance **trust and reliability in digital healthcare information** through advanced Natural Language Processing (NLP).

---

## 🚀 Overview

MediVerify performs **end-to-end claim verification** by analyzing a medical statement and evaluating its truthfulness using published biomedical literature.  

It combines **claim understanding**, **evidence retrieval**, **document ranking**, and **summarization** to provide users with accurate, explainable, and multilingual responses.

---

## 🧠 Key Features

- **Intelligent Claim Verification:** Detects and validates medical claims using contextual embeddings and similarity scoring.  
- **Evidence Retrieval from Trusted Sources:** Pulls supporting and refuting literature from PubMed, WHO, and ClinicalTrials APIs.  
- **Advanced NLP Pipeline:** Integrates **BioBERT**, **BERT**, **spaCy**, and **BART** for entity extraction, classification, and summarization.  
- **Multilingual Support:** Translates claims and reports into multiple languages for global accessibility.  
- **Evidence-Based Summarization:** Generates concise summaries explaining whether a claim is *supported*, *refuted*, or *inconclusive*.  
- **Performance:** Achieved **85–90% precision**, reducing average claim verification time from **7–10 minutes** (manual search) to **under a minute**.

---

## 🧩 System Architecture

**Claim Input**  
⬇️  
**Preprocessing (spaCy)**  
⬇️  
**Semantic Encoding (BioBERT)**  
⬇️  
**Evidence Retrieval (PubMed / WHO / ClinicalTrials)**  
⬇️  
**Claim Classification (BERT) and Confidence Score Calculation**  
⬇️  
**Summarization (BART)**  
⬇️  
**Multilingual Translation**
⬇️  
**Evidence Report Generation**


---

## 🧪 Core Components

- **Claim Preprocessor** : Cleans, tokenizes, and extracts key medical entities using spaCy. 
- **Retriever** : Fetches biomedical literature using PubMed, WHO, and ClinicalTrials APIs. 
- **Encoder** : Converts text into semantic embeddings using BioBERT for contextual matching. 
- **Classifier** : Determines claim status — *Supported*, *Refuted*, or *Inconclusive* — using fine-tuned BERT. 
- **Summarizer** : Generates concise, human-readable summaries of evidence using BART. 
- **Translator** : Provides multilingual translation of both claims and summaries. 
- **Report Generator** : Produces structured, explainable verification reports. 

---

## 💡 Future Enhancements

- Integration with **LLMs** (e.g., GPT-based models) for enhanced reasoning.  
- Expansion of **multilingual database coverage**.  
- Browser extension for **real-time fact-checking** of medical tweets, posts, and articles.  


