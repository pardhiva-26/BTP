# AI-Generated Misinformation Detection using Multimodal Fact Checking

## Project Overview

This project focuses on evaluating the robustness of multimodal fact-checking systems against synthetically generated misinformation. The work builds upon an end-to-end multimodal fact-checking framework consisting of evidence retrieval, claim verification, and explanation generation.

Traditional misinformation detection systems are generally trained on real-world fact-checking datasets containing claims, evidence, and veracity labels. However, with the emergence of large language models, highly realistic synthetic misinformation can now be generated at scale, including fabricated claims, persuasive news reports, official-sounding statements, temporal details, and contextual descriptions.

The objective of this project is to simulate such AI-generated misinformation scenarios and evaluate how an existing multimodal fact-checking pipeline performs under these adversarial conditions.

---

## Project Objectives

- Study multimodal misinformation detection using retrieval, verification, and explanation generation.
- Generate synthetic fake news claims using prompting strategies.
- Simulate realistic misinformation campaigns with AI-generated narratives.
- Evaluate the robustness of fact-checking pipelines against synthetic misinformation.
- Compare performance between real benchmark data and synthetic adversarial data.
- Analyse weaknesses in retrieval, verification, and explanation generation modules.

---

## System Architecture

The overall system consists of three major components:

### 1. Evidence Retrieval
This module retrieves relevant evidence for a given claim from multimodal information sources.

**Text Retrieval**
- Semantic text retrieval
- Dense embedding similarity search
- Re-ranking for improved relevance

**Image Retrieval**
- Cross-modal text-image retrieval
- Vision-language embedding matching

**Evaluation Metrics**
- Recall@K
- Precision@K
- NDCG
- MAP
- Success Recall

---

### 2. Claim Verification
This module performs stance verification between the claim and retrieved evidence.

Possible outputs:
- Supported
- Refuted
- Not Enough Information (NEI)

The verification component uses multimodal fusion between textual and visual evidence for stance prediction.

**Evaluation Metrics**
- F1 Score
- Precision
- Recall
- Confusion Matrix Analysis

---

### 3. Explanation Generation
This module generates human-readable explanations for the predicted verdict.

Input:
- Claim
- Retrieved evidence
- Predicted truthfulness label

Output:
- Fact-check style explanation

The explanation generator is implemented using transformer-based sequence-to-sequence generation.

**Evaluation Metrics**
- ROUGE-1
- ROUGE-2
- ROUGE-L
- BLEU
- BERTScore

---

## Synthetic Dataset Generation

A synthetic misinformation dataset was created to evaluate model robustness.

### Dataset Creation Strategy

Synthetic claims were generated using structured prompting with large language models.

Prompting objectives included:
- realistic fake news claim creation
- inclusion of dates, locations, and timelines
- official-sounding quotes
- expert statements
- government responses
- professional journalistic tone
- persuasive misinformation narratives

This setup simulates:
- AI-generated propaganda
- coordinated misinformation campaigns
- fabricated journalism
- persuasive synthetic media content

---
