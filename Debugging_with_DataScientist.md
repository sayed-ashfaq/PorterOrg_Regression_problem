# Debugging with Sr.DataScientist



## The Dataset
**Porter Delivery Time Prediction** — a regression problem where we predict delivery time using neural networks. Great choice because it has mixed data types, real-world messiness, and is perfect for end-to-end learning.

---

## The Journey: 10 Modules

## Day 1 Tasks - Completed

**Module 1 — Thinking Like a Data Scientist (Mindset + First Look)**
Before touching any code, we learn to *ask questions* about data. What are we predicting? Who uses this? What does "good" even mean here? Then we do our very first look at the raw data.

**Module 2 — Exploratory Data Analysis (EDA)**
Distributions, outliers, missing values, correlations. Learning to *see* the story the data is telling. We use statistics and visualizations as our detective tools.

**Module 3 — Statistical Analysis**
Hypothesis testing, skewness, kurtosis, understanding distributions deeply. This is where we go beyond charts and start *proving* things about our data.

**Module 4 — Data Cleaning & Preprocessing**
Handling nulls, outliers, duplicates, and data type issues. The unglamorous work that makes or breaks a model.

**Module 5 — Feature Engineering**
Extracting datetime features, creating new meaningful variables, encoding categoricals. Turning raw columns into signals a model can learn from.

## Day 2 Task list

**Module 6 — Preparing for Modeling**
Train/validation/test splits, scaling, pipelines. Making sure we're doing this *honestly* with no data leakage.

**Module 7 — Baseline Models**
Before a neural network, we start simple — linear regression, tree-based models. You can't know if your NN is good without a baseline.

**Module 8 — Building the Neural Network**
Designing architecture, choosing loss functions, activation functions, optimizers. Understanding *why* each choice, not just what to type.

**Module 9 — Training, Evaluation & Tuning**
Learning curves, overfitting, regularization, hyperparameter tuning. Making the model actually *good*.

**Module 10 — Deployment (FastAPI + TensorFlow Serving)** 
Save model, build inference pipeline, wrap in a REST API with FastAPI.

**Module 11 — AI Integration**
Build a thin LLM layer on top using the Anthropic API — user describes a delivery in natural language, LLM parses it into features, your model predicts, LLM explains the result back. A genuinely useful and modern pattern.