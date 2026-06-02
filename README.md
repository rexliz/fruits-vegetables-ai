---
title: Smart Nutrition Advisor
emoji: 🍎
colorFrom: green
colorTo: red
sdk: gradio
sdk_version: 5.20.0
app_file: app.py
pinned: false
---
 
# 🍎🥦 Smart Nutrition Advisor (CV + ML + NLP)
 
Upload a photo of a fruit or vegetable and get instant AI-powered nutrition advice.
 
This Space combines three AI blocks into one integrated application:
- 🔍 **Computer Vision (CLIP):** Identifies the food type using Zero-Shot classification
- 📊 **Machine Learning (Ridge Regression):** Predicts calories from macronutrients
- 💬 **NLP (GPT-4o-mini):** Generates a personalized recipe & health advice
## How it works
1. **CV block** detects the food from the uploaded image (e.g. "apple", 94%)
2. The detected label is used to look up macronutrients, which feed the **ML block**
3. The ML-predicted calories plus the food type are passed to the **NLP block**, which explains the result and suggests a recipe
## Required files
- `app.py`
- `requirements.txt`
- `best_model.pkl`
- `scaler.pkl`
- `nutrition_lookup.json`
- `apple.jpg`, `banana.jpg`, `broccoli.jpg`, `carrot.jpg`
## Configuration
The app expects an `OPENAI_API_KEY` set as a Secret in the Space settings.