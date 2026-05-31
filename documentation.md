# 🍎 Smart Nutrition Advisor – Project Documentation

## Project Overview

**Project Title:** Smart Nutrition Advisor – Fruits & Vegetables Recognition and Nutrition Advice  
**Student:** rexliz (ZHAW, 6th Semester Business Information Systems)  
**Module:** KI-Anwendungen  
**Submission Date:** 07. June 2026  
**GitHub Repository:** https://github.com/rexliz/fruits-vegetables-ai  
**Hugging Face Space:** https://huggingface.co/spaces/rexheliz/fruits-vegetables-nutrition-advisor  

---

## 1. Project Idea & Motivation

The **Smart Nutrition Advisor** combines three AI techniques in a single pipeline:

1. 📸 User uploads a photo of a fruit or vegetable  
2. 🖼️ **Computer Vision (CLIP)** identifies the food item  
3. 📊 **Machine Learning (Random Forest / Ridge)** predicts calorie content  
4. 💬 **NLP (GPT-4o-mini)** generates personalized recipes and health tips  

**Motivation:** Making nutrition information accessible and fun through AI – just take a photo and get instant advice.

---

## 2. Dataset

**Source:** USDA Nutritional Data  
**File:** `data/nutrition.csv`  
**Size:** 335 food items, 10 columns  
**Columns:** Food, Measure, Grams, Calories, Protein, Fat, Sat.Fat, Fiber, Carbs, Category  

**Preprocessing** (see `1_eda/eda_nutrition.ipynb`):
- Replaced trace values ('t') with 0
- Removed commas from numbers and converted to float
- Normalized all nutritional values to per 100g
- Dropped 3 rows with missing critical values
- Final clean dataset: 332 rows → `data/nutrition_clean.csv`

---

## 3. Machine Learning Model (Block 2)

**Notebook:** `2_ml_model/ml_calorie_prediction.ipynb`  
**Task:** Regression – predict calories per 100g  
**Features:** protein, fat, carbs, fiber (all per 100g), category_encoded, fat×carbs (interaction)  
**Target:** calories_per_100g  

### Model Comparison

| Model | Iteration | CV R² | Test R² | RMSE |
|-------|-----------|-------|---------|------|
| Linear Regression | 1 | 0.9021 | 0.9684 | 33.74 |
| Random Forest (100 trees) | 1 | 0.9106 | 0.9578 | 38.93 |
| Ridge (α=1.0) | 2 | 0.9005 | **0.9655** | 35.25 |
| Tuned RF (300 trees, depth=15) | 2 | 0.9130 | 0.9598 | 38.02 |

**Best Model:** Ridge Regression (R² = 0.9655, RMSE = 35.25 kcal/100g)  

**Why Ridge?** Ridge regression showed the best test R² with excellent generalization. The interaction feature fat×carbs added in Iteration 2 improved stability.

**Saved artifacts:**
- `data/best_model.pkl` – Ridge regression model
- `data/scaler.pkl` – StandardScaler
- `data/feature_names.pkl` – feature names list

---

## 4. Computer Vision Model (Block 3)

**Notebook:** `3_cv/cv_classification.ipynb`  
**Model:** CLIP (openai/clip-vit-base-patch32) – Zero-Shot Classification  
**No training required!** CLIP uses pre-trained vision-language embeddings.  

**24 classes:**
- 🍎 **Fruits (12):** apple, banana, orange, strawberry, grape, watermelon, pineapple, mango, peach, pear, cherry, lemon  
- 🥦 **Vegetables (12):** carrot, broccoli, tomato, cucumber, pepper, spinach, potato, onion, garlic, lettuce, corn, mushroom  

**Output:** `data/nutrition_lookup.json` – macronutrient lookup table (bridge between CV and ML)

---

## 5. NLP (Block 4)

**Notebook:** `4_nlp/nlp_recipe_advisor.ipynb`  
**Model:** GPT-4o-mini (OpenAI API)  

**Prompt Engineering Iterations:**

| Iteration | Prompt Style | Output |
|-----------|-------------|--------|
| 1 | Simple: "Give me a recipe for {food} with {calories} kcal" | Basic recipe, no structure |
| 2 | Structured: Role + Nutritional data + 3 specific sections requested | Formatted: Health Benefits + Recipe + Serving Tip |

**Iteration 2 is better** because:
- Assigns a professional role to the model ("You are a nutritionist and chef")
- Provides full nutritional context (all macros)
- Requests structured output → more consistent and useful responses

---

## 6. Gradio App (Block 5)

**File:** `app/app.py`  
**Framework:** Gradio  

**Flow:**
1. User uploads image → CLIP classifies food item
2. Food label → lookup macros → Ridge model predicts calories
3. Calories + macros + label → GPT-4o-mini generates advice
4. App displays: CV result, ML prediction, NLP advice

**Example images:** `app/example_images/` (apple, banana, broccoli, carrot)

---

## 7. Setup & Reproduction

```bash
# Clone repository
git clone https://github.com/rexliz/fruits-vegetables-ai.git
cd fruits-vegetables-ai

# Install dependencies
pip install -r requirements.txt

# Create .env file
echo "OPENAI_API_KEY=your_key_here" > .env

# Run notebooks in order:
# 1. 1_eda/eda_nutrition.ipynb       → generates nutrition_clean.csv
# 2. 2_ml_model/ml_calorie_prediction.ipynb → generates best_model.pkl, scaler.pkl
# 3. 3_cv/cv_classification.ipynb    → generates nutrition_lookup.json
# 4. 4_nlp/nlp_recipe_advisor.ipynb  → tests NLP

# Run app
cd app
python app.py
```

---

## 8. Reflection

**What worked well:**
- CLIP Zero-Shot classification works surprisingly well without any training
- Ridge regression achieves R² > 0.96 on nutritional data (fat and carbs are strong predictors)
- GPT-4o-mini provides high-quality, structured advice with the right prompt

**Challenges:**
- Nutritional data has some noise (trace values, different portion sizes)
- CLIP confidence can be low for similar-looking items (e.g., apple vs. pear)
- Normalizing data per 100g was important for fair ML comparison

**Limitations:**
- CV works best with clear, isolated food images (not mixed dishes)
- The nutrition lookup uses fixed typical values, not actual measured values
- NLP responses vary by temperature setting

---

## 9. File Structure

```
fruits-vegetables-ai/
├── 1_eda/
│   └── eda_nutrition.ipynb        # EDA + data cleaning → nutrition_clean.csv
├── 2_ml_model/
│   └── ml_calorie_prediction.ipynb # ML training → best_model.pkl, scaler.pkl
├── 3_cv/
│   └── cv_classification.ipynb    # CLIP Zero-Shot → nutrition_lookup.json
├── 4_nlp/
│   └── nlp_recipe_advisor.ipynb   # GPT-4o-mini recipe generator
├── app/
│   ├── app.py                     # Gradio app
│   ├── requirements.txt
│   └── example_images/            # apple.jpg, banana.jpg, broccoli.jpg, carrot.jpg
├── data/
│   ├── nutrition.csv              # Raw USDA data
│   ├── nutrition_clean.csv        # Cleaned data (332 rows)
│   ├── best_model.pkl             # Trained Ridge model
│   ├── scaler.pkl                 # StandardScaler
│   ├── feature_names.pkl          # Feature names
│   └── nutrition_lookup.json      # Macros per food item (24 classes)
├── .env                           # API keys (NOT in git)
├── .gitignore
├── requirements.txt
├── documentation.md
└── README.md
```

