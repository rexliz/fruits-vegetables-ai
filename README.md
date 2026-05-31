# 🍎🥦 Smart Nutrition Advisor – Fruits & Vegetables AI

A smart nutrition advisor that identifies fruits & vegetables from photos and provides personalized nutrition advice.

## 🎯 Use Case
📸 Upload a photo → 🖼️ CV identifies food → 📊 ML predicts calories → 💬 NLP generates recipe & health tips

## 🔗 AI Blocks
| Block | Technology | Task |
|-------|-----------|------|
| **CV** | CLIP (Zero-Shot) | Classify fruit/vegetable from image |
| **ML** | Ridge Regression (R²=0.97) | Predict calories per 100g |
| **NLP** | GPT-4o-mini | Generate recipes & health advice |

## 🚀 Links
- **GitHub:** https://github.com/rexliz/fruits-vegetables-ai
- **HF Space:** https://huggingface.co/spaces/rexheliz/fruits-vegetables-nutrition-advisor

## 📦 Dataset
- USDA Nutritional Data (nutrition.csv, 332 items after cleaning)

## ▶️ How to Run
```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file with your OpenAI API key
echo "OPENAI_API_KEY=your_key_here" > .env

# Run notebooks in order (1 → 2 → 3 → 4)

# Run Gradio App
cd app
python app.py
```

## 📁 Structure
```
1_eda/          → EDA + data cleaning
2_ml_model/     → Calorie prediction model
3_cv/           → CLIP Zero-Shot classification
4_nlp/          → GPT-4o-mini recipe advisor
app/            → Gradio web app
data/           → nutrition.csv + generated artifacts
```
