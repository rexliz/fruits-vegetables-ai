import os
import json
import numpy as np
import joblib
import gradio as gr
from transformers import pipeline
from openai import OpenAI
from PIL import Image

# ── Load models ──────────────────────────────────────────────────────────────
print("Loading CLIP model...")
clip_classifier = pipeline(
    model="openai/clip-vit-base-patch32",
    task="zero-shot-image-classification"
)

ml_model = joblib.load("../data/best_model.pkl")
scaler = joblib.load("../data/scaler.pkl")

with open("../data/nutrition_lookup.json") as f:
    NUTRITION_LOOKUP = json.load(f)

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ── Labels ────────────────────────────────────────────────────────────────────
ALL_LABELS = list(NUTRITION_LOOKUP.keys())

FRUIT_LABELS = [
    'apple', 'banana', 'orange', 'strawberry', 'grape', 'watermelon',
    'pineapple', 'mango', 'peach', 'pear', 'cherry', 'lemon'
]

# ── Functions ─────────────────────────────────────────────────────────────────
def predict_calories(protein, fat, carbs, fiber, category_code=0):
    fat_x_carbs = fat * carbs
    features = np.array([[protein, fat, carbs, fiber, category_code, fat_x_carbs]])
    features_scaled = scaler.transform(features)
    return round(ml_model.predict(features_scaled)[0], 1)


def generate_advice(food_label, calories, protein, fat, carbs, fiber):
    prompt = f"""You are a professional nutritionist and chef.

A user uploaded an image and the AI detected: **{food_label}**

Nutritional values per 100g:
- Calories: {calories} kcal (predicted by ML model)
- Protein: {protein}g | Fat: {fat}g | Carbs: {carbs}g | Fiber: {fiber}g

Please provide:
1. **Health Benefits** (2-3 sentences)
2. **Best Recipe** (1 simple recipe, max 5 steps)
3. **Serving Tip** (1 short tip)

Keep it friendly and concise."""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.7
    )
    return response.choices[0].message.content


def analyze_food(image):
    if image is None:
        return "No image provided", "", "", ""

    # ── Block 1: CV – Classify food ───────────────────────────────────────────
    results = clip_classifier(image, candidate_labels=ALL_LABELS)
    top3 = results[:3]
    food_label = top3[0]["label"]
    confidence = top3[0]["score"]

    cv_output = f"🔍 **Detected:** {food_label.capitalize()} ({confidence*100:.1f}%)\n\n"
    cv_output += "**Top 3 predictions:**\n"
    for r in top3:
        cv_output += f"- {r['label']}: {r['score']*100:.1f}%\n"
    cv_output += f"\n**Type:** {'🍎 Fruit' if food_label in FRUIT_LABELS else '🥦 Vegetable'}"

    # ── Block 2: ML – Predict calories ────────────────────────────────────────
    if food_label in NUTRITION_LOOKUP:
        nutrition = NUTRITION_LOOKUP[food_label]
    else:
        nutrition = {"protein": 1.0, "fat": 0.2, "carbs": 10.0, "fiber": 2.0}

    category_code = 0 if food_label in FRUIT_LABELS else 1
    calories = predict_calories(
        nutrition["protein"], nutrition["fat"],
        nutrition["carbs"], nutrition["fiber"], category_code
    )

    ml_output = f"📊 **Predicted Calories:** {calories} kcal / 100g\n\n"
    ml_output += "**Macronutrients per 100g:**\n"
    ml_output += f"- 🥩 Protein: {nutrition['protein']}g\n"
    ml_output += f"- 🫒 Fat: {nutrition['fat']}g\n"
    ml_output += f"- 🍞 Carbohydrates: {nutrition['carbs']}g\n"
    ml_output += f"- 🌾 Fiber: {nutrition['fiber']}g\n"

    # ── Block 3: NLP – Generate advice ────────────────────────────────────────
    nlp_output = generate_advice(
        food_label, calories,
        nutrition["protein"], nutrition["fat"],
        nutrition["carbs"], nutrition["fiber"]
    )

    summary = f"✅ **{food_label.capitalize()}** detected with {confidence*100:.1f}% confidence → {calories} kcal/100g"

    return cv_output, ml_output, nlp_output, summary


# ── Gradio Interface ──────────────────────────────────────────────────────────
with gr.Blocks(title="🍎 Fruits & Vegetables AI – Smart Nutrition Advisor") as demo:
    gr.Markdown("""
    # 🍎🥦 Smart Nutrition Advisor
    Upload a photo of a fruit or vegetable and get:
    - 🔍 **CV (CLIP):** Identifies the food type
    - 📊 **ML (Random Forest):** Predicts calories & nutrition
    - 💬 **NLP (GPT-4o-mini):** Generates recipe & health advice
    """)

    with gr.Row():
        image_input = gr.Image(type="pil", label="📸 Upload Fruit or Vegetable Image")

    analyze_btn = gr.Button("🔍 Analyze", variant="primary", size="lg")

    summary_output = gr.Markdown(label="Summary")

    with gr.Row():
        cv_output = gr.Markdown(label="🔍 Block 1: CV – Food Detection")
        ml_output = gr.Markdown(label="📊 Block 2: ML – Nutrition Prediction")

    nlp_output = gr.Markdown(label="💬 Block 3: NLP – Recipe & Health Advice")

    analyze_btn.click(
        fn=analyze_food,
        inputs=image_input,
        outputs=[cv_output, ml_output, nlp_output, summary_output]
    )

    gr.Examples(
        examples=[
            ["example_images/apple.jpg"],
            ["example_images/banana.jpg"],
            ["example_images/broccoli.jpg"],
            ["example_images/carrot.jpg"],
        ],
        inputs=image_input,
        label="Example Images"
    )

demo.launch()

