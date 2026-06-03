import os
import json
import numpy as np
import joblib
import gradio as gr
from transformers import pipeline
from openai import OpenAI
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
IMAGE_DIR = os.path.join(BASE_DIR, "app", "example_images")

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading CLIP model...")
clip_classifier = pipeline(
    model="openai/clip-vit-base-patch32",
    task="zero-shot-image-classification"
)
print("✅ CLIP loaded!")

ml_model = joblib.load(os.path.join(DATA_DIR, "best_model.pkl"))
scaler   = joblib.load(os.path.join(DATA_DIR, "scaler.pkl"))

with open(os.path.join(DATA_DIR, "nutrition_lookup.json")) as f:
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
    fat_x_carbs     = fat * carbs
    features        = np.array([[protein, fat, carbs, fiber, category_code, fat_x_carbs]])
    features_scaled = scaler.transform(features)
    return round(ml_model.predict(features_scaled)[0], 1)


def generate_advice(food_label, calories, protein, fat, carbs, fiber):
    """Iteration 3 prompt: system message + grounding + strict format + temperature 0.3"""
    system_msg = (
        "You are a certified nutritionist and chef. "
        "Always base your nutritional advice strictly on the values provided. "
        "Do not invent or estimate nutritional data. "
        "Be concise, factual, and practical."
    )
    user_msg = f"""The AI system detected: {food_label}

Measured nutritional values per 100g:
- Calories: {calories} kcal
- Protein: {protein}g | Fat: {fat}g | Carbohydrates: {carbs}g | Fiber: {fiber}g

Respond EXACTLY in this format:

HEALTH BENEFITS:
[2 evidence-based sentences referencing the values above]

RECIPE: [Name]
Ingredients: [max 5 items]
Steps:
1. [step]
2. [step]
3. [step]

SERVING TIP:
[1 practical tip]"""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg}
        ],
        max_tokens=400,
        temperature=0.3
    )
    return response.choices[0].message.content


def analyze_food(image):
    if image is None:
        return "No image provided", "", "", ""

    # ── Block 1: CV ───────────────────────────────────────────────────────────
    results    = clip_classifier(image, candidate_labels=ALL_LABELS)
    top3       = results[:3]
    food_label = top3[0]["label"]
    confidence = top3[0]["score"]

    cv_output  = f"🔍 **Detected:** {food_label.capitalize()} ({confidence*100:.1f}%)\n\n"
    cv_output += "**Top 3 predictions:**\n"
    for r in top3:
        cv_output += f"- {r['label'].capitalize()}: {r['score']*100:.1f}%\n"
    cv_output += f"\n**Type:** {'🍎 Fruit' if food_label in FRUIT_LABELS else '🥦 Vegetable'}"
    if confidence < 0.70:
        cv_output += "\n\n⚠️ *Low confidence — result may be unreliable*"

    # ── Block 2: ML ───────────────────────────────────────────────────────────
    nutrition = NUTRITION_LOOKUP.get(
        food_label,
        {"protein": 1.0, "fat": 0.2, "carbs": 10.0, "fiber": 2.0}
    )
    category_code = 0 if food_label in FRUIT_LABELS else 1
    calories = predict_calories(
        nutrition["protein"], nutrition["fat"],
        nutrition["carbs"],   nutrition["fiber"],
        category_code
    )

    ml_output  = f"📊 **Predicted Calories:** {calories} kcal / 100g\n\n"
    ml_output += "**Macronutrients per 100g:**\n"
    ml_output += f"- 🥩 Protein: {nutrition['protein']}g\n"
    ml_output += f"- 🫒 Fat: {nutrition['fat']}g\n"
    ml_output += f"- 🍞 Carbohydrates: {nutrition['carbs']}g\n"
    ml_output += f"- 🌾 Fiber: {nutrition['fiber']}g\n"

    # ── Block 3: NLP ──────────────────────────────────────────────────────────
    nlp_output = generate_advice(
        food_label, calories,
        nutrition["protein"], nutrition["fat"],
        nutrition["carbs"],   nutrition["fiber"]
    )

    summary = (
        f"✅ **{food_label.capitalize()}** detected with "
        f"{confidence*100:.1f}% confidence → {calories} kcal/100g"
    )

    return cv_output, ml_output, nlp_output, summary


# ── Gradio Interface ──────────────────────────────────────────────────────────
with gr.Blocks(title="🍎 Smart Nutrition Advisor") as demo:
    gr.Markdown("""
    # 🍎🥦 Smart Nutrition Advisor
    Upload a photo of a fruit or vegetable and get instant AI-powered nutrition advice!
    - 🔍 **CV (CLIP):** Identifies the food type using Zero-Shot classification
    - 📊 **ML (Ridge Regression):** Predicts calories from macronutrients
    - 💬 **NLP (GPT-4o-mini):** Generates a personalized recipe & health advice
    """)

    with gr.Row():
        image_input = gr.Image(type="pil", label="📸 Upload Fruit or Vegetable Image")

    analyze_btn = gr.Button("🔍 Analyze", variant="primary", size="lg")

    summary_output = gr.Markdown(label="Summary")

    with gr.Row():
        cv_output  = gr.Markdown(label="🔍 Block 1: CV – Food Detection")
        ml_output  = gr.Markdown(label="📊 Block 2: ML – Nutrition Prediction")

    nlp_output = gr.Markdown(label="💬 Block 3: NLP – Recipe & Health Advice")

    analyze_btn.click(
        fn=analyze_food,
        inputs=image_input,
        outputs=[cv_output, ml_output, nlp_output, summary_output]
    )

    gr.Examples(
        examples=[
            [os.path.join(IMAGE_DIR, "apple.jpg")],
            [os.path.join(IMAGE_DIR, "banana.jpg")],
            [os.path.join(IMAGE_DIR, "broccoli.jpg")],
            [os.path.join(IMAGE_DIR, "carrot.jpg")],
        ],
        inputs=image_input,
        label="🖼️ Example Images"
    )

demo.launch()
