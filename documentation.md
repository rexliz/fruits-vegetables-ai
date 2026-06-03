# AI Applications Project Documentation Template

Use this template to document your project concisely and completely.
Fill in all required fields. Keep answers short and precise.

## Documentation Hint

Important:
When possible, reference the corresponding code location directly in your description.

### Example: Reference to a notebook section
Reference to the header `## Data Preprocessing` in the notebook `analysis.ipynb`:

> See *Data Preprocessing* in
> [`analysis.ipynb`](analysis.ipynb#data-preprocessing)

### Example: Reference to Python code

Reference to a single line in `model.py`, line 42:
> [`model.py`, line 42](model.py#L42)

Reference to multiple lines in `train.py`, lines 15-38:
> [`train.py`, lines 15-38](train.py#L15-L38)

## Project Metadata

- Project title: Smart Nutrition Advisor – Fruits & Vegetables AI
- Student: Liza Rexhepi
- GitHub repository URL: https://github.com/rexliz/fruits-vegetables-ai
- Deployment URL: https://huggingface.co/spaces/rexheliz/fruits-vegetables-nutrition-advisor
- Submission date: 03 June 2026

### Mandatory Setup Checks

- [x] At least 2 blocks selected
- [x] Multiple and different data sources used
- [x] Deployment URL provided
- [x] Required GitHub users added to repository (`jasminh`, `bkuehnis`)

## Selected AI Blocks

- [x] ML Numeric Data
- [x] NLP
- [x] Computer Vision

Primary blocks used for core solution (choose 2):
- Primary block 1: ML Numeric Data
- Primary block 2: NLP

If a third block is selected, it is documented and graded separately as extra work.
Computer Vision is used as the third block (bonus).

---

## 1. Project Foundation (Short)

### 1.1 Problem Definition
- Problem statement: Users cannot easily identify the nutritional value of fresh produce from a photo, nor receive personalised cooking advice based on that information.
- Goal: Build an end-to-end AI application that takes a fruit or vegetable photo as input and returns (1) the food type, (2) predicted calorie content, and (3) a personalised recipe and health advice.
- Success criteria: The system correctly identifies common fruits and vegetables, predicts calories within a reasonable margin, and generates coherent, grounded advice. A live deployment is accessible via URL.

### 1.2 Integration Logic
- How the selected blocks interact: The three blocks form a sequential pipeline. CV identifies the food from the image. The food label is used to look up macronutrients, which serve as input features to the ML model. The ML-predicted calories and macronutrients are then passed as context to the NLP block, which generates advice grounded in those values.
- Data and output flow between blocks:

```
Image
  → [CV Block: CLIP Zero-Shot]
  → food_label + confidence
  → nutrition_lookup.json (bridge)
  → protein, fat, carbs, fiber
  → [ML Block: Ridge Regression]
  → calories_per_100g (predicted)
  → [NLP Block: GPT-4o-mini Prompt V3]
  → health benefits + recipe + serving tip
```

> See *Integration with other blocks* in [`1_eda/eda_nutrition.ipynb`](1_eda/eda_nutrition.ipynb), [`2_ml_model/ml_calorie_prediction.ipynb`](2_ml_model/ml_calorie_prediction.ipynb), [`3_cv/cv_classification.ipynb`](3_cv/cv_classification.ipynb), [`4_nlp/nlp_recipe_advisor.ipynb`](4_nlp/nlp_recipe_advisor.ipynb)

---

## 2. Block Documentation

### 2A. ML Numeric Data

#### 2A.1 Data Source(s)

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | [USDA Nutritional Data](https://www.kaggle.com/datasets/niharika41298/nutrition-details-for-most-common-foods) | CSV | 335 rows, 10 columns | Training data for calorie prediction model |
| 2 | `data/nutrition_clean.csv` | CSV (derived) | 329 rows, 11 columns | Cleaned, normalised version used for model training |
| 3 | `data/nutrition_lookup.json` | JSON (derived) | 24 entries | Bridge from CV label to macronutrient features at inference time |

#### 2A.2 Preprocessing and Features
- Cleaning steps: Replaced trace values ('t') with 0; converted string columns to numeric using `pd.to_numeric`; removed rows with missing critical values (`dropna()`); removed rows where Grams = 0 to avoid division-by-zero errors. 3 rows removed from 332 (< 1% data loss).
- Preprocessing steps: Normalised all nutritional values to per-100g basis by dividing by serving weight. Applied `StandardScaler` to all features before model training.
- Feature engineering and selection: Added `fat_x_carbs` interaction feature in Iteration 2 to capture the non-linear calorie contribution of fat (9 kcal/g vs 4 kcal/g). Encoded `Category` as integer (`category_encoded`). Final features: `protein_per_100g`, `fat_per_100g`, `carbs_per_100g`, `fiber_per_100g`, `category_encoded`, `fat_x_carbs`.

> See *Feature Engineering* in [`2_ml_model/ml_calorie_prediction.ipynb`](2_ml_model/ml_calorie_prediction.ipynb#feature-engineering)

#### 2A.3 Model Selection
- Models tested: Linear Regression, Random Forest (Iteration 1); Ridge Regression, Tuned Random Forest (Iteration 2).
- Why these models were chosen: Linear/Ridge regression serves as an interpretable baseline with low variance. Random Forest and Gradient Boosting were tested as non-linear alternatives that can capture interaction effects. Ridge was preferred over plain Linear Regression to reduce overfitting on the small dataset (329 rows).

#### 2A.4 Model Comparison and Iterations

| Iteration | Objective | Key changes | Models used | Main metric | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Baseline | Original 5 features, 5-fold CV | Linear Regression, Random Forest (100 trees) | CV R²: 0.95 (LR), 0.94 (RF) | Baseline |
| 2 | Improve with feature engineering | Added `fat_x_carbs` interaction term; tuned RF (300 trees, max_depth=15) | Ridge (α=1.0), Tuned Random Forest | CV R²: 0.97 (Ridge), 0.96 (RF) | +0.02 R² for Ridge |

> See *Model Comparison* in [`2_ml_model/ml_calorie_prediction.ipynb`](2_ml_model/ml_calorie_prediction.ipynb#model-comparison)

#### 2A.5 Evaluation and Error Analysis
- Metrics used: R² (primary), RMSE, MAE on held-out test set (20%, 66 samples), 5-fold cross-validation.
- Final results: Ridge Regression selected. Test R² = 0.97, MAE = 24.32 kcal/100g, Median Absolute Error = 10.25 kcal/100g, Max Error = 246.60 kcal/100g (Gin).
- Error patterns and likely causes: (1) Worst prediction: Gin (actual 250 kcal, predicted 3.4 kcal, residual +246.6). Gin calories come from alcohol (7 kcal/g), which is not a feature in the model — a fundamental out-of-distribution case. (2) Median error of 10.25 kcal vs mean of 24.32 kcal reveals a skewed distribution: most predictions are accurate, but a small number of outlier foods (Drinks, Meat, Dairy) pull the average up. (3) The model is correctly scoped: CLIP only classifies fruits and vegetables, so Gin/Bacon/Lard-type errors never occur in the deployed app.

> See *Error Analysis* in [`2_ml_model/ml_calorie_prediction.ipynb`](2_ml_model/ml_calorie_prediction.ipynb#error-analysis)

#### 2A.6 Integration with Other Block(s)
- Inputs received from other block(s): Food label from CV block → macronutrient lookup in `nutrition_lookup.json` → `protein`, `fat`, `carbs`, `fiber`, `category_code` as model input features.
- Outputs provided to other block(s): `calories_per_100g` (predicted float) passed to NLP block as part of the prompt context. Also passed: macronutrient values for grounded advice generation.

---

### 2B. NLP

#### 2B.1 Data Source(s)

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | GPT-4o-mini (OpenAI API) | Pretrained LLM | – | Generates health advice and recipe from structured prompt |
| 2 | ML block output | Structured data (float + dict) | 1 prediction per inference | Provides calories + macronutrients as grounding context in prompt |
| 3 | CV block output | Structured data (string + float) | 1 label + confidence per inference | Provides food label as topic anchor in prompt |

#### 2B.2 Preprocessing and Prompt Design
- Text preprocessing: No classical NLP preprocessing (tokenisation, stemming) required. Input is structured numeric data from ML and CV blocks, formatted into a prompt template at runtime.
- Prompt design or retrieval setup: Prompt engineering approach (zero-shot). Three prompt iterations designed and compared. Final prompt (Iteration 3) uses a system message for role assignment, a grounding instruction ("do not invent or estimate nutritional data"), a strict output template (HEALTH BENEFITS / RECIPE / SERVING TIP), and temperature 0.3 for reproducible output.

> See *Prompt Design* in [`4_nlp/nlp_recipe_advisor.ipynb`](4_nlp/nlp_recipe_advisor.ipynb#prompt-design)

#### 2B.3 Approach Selection
- Approach used: Prompt engineering with GPT-4o-mini (zero-shot, no fine-tuning, no RAG).
- Alternatives considered: RAG over nutrition knowledge base (PDFs) was considered but not implemented in the primary scope, as the ML block already provides structured, grounded nutritional values — making RAG redundant for the core use case. Prompt engineering was sufficient and faster to iterate.

#### 2B.4 Comparison and Iterations

| Iteration | Objective | Key changes | Model or prompt setup | Main metric or qualitative check | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Minimal baseline | One-sentence prompt, no context | GPT-4o-mini, user-only, temp 0.7 | Score 4/12 (Completeness 1, Structure 1, Actionability 1, Hallucination Risk 1) | Baseline |
| 2 | Structured output | Role assignment, macronutrient context, numbered sections | GPT-4o-mini, user-only, temp 0.7 | Score 9/12 (all criteria 2-3) | +5/12 vs baseline |
| 3 | Grounded, reproducible | System message, grounding instruction ("do not invent data"), strict template, temp 0.3 | GPT-4o-mini, system+user, temp 0.3 | Score 12/12 (all criteria 3) | +3/12 vs Iteration 2 |

> See *Comparison and Iterations* in [`4_nlp/nlp_recipe_advisor.ipynb`](4_nlp/nlp_recipe_advisor.ipynb#comparison-and-iterations)

#### 2B.5 Evaluation and Error Analysis
- Evaluation strategy: Qualitative evaluation across 4 criteria (Completeness, Structure, Actionability, Hallucination Risk) on a 1–3 scale, tested on 2 food items (apple, broccoli) per iteration. Quantitative comparison of word count and section coverage across iterations.
- Results: Iteration 3 achieves maximum score (12/12). Word count consistent (130–170 words). All required sections present in every output. Avg. confidence in output: high (strict format followed consistently).
- Error patterns and likely causes: (1) Residual hallucination risk: even Iteration 3 may cite micronutrients (e.g. vitamin C) not present in the feature set — an inherent LLM limitation. (2) API dependency: if the OpenAI API is unavailable, the NLP block fails entirely — no fallback implemented. (3) CV misclassification propagates: if pepper is detected as tomato, advice is generated for tomato — still useful, but not optimal.

> See *Evaluation* in [`4_nlp/nlp_recipe_advisor.ipynb`](4_nlp/nlp_recipe_advisor.ipynb#evaluation)

#### 2B.6 Integration with Other Block(s)
- Inputs received from other block(s): `food_label` (string) from CV block; `calories` (float), `protein`, `fat`, `carbs`, `fiber` (floats) from ML block. All values injected into the Iteration 3 prompt template at runtime.
- Outputs provided to other block(s): Final text output (health benefits + recipe + serving tip) displayed in the Gradio app. Not passed to any further block.

---

### 2C. Computer Vision (Bonus Block)

#### 2C.1 Data Source(s)

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | CLIP pretrained model (`openai/clip-vit-base-patch32`, HuggingFace) | Pretrained vision-language model | 605 MB | Zero-Shot classifier for food type detection |
| 2 | [Fruits and Vegetables Image Recognition (kritikseth, Kaggle)](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition) | Image dataset | ~2 GB, 36 classes, train/test/val split | Evaluation dataset for quantitative accuracy measurement |
| 3 | Wikimedia Commons (4 public domain images) | Images | 4 images | Qualitative demo in `cv_classification.ipynb` |

#### 2C.2 Preprocessing and Augmentation
- Image preprocessing: Images loaded as PIL RGB via `Image.open().convert('RGB')`. No resizing or normalisation applied manually — the CLIP pipeline handles all internal preprocessing (resize to 224×224, normalise with ImageNet stats). No augmentation applied (Zero-Shot: no training).
- Augmentation strategy: Not applicable — CLIP is used in Zero-Shot mode with no training on our data.

#### 2C.3 Model Selection
- Vision model(s) used: `openai/clip-vit-base-patch32` via HuggingFace `transformers` pipeline, Zero-Shot image classification.
- Why these model(s) were chosen: Zero-Shot classification eliminates the need for a labelled training dataset and GPU training time. CLIP's pretrained visual-language knowledge generalises well to food classification. New classes can be added by simply adding a label string — no retraining required.

#### 2C.4 Model Comparison and Iterations

| Iteration | Objective | Key changes | Model(s) used | Main metric | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Qualitative test (4 images) | Defined 24 class labels (12 fruits + 12 vegetables) | CLIP ViT-B/32, Zero-Shot | Visual inspection: 4/4 correct | Baseline (qualitative only) |
| 2 | Quantitative evaluation | Added 209-image test set (kritikseth Kaggle dataset), computed accuracy + confusion matrix | CLIP ViT-B/32, Zero-Shot | Top-1 Accuracy: 81.8% | Added systematic metrics |

> See *Quantitative Evaluation* in [`3_cv/cv_evaluation.ipynb`](3_cv/cv_evaluation.ipynb)

#### 2C.5 Evaluation and Error Analysis
- Metrics and/or visual checks: Top-1 Accuracy, per-class Precision/Recall/F1 (sklearn `classification_report`), Confusion Matrix (heatmap), visual inspection of misclassified examples.
- Final results: Overall Top-1 Accuracy = **81.8%** on 209 test images (18 classes). Avg. confidence correct predictions: 94.0%. Avg. confidence incorrect: 65.2%. 6 classes achieve perfect F1=1.00 (banana, carrot, corn, lettuce, pineapple, watermelon, spinach).
- Error patterns and limitations: (1) Pepper completely fails (F1=0.00): 20/30 images misclassified as tomato, 10/30 as cucumber. Both red bell peppers and tomatoes are round and red — CLIP has no fine-grained shape/texture knowledge for these overlapping classes. (2) Confidence gap (94% vs 65%) is a useful reliability signal: a confidence threshold of 70% is implemented in the app to warn users of uncertain predictions. (3) Zero-Shot limitation: no fine-tuning means accuracy is capped by pretrained knowledge. Fine-tuning on this dataset would likely exceed 95%.

> See *Evaluation and Error Analysis* in [`3_cv/cv_evaluation.ipynb`](3_cv/cv_evaluation.ipynb#error-analysis)

#### 2C.6 Integration with Other Block(s)
- Inputs received from other block(s): User-uploaded PIL image (from Gradio interface).
- Outputs provided to other block(s): `food_label` (string, e.g. "apple") and `confidence` (float, e.g. 0.94) passed to ML block (via `nutrition_lookup.json`) and to NLP block (as topic anchor in prompt). Confidence below 0.70 triggers a user-visible warning in the app.

---

## 3. Deployment

- Deployment URL: https://huggingface.co/spaces/rexheliz/fruits-vegetables-nutrition-advisor
- Main user flow: (1) User uploads a fruit or vegetable photo (or selects an example image). (2) Clicks "Analyze". (3) CV block detects the food type and displays top-3 predictions with confidence scores. (4) ML block displays predicted calories and macronutrients per 100g. (5) NLP block displays personalised health benefits, a recipe, and a serving tip.
- Screenshot or short demo:

> Screenshot 1 (`Apple.png`): Correct classification — Apple detected (82.5% confidence), 59.2 kcal/100g predicted, full recipe and health advice generated in Iteration 3 format.
> Screenshot 2 (`Strawberry.png`): Correct classification — Strawberry detected (96.6% confidence), 41.1 kcal/100g predicted, strawberry smoothie recipe generated.
> Screenshot 3 (`Spinach.png`): High-confidence vegetable classification — Spinach detected (99.6% confidence), 29.9 kcal/100g predicted. Demonstrates the strong performance on clearly distinguishable classes.
> Screenshot 4 (`Bell_Pepper_-_Tomato.png`): Known failure case — Red bell pepper uploaded, detected as Tomato (86.2% confidence). Consistent with `cv_evaluation.ipynb` findings (pepper→tomato: 20/30 errors). The downstream calorie prediction (30.9 kcal) and recipe remain reasonable since pepper and tomato have similar nutritional profiles, demonstrating the system's graceful degradation.

> See `app/app.py` for full deployment code. Training is separated from inference: all models trained in notebooks, saved as `.pkl`, loaded at runtime by the app.

---

## 4. Execution Instructions

- Environment setup:
```bash
git clone https://github.com/rexliz/fruits-vegetables-ai
cd fruits-vegetables-ai
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

- Data setup: Place `nutrition.csv` in `data/`. Run `1_eda/eda_nutrition.ipynb` to generate `nutrition_clean.csv`.

- Training command(s): Run notebooks in order:
  1. `1_eda/eda_nutrition.ipynb` — generates `data/nutrition_clean.csv`
  2. `2_ml_model/ml_calorie_prediction.ipynb` — generates `data/best_model.pkl`, `data/scaler.pkl`, `data/feature_names.pkl`
  3. `3_cv/cv_classification.ipynb` — generates `data/nutrition_lookup.json`
  4. `3_cv/cv_evaluation.ipynb` — runs on Google Colab with GPU (requires Kaggle dataset)
  5. `4_nlp/nlp_recipe_advisor.ipynb` — requires `OPENAI_API_KEY` in `.env`

- Inference/run command(s):
```bash
# Set OPENAI_API_KEY in .env file, then:
python app/app.py
# App available at http://localhost:7860
```

- Reproducibility notes: All random seeds set to 42. Python 3.11, sklearn 1.7.2. Full dependency list in `requirements.txt`. CLIP model downloaded automatically from HuggingFace on first run (~605 MB). OpenAI API key required for NLP block.

---

## 5. Optional Bonus Evidence

- [x] Third selected block implemented with strong quality — Computer Vision block with full quantitative evaluation (81.8% accuracy on 209-image test set, confusion matrix, per-class F1, error analysis)
- [x] More than two data sources used with clear added value — USDA Nutritional CSV (ML training), Kaggle Fruits & Vegetables image dataset (CV evaluation), CLIP pretrained model (CV inference), OpenAI GPT-4o-mini (NLP generation): 4 distinct external sources
- [x] Extended evaluation — CV block: accuracy + confusion matrix + per-class F1 + confidence analysis + misclassified image visualisation. ML block: R² + MAE + Median AE + residual distribution + worst-10 predictions + error-by-category chart.
- [x] Ethics, bias, or fairness analysis — See below.

Evidence for selected bonus items:

**Third block (CV):** Full evaluation notebook at `3_cv/cv_evaluation.ipynb`. 81.8% Top-1 Accuracy without any training or fine-tuning (random baseline = 5.6%). Confusion matrix and misclassified image gallery included.

**Multiple data sources:** (1) USDA Nutritional Data CSV — ML training; (2) Kaggle kritikseth image dataset — CV evaluation; (3) CLIP ViT-B/32 pretrained weights — CV inference; (4) GPT-4o-mini API — NLP generation. Each source has a distinct role and is documented in the respective block's data source table.

**Extended evaluation:** ML error analysis produces 3 plots (Predicted vs Actual, Residuals Distribution, Error by Category) with interpretation of outlier foods. CV evaluation produces confusion matrix + misclassified image gallery + confidence gap analysis (94% vs 65%).

**Ethics and bias analysis:**
The system has two notable bias risks. First, the CV block shows class imbalance in error rates: pepper completely fails (F1=0.00) while most other classes perform well. This means users photographing peppers systematically receive incorrect advice — a fairness concern if the app were used in a health context. Second, the ML training data (USDA) is dominated by basic fruits and vegetables; high-fat foods (Lard, Bacon) and alcoholic beverages (Gin) are severely underrepresented, leading to extreme prediction errors for these categories. Mitigation implemented: a confidence threshold (< 70%) triggers a visible warning in the app, informing users when the CV prediction is uncertain. Future work: fine-tune CLIP on a balanced dataset; add alcohol content as a feature to the ML model.
