from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch

app = Flask(__name__)

# ============================================================
# 1. ENGINE CONFIGURATION & MODEL INITIALIZATION
# ============================================================

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

print("⏳ Loading Model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

tone_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=0 if torch.cuda.is_available() else -1
)

print("✅ Model Loaded Successfully!")


# ============================================================
# 2. PROMPT ENGINEERING LAYER
# ============================================================

def build_pro_marketing_prompt(data):

    savings = round(data['retail'] - data['discount'], 2)
    final_price = round(data['discount'] if data['discount'] > 0 else data['retail'], 2)
    is_discounted = savings > 0 and data['discount'] != 0

    if is_discounted:
        pricing_instruction = (
            f"Show ONLY the Sale Price ₹{final_price} and the exact amount saved (₹{savings}). "
            f"ABSOLUTELY DO NOT mention the original price ₹{data['retail']}."
        )
    else:
        pricing_instruction = f"Show the price clearly as ₹{final_price}. No discount mentions."

    prompt = f"""<s>[INST] <<SYS>>
You are a world-class luxury ad copywriter.
Never use casual language.
<</SYS>>

Create a {data['platform']} post.

Product Name: {data['name']}
Features: {data['desc']}
Final Price: ₹{final_price}
{f"Amount Saved: ₹{savings}" if is_discounted else ""}
Shipping: {"FREE SHIPPING (Mention it as 'complimentary delivery' or 'on the house')" if data['free_shipping'] else "Standard rates"}
Tone: {data['tone']}

GUIDELINES:
1. Start with a professional hook.
2. {pricing_instruction}
3. Use 📦 ✨ 💎 emojis only.
4. Add 2-3 hashtags.
5. Output ONLY post text.
[/INST]</s>"""

    return prompt


# ============================================================
# 3. VALIDATION LAYER
# ============================================================

def validate_output(text, data):

    savings = round(data['retail'] - data['discount'], 2)
    final_price = round(data['discount'] if data['discount'] > 0 else data['retail'], 2)
    is_discounted = savings > 0 and data['discount'] != 0

    has_price = str(final_price) in text
    mentions_retail = str(int(data['retail'])) in text if is_discounted else False

    if has_price and not mentions_retail:
        return "Valid"
    elif mentions_retail:
        return "Needs Review (Original price leaked)"
    return "Needs Review"


# ============================================================
# 4. GENERATION FUNCTION
# ============================================================

def generate_post(prompt):

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=350,
            temperature=0.1,
            do_sample=True,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )

    raw_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    clean_text = raw_text.split("[/INST]")[-1].strip()

    return clean_text


# ============================================================
# 5. FLASK ROUTE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    result = None

    if request.method == "POST":

        product_info = {
            "name": request.form["name"],
            "desc": request.form["desc"],
            "retail": float(request.form["retail"]),
            "discount": float(request.form["discount"]),
            "free_shipping": True if request.form.get("free_shipping") == "y" else False,
            "platform": request.form["platform"],
            "tone": request.form["tone"]
        }

        # Build Prompt
        prompt = build_pro_marketing_prompt(product_info)

        # Generate
        post = generate_post(prompt)

        # Validate
        fact_check = validate_output(post, product_info)

        # Tone Analysis
        sentiment = tone_analyzer(post[:512])[0]['label']

        result = {
            "post": post,
            "validation": fact_check,
            "tone": sentiment,
            "platform": product_info["platform"]
        }

    return render_template("index.html", result=result)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)