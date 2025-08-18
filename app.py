import os
import json
import re
from io import BytesIO
from typing import List, Dict, Tuple

import streamlit as st
from PIL import Image
import pytesseract

# Optional: point pytesseract to your tesseract binary on Windows
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---- LLM client (OpenAI) ----
# pip install openai>=1.0  (using the new SDK)
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# ------------- UI HEADER ----------------
st.set_page_config(page_title="Ingredient Reader (OCR + LLM)", page_icon="🧪", layout="centered")
st.title("🧪 Ingredient Reader — OCR + LLM")
st.caption("Scan a label → Extract ingredients → Flag harmful ones → Explain risks + alternatives")

with st.expander("🔧 Setup Notes", expanded=False):
    st.markdown(
        """
- Install **Tesseract OCR** on your system and ensure it’s on PATH.
- Set `OPENAI_API_KEY` in your environment (for LLM explanations).  
  On macOS/Linux: `export OPENAI_API_KEY="sk-..."`  
  On Windows (PowerShell): `$env:OPENAI_API_KEY="sk-..."`
- Extend **harmful_ingredients.json** anytime (no code changes needed).
        """
    )

# ------------- LOAD DATASET -------------
@st.cache_data
def load_dataset() -> Dict:
    with open("harmful_ingredients.json", "r", encoding="utf-8") as f:
        return json.load(f)

data = load_dataset()
HARMFUL = [s.lower().strip() for s in data.get("harmful_ingredients", [])]
SYNONYMS = {k.lower(): [x.lower() for x in v] for k, v in data.get("synonyms", {}).items()}

# Build a quick lookup that includes synonyms
EXPANDED_SET = set(HARMFUL)
for base, syns in SYNONYMS.items():
    EXPANDED_SET.add(base)
    for s in syns:
        EXPANDED_SET.add(s)

# ------------- HELPERS ------------------
def extract_text_from_image(img: Image.Image) -> str:
    """Run OCR and return raw text."""
    # Optional pre-processing: convert to grayscale & increase contrast if needed
    gray = img.convert("L")
    text = pytesseract.image_to_string(gray)
    return text

def normalize(text: str) -> str:
    """normalize punctuation/spaces"""
    text = text.lower()
    # unify separators: commas/semicolons/periods
    text = re.sub(r"[\n\r]+", " ", text)
    text = text.replace("•", " ").replace("|", " ")
    text = re.sub(r"[;:/\-–—]+", ",", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def split_ingredients(text: str) -> List[str]:
    """
    Attempt to split an 'ingredients:' paragraph into list items.
    Uses commas as primary separators; strips parens.
    """
    # try to find the section after 'ingredient' token
    m = re.search(r"(ingredient[s]?:)(.*)", text, flags=re.IGNORECASE)
    segment = m.group(2) if m else text

    # remove parenthetical content for matching phase (but keep original later if needed)
    cleaned = re.sub(r"\(.*?\)", "", segment)
    # split by commas
    items = [x.strip() for x in cleaned.split(",") if x.strip()]
    # shorten long tokens and keep alphanumeric + spaces
    items = [re.sub(r"[^a-z0-9\s\-\&\.]", "", it) for it in items]
    # remove duplicates while preserving order
    seen = set()
    uniq = []
    for it in items:
        if it not in seen:
            uniq.append(it)
            seen.add(it)
    return uniq

def match_harmful(ingredients: List[str]) -> Tuple[List[str], List[str]]:
    """Return (found_harmful, unknown_or_uncertain)"""
    found = []
    unknown = []
    for it in ingredients:
        token = it.lower()
        # direct or synonym hit by contains OR exact token
        # Try exact first:
        if token in EXPANDED_SET:
            found.append(it)
            continue
        # Try substring contains any harmful key:
        hit = False
        for h in EXPANDED_SET:
            if h in token:
                found.append(it)
                hit = True
                break
        if not hit:
            unknown.append(it)
    return (sorted(set(found)), unknown)

def call_llm_explain(unknown_list: List[str], user_allergies: List[str]) -> str:
    """Ask LLM to classify unknowns + explain risks and suggest safer alternatives.
       Returns formatted markdown. If no API key or lib, skip gracefully."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or OpenAI is None or len(unknown_list) == 0:
        return ""

    client = OpenAI(api_key=api_key)
    allergies = ", ".join(user_allergies) if user_allergies else "None"
    ing_text = "; ".join(unknown_list)

    prompt = f"""
You are a food/cosmetic ingredient safety assistant.
Given these ingredients: {ing_text}
1) Flag any potentially harmful/allergenic items (even if not in a fixed list), identify their type (e.g., preservative, color, fragrance, sweetener).
2) Explain risks in one simple sentence each (avoid fear-mongering).
3) Consider user allergies: {allergies}; add a note if relevant.
4) Suggest one safer alternative per flagged item, if possible.
Return concise, structured markdown with bullets.
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"*LLM explanation unavailable:* `{e}`"

# ------------- UI -----------------------
st.subheader("1) Upload a label photo")
img_file = st.file_uploader("Upload a product label image (JPG/PNG).", type=["jpg", "jpeg", "png"])

st.subheader("2) (Optional) Personalize")
col1, col2 = st.columns(2)
with col1:
    user_allergies = st.tags(label="Allergies (type and press Enter)", text="e.g., peanuts, gluten, dairy", suggestions=["peanuts","gluten","dairy","soy","eggs","shellfish"]) if hasattr(st, "tags") else []
with col2:
    run_llm = st.checkbox("Use LLM for unknowns (explanations)", value=True)

if img_file is not None:
    image = Image.open(BytesIO(img_file.read()))
    st.image(image, caption="Uploaded Label", use_column_width=True)

    if st.button("🔍 Scan Label (OCR)"):
        with st.spinner("Running OCR..."):
            raw_text = extract_text_from_image(image)
            norm_text = normalize(raw_text)
            items = split_ingredients(norm_text)

        st.success("OCR complete.")
        st.markdown("**Extracted Text (first 600 chars)**")
        st.code(raw_text[:600] + ("..." if len(raw_text) > 600 else ""), language="markdown")

        st.markdown("**Parsed Ingredients**")
        if items:
            st.write(items)
        else:
            st.info("Could not parse ingredient list. You can still review OCR text above.")

        found, unknown = match_harmful(items)

        # Results
        if found:
            st.error(f"⚠️ Harmful/flagged detected ({len(found)}): " + ", ".join(found))
        else:
            st.success("✅ No known harmful items from local dataset.")

        # LLM explanations for unknowns
        llm_md = ""
        if run_llm and unknown:
            with st.spinner("Asking LLM to analyze unknowns & explain…"):
                llm_md = call_llm_explain(unknown, user_allergies)

        if unknown:
            st.markdown("**Unknown / not in local dataset**")
            st.write(unknown)

        if llm_md:
            st.markdown("**LLM Risk & Alternatives (for unknowns)**")
            st.markdown(llm_md)

        # Final badge
        if found:
            st.markdown("### 🟥 Verdict: **WARNING**")
        else:
            st.markdown("### 🟩 Verdict: **SAFE (based on local dataset)**")
            if llm_md and ("potentially" in llm_md.lower() or "avoid" in llm_md.lower()):
                st.caption("Note: LLM flagged context-specific cautions; review above.")

else:
    st.info("Upload an image to begin.")

st.divider()
st.caption("Tip: Expand your dataset in harmful_ingredients.json. LLM adds reasoning beyond the list (synonyms, classes, context).")
