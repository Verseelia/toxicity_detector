import streamlit as st
import numpy as np
import torch

from transformers import BertTokenizerFast, BertForSequenceClassification


# =========================================================
# CONFIG
# =========================================================

MODEL_PATH = "./toxicbert_model"

LABELS = [
    ("toxic", "Toxic", "Rude, disrespectful, or harmful tone."),
    ("severe_toxic", "Severe toxic", "Extremely hateful or aggressive language."),
    ("obscene", "Obscene", "Profane, vulgar, or sexually explicit language."),
    ("threat", "Threat", "Expresses intent to cause harm."),
    ("insult", "Insult", "Intended to demean or offend someone."),
    ("identity_hate", "Identity hate", "Targets a person's identity or group."),
]

CLEAR = "#0f766e"      # below threshold
FLAG = "#b42318"       # above threshold
INK = "#1c1f26"
MUTED = "#6b7280"
LINE = "#e4e7ec"
PAPER = "#f6f7f9"


st.set_page_config(page_title="Toxicity Detector", page_icon="◆", layout="centered")


# =========================================================
# STYLE
# =========================================================

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {PAPER}; }}
    .block-container {{ max-width: 760px; padding-top: 3rem; padding-bottom: 3rem; }}

    * {{ font-variant-numeric: tabular-nums; }}

    .eyebrow {{
        color: {MUTED}; font-size: 13px; letter-spacing: 0.02em;
        margin-bottom: 4px;
    }}
    h1.title {{
        font-size: 32px; font-weight: 650; color: {INK};
        margin: 0 0 6px 0; letter-spacing: -0.01em;
    }}
    p.subtitle {{
        color: {MUTED}; font-size: 15.5px; margin: 0 0 28px 0; max-width: 52ch;
    }}

    div[data-testid="stTextArea"] textarea {{
        border-radius: 10px; border: 1px solid {LINE}; font-size: 15px;
    }}

    div[data-testid="stButton"] button {{
        background: {INK}; color: white; border: none; border-radius: 8px;
        font-weight: 600; height: 44px; padding: 0 22px;
    }}
    div[data-testid="stButton"] button:hover {{ background: #33384a; }}

    .verdict {{
        border-radius: 10px; padding: 16px 18px; margin: 26px 0 22px 0;
        display: flex; align-items: baseline; justify-content: space-between;
        border: 1px solid;
    }}
    .verdict.flag {{ background: #fef2f1; border-color: #f6c2bd; color: {FLAG}; }}
    .verdict.clear {{ background: #f0faf8; border-color: #b9e4dc; color: {CLEAR}; }}
    .verdict .headline {{ font-size: 17px; font-weight: 650; }}
    .verdict .detail {{ font-size: 13.5px; color: {MUTED}; }}

    .meter-row {{ padding: 13px 0; border-bottom: 1px solid {LINE}; }}
    .meter-row:last-child {{ border-bottom: none; }}
    .meter-top {{
        display: flex; justify-content: space-between; align-items: baseline;
        margin-bottom: 8px;
    }}
    .meter-label {{ font-size: 15px; font-weight: 600; color: {INK}; }}
    .meter-desc {{ font-size: 12.5px; color: {MUTED}; margin-top: 1px; }}
    .meter-value {{ font-size: 14px; font-weight: 650; }}
    .meter-track {{
        position: relative; height: 7px; background: #eceef2; border-radius: 4px;
        overflow: visible;
    }}
    .meter-fill {{ position: absolute; top: 0; left: 0; height: 100%; border-radius: 4px; }}
    .meter-threshold {{
        position: absolute; top: -3px; width: 2px; height: 13px; background: {INK};
        opacity: 0.35;
    }}

    .section-label {{
        font-size: 13px; font-weight: 650; color: {MUTED}; letter-spacing: 0.02em;
        margin: 30px 0 4px 0; text-transform: uppercase;
    }}

    footer {{ visibility: hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# MODEL
# =========================================================

@st.cache_resource
def load_model():
    tok = BertTokenizerFast.from_pretrained(MODEL_PATH)
    mdl = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    mdl.eval()
    return tok, mdl


@st.cache_resource
def load_thresholds():
    return np.load(f"{MODEL_PATH}/thresholds.npy")


def predict(text: str):
    tokenizer, model = load_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    return torch.sigmoid(logits).numpy()[0]


# =========================================================
# HEADER
# =========================================================

st.markdown('<div class="eyebrow">Content moderation</div>', unsafe_allow_html=True)
st.markdown('<h1 class="title">Toxicity Detector</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Paste a comment and Toxic-BERT will score it across six '
    'categories, each checked against its own tuned threshold.</p>',
    unsafe_allow_html=True,
)

comment = st.text_area(
    "Comment",
    height=150,
    placeholder="Type or paste a comment here…",
    label_visibility="collapsed",
)

analyze = st.button("Analyze comment")


# =========================================================
# RESULTS
# =========================================================

if analyze:
    if not comment.strip():
        st.warning("Enter a comment before analyzing.")
    else:
        thresholds = load_thresholds()
        probabilities = predict(comment)
        flags = probabilities >= thresholds
        detected = int(flags.sum())

        # ---- verdict banner ----
        if detected > 0:
            st.markdown(
                f"""
                <div class="verdict flag">
                    <span class="headline">Potentially toxic content detected</span>
                    <span class="detail">{detected} of {len(LABELS)} categories crossed threshold</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="verdict clear">
                    <span class="headline">No toxicity detected</span>
                    <span class="detail">All categories stayed under threshold</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ---- meters, sorted by probability ----
        st.markdown('<div class="section-label">Category breakdown</div>', unsafe_allow_html=True)

        order = np.argsort(-probabilities)
        rows = []
        for i in order:
            key, name, desc = LABELS[i]
            prob = float(probabilities[i])
            thresh = float(thresholds[i])
            color = FLAG if bool(flags[i]) else CLEAR
            rows.append(
                f"""
                <div class="meter-row">
                    <div class="meter-top">
                        <div>
                            <div class="meter-label">{name}</div>
                            <div class="meter-desc">{desc}</div>
                        </div>
                        <div class="meter-value" style="color:{color};">{prob:.1%}</div>
                    </div>
                    <div class="meter-track">
                        <div class="meter-fill" style="width:{prob*100:.2f}%; background:{color};"></div>
                        <div class="meter-threshold" style="left:{thresh*100:.2f}%;"></div>
                    </div>
                </div>
                """
            )
        st.markdown("".join(rows), unsafe_allow_html=True)
        st.markdown(
            f'<p style="font-size:12px;color:{MUTED};margin-top:10px;">'
            'The faint vertical tick on each bar marks that category\'s classification threshold.</p>',
            unsafe_allow_html=True,
        )

        # ---- analyzed text ----
        st.markdown('<div class="section-label">Analyzed text</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="background:white;border:1px solid {LINE};border-radius:10px;'
            f'padding:14px 16px;font-size:14.5px;color:{INK};">{comment}</div>',
            unsafe_allow_html=True,
        )


# =========================================================
# ABOUT
# =========================================================

with st.expander("About this model"):
    st.write(
        """
        This app uses a fine-tuned Toxic-BERT transformer for multi-label
        toxicity classification. Each comment is scored independently across
        six categories — toxic, severe toxic, obscene, threat, insult, and
        identity hate — and each category has its own optimized decision
        threshold rather than one shared cutoff.
        """
    )

st.markdown(
    f'<p style="text-align:center;color:{MUTED};font-size:12.5px;margin-top:40px;">'
    'Toxicity Detection · Toxic-BERT + Transformers</p>',
    unsafe_allow_html=True,
)