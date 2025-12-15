import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import os
import json
import io

# --- File handling ---
import pypdf
from docx import Document
from pptx import Presentation

# --- Gemini ---
import google.generativeai as genai


# =========================================================
# DB
# =========================================================

DB_FILE = "study_log.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            topic TEXT,
            is_correct INTEGER
        )
    """)
    conn.commit()
    conn.close()

def log_result(topic, is_correct):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        "INSERT INTO logs (timestamp, topic, is_correct) VALUES (?, ?, ?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), topic, int(is_correct))
    )
    conn.commit()
    conn.close()

def get_stats():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT * FROM logs", conn)
    conn.close()
    return df


# =========================================================
# Gemini
# =========================================================

def configure_gemini():
    api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("Gemini APIキーが設定されていません")
        return False
    genai.configure(api_key=api_key)
    return True


# =========================================================
# File extraction
# =========================================================

def extract_from_pdf(file_bytes):
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    texts = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            texts.append(f"【ページ {i+1}】\n{text}")
    return "\n\n".join(texts)

def extract_from_docx(file_bytes):
    doc = Document(io.BytesIO(file_bytes))
    texts = []
    for p in doc.paragraphs:
        if p.style.name.startswith("Heading"):
            texts.append(f"\n## {p.text}\n")
        else:
            texts.append(p.text)
    return "\n".join(texts)

def extract_from_xlsx(file_bytes):
    xl = pd.ExcelFile(io.BytesIO(file_bytes))
    texts = []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        texts.append(f"\n## シート: {sheet}\n")
        texts.append(df.to_csv(index=False))
    return "\n".join(texts)

def extract_from_pptx(file_bytes):
    prs = Presentation(io.BytesIO(file_bytes))
    texts = []
    for i, slide in enumerate(prs.slides):
        texts.append(f"\n## スライド {i+1}\n")
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                texts.append(shape.text)
    return "\n".join(texts)

def extract_text(uploaded_file):
    suffix = uploaded_file.name.split(".")[-1].lower()
    data = uploaded_file.read()

    if suffix == "pdf":
        return extract_from_pdf(data)
    if suffix == "docx":
        return extract_from_docx(data)
    if suffix == "xlsx":
        return extract_from_xlsx(data)
    if suffix == "pptx":
        return extract_from_pptx(data)

    raise ValueError("未対応形式")


# =========================================================
# AI problem generation
# =========================================================

def safe_json_load(text):
    try:
        return json.loads(text)
    except Exception:
        # JSON修復（最低限）
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1:
            return json.loads(text[start:end+1])
        raise

def generate_ai_problems(text, n=5):
    model = genai.GenerativeModel("gemini-1.5-flash")

    system_prompt = """
あなたは大学レベル教材の教育AIです。
与えられた資料内容のみに基づいて問題を作成してください。

- 表（CSV形式）は関係性として理解する
- スライド文章は講義要点として扱う
- 資料外知識は禁止
- JSONのみ出力
"""

    prompt = f"""
以下の資料から {n} 問の一問一答問題を作成してください。

JSON形式:
[
  {{
    "question": "...",
    "answer": "...",
    "explanation": "..."
  }}
]

資料:
{text[:3000]}
"""

    response = model.generate_content(
        [system_prompt, prompt],
        generation_config={"temperature": 0.2}
    )

    return safe_json_load(response.text)


# =========================================================
# UI
# =========================================================

def main():
    st.set_page_config("AIコーチング", layout="centered")
    st.title("📘 AIコーチング学習アプリ")

    init_db()
    if not configure_gemini():
        return

    if "text" not in st.session_state:
        st.session_state.text = None
    if "problems" not in st.session_state:
        st.session_state.problems = []
    if "idx" not in st.session_state:
        st.session_state.idx = 0

    tab1, tab2, tab3 = st.tabs(["資料", "問題", "履歴"])

    # -------------------------
    with tab1:
        file = st.file_uploader(
            "資料アップロード",
            type=["pdf", "docx", "xlsx", "pptx"]
        )
        if file:
            with st.spinner("解析中..."):
                st.session_state.text = extract_text(file)
            st.success("資料を読み込みました")
            if st.button("問題生成"):
                st.session_state.problems = generate_ai_problems(st.session_state.text)
                st.session_state.idx = 0
                st.rerun()

    # -------------------------
    with tab2:
        if not st.session_state.problems:
            st.info("問題がありません")
            return

        p = st.session_state.problems[st.session_state.idx]
        st.subheader(f"問題 {st.session_state.idx + 1}")
        st.markdown(p["question"])

        st.markdown("---")
        st.markdown(f"**正解:** {p['answer']}")
        st.markdown(p["explanation"])

        col1, col2 = st.columns(2)
        if col1.button("⭕ 正解"):
            log_result("AI問題", 1)
            st.session_state.idx += 1
            st.rerun()
        if col2.button("❌ 不正解"):
            log_result("AI問題", 0)
            st.session_state.idx += 1
            st.rerun()

    # -------------------------
    with tab3:
        df = get_stats()
        if df.empty:
            st.info("履歴なし")
        else:
            stats = df.groupby("topic").agg(
                正解数=("is_correct", "sum"),
                回答数=("id", "count")
            )
            stats["正答率"] = stats["正解数"] / stats["回答数"]
            st.dataframe(stats)

if __name__ == "__main__":
    main()
