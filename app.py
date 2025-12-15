import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime
import os
import json
import io
import google.generativeai as genai

# ---------- File parsing ----------
import pypdf
from docx import Document
from pptx import Presentation

# ---------- Gemini ----------
import google.generativeai as genai


# =====================================================
# DB
# =====================================================

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


# =====================================================
# Gemini
# =====================================================

def configure_gemini():
    api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("❌ GEMINI_API_KEY が設定されていません")
        return False
    genai.configure(api_key=api_key)
    return True


# =====================================================
# File extraction
# =====================================================

def extract_from_pdf(data):
    reader = pypdf.PdfReader(io.BytesIO(data))
    texts = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            texts.append(f"【ページ {i+1}】\n{text}")
    return "\n\n".join(texts)

def extract_from_docx(data):
    doc = Document(io.BytesIO(data))
    texts = []
    for p in doc.paragraphs:
        if p.style.name.startswith("Heading"):
            texts.append(f"\n## {p.text}\n")
        else:
            texts.append(p.text)
    return "\n".join(texts)

def extract_from_xlsx(data):
    xl = pd.ExcelFile(io.BytesIO(data))
    texts = []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        texts.append(f"\n## シート: {sheet}\n")
        texts.append(df.to_csv(index=False))
    return "\n".join(texts)

def extract_from_pptx(data):
    prs = Presentation(io.BytesIO(data))
    texts = []
    for i, slide in enumerate(prs.slides):
        texts.append(f"\n## スライド {i+1}\n")
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                texts.append(shape.text)
    return "\n".join(texts)

def extract_text(uploaded_file):
    data = uploaded_file.read()
    ext = uploaded_file.name.split(".")[-1].lower()

    if ext == "pdf":
        return extract_from_pdf(data)
    if ext == "docx":
        return extract_from_docx(data)
    if ext == "xlsx":
        return extract_from_xlsx(data)
    if ext == "pptx":
        return extract_from_pptx(data)

    raise ValueError("未対応のファイル形式です")


# =====================================================
# AI problem generation
# =====================================================

def safe_json_load(text):
    try:
        return json.loads(text)
    except Exception:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1:
            return json.loads(text[start:end+1])
        raise

def generate_ai_problems(text, n=5):
    model = genai.GenerativeModel("gemini-flash-latest")

    system_prompt = """
あなたは薬剤師国家試験対策問題を作成する教育AIです。

【厳守事項】
・提供資料の内容のみから作問する
・薬剤師国家試験形式（5択単一選択）とする
・正解は必ず1つ
・誤選択肢は知識不足で選びやすいものにする
・JSONのみ出力
"""

    prompt = f"""
以下の資料から {n} 問の五肢択一問題を作成してください。

出力形式:
[
  {{
    "question": "...",
    "choices": {{
      "A": "...",
      "B": "...",
      "C": "...",
      "D": "...",
      "E": "..."
    }},
    "correct": "A",
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


資料:
{text[:3000]}
"""

    response = model.generate_content(
        [system_prompt, prompt],
        generation_config={"temperature": 0.2}
    )

    return safe_json_load(response.text)


def get_ai_coaching_message(df):
    if df.empty:
        return "まだ学習履歴がありません。"

    latest_csv = (
        df.sort_values("timestamp", ascending=False)
          .head(10)[["timestamp", "topic", "is_correct"]]
          .to_csv(index=False)
    )

    stats = df.groupby("topic").agg(
        正解数=("is_correct", "sum"),
        回答数=("id", "count")
    )
    stats["正答率"] = stats["正解数"] / stats["回答数"]
    stats_csv = stats.to_csv()

    model = genai.GenerativeModel("gemini-flash-latest")

    prompt = f"""
以下の学習履歴と統計（CSV形式）を分析し、学習者への具体的なコーチングメッセージを日本語で作成してください。

【直近ログ】
{latest_csv}

【分野別統計】
{stats_csv}
"""

    response = model.generate_content(prompt)
    return response.text



# =====================================================
# UI
# =====================================================

def main():
    st.set_page_config("AIコーチング学習アプリ", layout="centered")
    st.title("📚 AIコーチング学習アプリ")

    init_db()
    if not configure_gemini():
        return

    if "text" not in st.session_state:
        st.session_state.text = None
    if "problems" not in st.session_state:
        st.session_state.problems = []
    if "idx" not in st.session_state:
        st.session_state.idx = 0

    tab1, tab2, tab3 = st.tabs(["資料", "問題演習", "コーチング"])

    # ---------- 資料 ----------
    with tab1:
        file = st.file_uploader(
            "資料をアップロード",
            type=["pdf", "docx", "xlsx", "pptx"]
        )
        if file:
            with st.spinner("資料解析中..."):
                st.session_state.text = extract_text(file)
            st.success("資料を読み込みました")

            if st.button("AI問題を生成"):
                st.session_state.problems = generate_ai_problems(st.session_state.text)
                st.session_state.idx = 0
                st.rerun()

    # ---------- 問題 ----------
    with tab2:
        if not st.session_state.problems:
            st.info("問題がまだありません")
            return

        p = st.session_state.problems[st.session_state.idx]
        st.subheader(f"問題 {st.session_state.idx + 1}")
        st.markdown(p["question"])
        st.markdown("---")
        st.markdown(f"**正解:** {p['answer']}")
        st.markdown(p["explanation"])

        p = st.session_state.problems[st.session_state.idx]

st.subheader(f"問題 {st.session_state.idx + 1}")
st.markdown(p["question"])

choice = st.radio(
    "選択肢",
    options=list(p["choices"].keys()),
    format_func=lambda x: f"{x}: {p['choices'][x]}"
)

if st.button("解答する"):
    is_correct = (choice == p["correct"])
    log_result("AI生成問題", is_correct)

    if is_correct:
        st.success("正解です 🎉")
    else:
        st.error(f"不正解です。正解は {p['correct']} です。")

    st.markdown("### 解説")
    st.markdown(p["explanation"])

    if st.button("次の問題へ"):
        st.session_state.idx += 1
        st.rerun()


    # ---------- コーチング ----------
    with tab3:
        df = get_stats()
        if df.empty:
            st.info("学習履歴がありません")
        else:
            st.subheader("分野別 正答率")
            stats = df.groupby("topic").agg(
                正解数=("is_correct", "sum"),
                回答数=("id", "count")
            )
            stats["正答率"] = stats["正解数"] / stats["回答数"]
            st.dataframe(stats, width="stretch")

            if st.button("AIコーチングを更新"):
                with st.spinner("分析中..."):
                    msg = get_ai_coaching_message(df)
                st.info(msg)


if __name__ == "__main__":
    main()







