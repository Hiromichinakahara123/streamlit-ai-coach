import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import os
import google.generativeai as genai
import json
import pypdf # PDF読み取り用ライブラリ

# --- データベース設定 ---
DB_FILE = 'pk_study_log.db'

def init_db():
    """データベースとログテーブルを初期化する"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            topic TEXT,
            is_correct INTEGER
        )
    ''')
    conn.commit()
    conn.close()

def log_result(topic, is_correct):
    """学習結果をデータベースに記録する"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    c.execute('INSERT INTO logs (timestamp, topic, is_correct) VALUES (?, ?, ?)', 
              (timestamp, topic, int(is_correct)))
    conn.commit()
    conn.close()

def get_stats():
    """全学習ログをDataFrameとして取得する"""
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM logs", conn)
    conn.close()
    return df

# --- Google Gemini AI Configuration ---
def configure_gemini():
    """Gemini APIキーを設定する"""
    try:
        # SecretsからAPIキーを取得
        api_key = st.secrets['GEMINI_API_KEY']
        genai.configure(api_key=api_key)
        return True
    except KeyError:
        st.error("❌ Gemini APIキーが設定されていません。Streamlit Secretsに 'GEMINI_API_KEY' を設定してください。")
        return False
    except Exception as e:
        st.error(f"❌ Gemini API設定エラー: {e}")
        return False

# --- PDF処理関数 ---

@st.cache_data
def load_and_process_pdf(uploaded_file):
    """
    アップロードされたPDFファイルからテキストを抽出し、セッションステートに保存する。
    """
    if uploaded_file is None:
        return "資料がアップロードされていません。"
        
    try:
        # アップロードされたファイルをメモリ内のバイトストリームとして開く
        pdf_reader = pypdf.PdfReader(uploaded_file)
        
        full_text = ""
        for page in pdf_reader.pages:
            full_text += page.extract_text() + "\n\n"
        
        if len(full_text.strip()) < 100:
             return f"抽出されたテキストが少なすぎます（{len(full_text.strip())}文字）。PDFがテキストベースであることを確認してください。"
             
        # テキストをセッションステートに保存
        st.session_state.pdf_content = full_text
        
        # ファイル名をセッションステートに保存
        st.session_state.file_name = uploaded_file.name
        
        st.success(f"✅ 資料「{uploaded_file.name}」のテキスト抽出が完了しました。（{len(full_text.strip())}文字）")
        return full_text
        
    except Exception as e:
        st.error(f"❌ PDF処理エラー: {e}")
        return f"PDF処理中にエラーが発生しました: {e}"

# --- AI生成関数 ---

def generate_ai_problems(pdf_text, num_questions=3):
    """PDFテキストを基にAIに問題を生成させる"""
    if not pdf_text or "資料がアップロードされていません。" in pdf_text or "テキストベースであることを確認してください。" in pdf_text:
        st.error("問題生成には有効な資料が必要です。")
        return []

    system_prompt = (
        "あなたはプロの家庭教師です。提供されたPDF資料の内容を完璧に理解し、"
        "その資料の内容のみに基づいて、指定された数の問題をJSON形式で生成してください。"
        "ユーザーの学習を深めるための、難易度が高すぎない一問一答形式にしてください。"
    )
    
    user_prompt = f"""
    以下の資料の内容に基づいて、{num_questions}問の問題を生成してください。

    【資料内容】
    {pdf_text[:3000]}...（一部省略）

    【出力形式】
    必ず以下のJSON Schemaに従って出力してください。他のテキストや説明は一切含めないでください。
    """

    # JSONスキーマ定義
    response_schema = {
        "type": "ARRAY",
        "description": "資料に基づいた問題のリスト",
        "items": {
            "type": "OBJECT",
            "properties": {
                "question": {"type": "STRING", "description": "問題文"},
                "answer": {"type": "STRING", "description": "正解の簡潔な説明"},
                "explanation": {"type": "STRING", "description": "解説。正解の根拠と関連知識を含める。Markdown形式で記述し、特に重要な用語は**太字**にする。"}
            },
            "required": ["question", "answer", "explanation"]
        }
    }
    
    st.info("🤖 AIが資料を分析し、問題を作成中です... しばらくお待ちください。")
    
    try:
        client = genai.Client()
        response = client.models.generate_content(
            model='gemini-2.5-flash-preview-09-2025',
            contents=[
                {"role": "user", "parts": [{"text": user_prompt}]}
            ],
            config={
                "system_instruction": system_prompt,
                "response_mime_type": "application/json",
                "response_schema": response_schema,
                "temperature": 0.2
            }
        )
        
        # JSON文字列をパース
        problems_list = json.loads(response.text)
        
        if problems_list and isinstance(problems_list, list):
            st.success(f"🎉 AIによる {len(problems_list)} 問の問題生成が完了しました！")
            return problems_list
        else:
            st.error("❌ AIからのレスポンス形式が不正です。再試行してください。")
            return []
            
    except Exception as e:
        st.error(f"❌ AI問題生成エラー: {e}")
        st.text(f"API応答: {response.text if 'response' in locals() else 'N/A'}")
        return []

def get_ai_coaching_message(df):
    """学習履歴に基づいてAIコーチングメッセージを生成する"""
    if df.empty:
        return "まだ学習履歴がありません。問題を解いてコーチングを開始しましょう！"

    # 最新の学習記録を取得
    latest_logs = df.sort_values('timestamp', ascending=False).head(10).to_markdown()
    
    # 統計情報の計算
    stats = df.groupby('topic').agg(
        正解数=('is_correct', 'sum'),
        回答数=('id', 'count')
    )
    stats['正答率'] = stats['正解数'] / stats['回答数']
    stats_markdown = stats.to_markdown() # tabulateが必要な箇所

    system_prompt = (
        "あなたは非常に優秀な学習コーチAIです。提供された学習履歴（DataFrame）を分析し、"
        "学習者の次の行動を促すための、具体的で励ましになるアドバイスを提供してください。"
        "返答は親しみやすいトーンで、日本語で記述してください。"
    )
    
    user_prompt = f"""
    以下の学習履歴と統計情報に基づいて、学習者へのコーチングメッセージを作成してください。

    【最新の学習ログ（直近10件）】
    {latest_logs}

    【分野別 正答率統計】
    {stats_markdown}
    
    【分析とアドバイスの構造】
    1. 全体的な評価と励まし。
    2. 最も正答率が低い分野（もしあれば）を特定し、その分野を重点的に復習するよう具体的に促す。
    3. 次に解くべき問題の種類（例：AI生成問題、特定の分野）を提案する。
    """
    
    try:
        client = genai.Client()
        response = client.models.generate_content(
            model='gemini-2.5-flash-preview-09-2025',
            contents=[user_prompt],
            config={"system_instruction": system_prompt, "temperature": 0.5}
        )
        return response.text
    except Exception as e:
        return f"❌ AIコーチング生成エラー: {e}"


# --- Streamlit UI ---

def main():
    """メインアプリケーション関数"""
    
    # セッションステートの初期化
    if 'pdf_content' not in st.session_state:
        st.session_state.pdf_content = None
    if 'file_name' not in st.session_state:
        st.session_state.file_name = None
    if 'ai_problems' not in st.session_state:
        st.session_state.ai_problems = None
    if 'ai_idx' not in st.session_state:
        st.session_state.ai_idx = 0
    if 'coaching_message' not in st.session_state:
        st.session_state.coaching_message = "問題を解いてAIコーチングを開始しましょう！"
    if 'pdf_uploaded_key' not in st.session_state:
        st.session_state.pdf_uploaded_key = 0

    st.set_page_config(page_title="AIコーチングアプリ", layout="centered")
    
    st.title("📚 AIコーチング 学習アプリ")
    
    # データベースの初期化
    init_db()

    # Gemini API設定チェック
    if not configure_gemini():
        return

    # 全学習ログの取得
    df = get_stats()

    # タブの作成
    tab1, tab2, tab3 = st.tabs(["資料設定", "問題演習", "コーチング"])

    # --- Tab 1: 資料設定 ---
    with tab1:
        st.header("ステップ1: PDF資料のアップロード")

        # ファイルアップローダーの設置
        uploaded_file = st.file_uploader(
            "学習に使いたいPDFファイルをアップロードしてください。", 
            type="pdf",
            key=st.session_state.pdf_uploaded_key
        )
        
        if uploaded_file is not None:
            # ファイルがアップロードされたら処理
            with st.spinner(f"資料「{uploaded_file.name}」を処理中..."):
                pdf_text = load_and_process_pdf(uploaded_file)
            
            # 処理結果の表示
            if st.session_state.pdf_content:
                st.success(f"現在処理中の資料: **{st.session_state.file_name}**")
                
                # 問題生成ボタン
                if st.button("この資料でAI問題を生成する", key="generate_problems"):
                    st.session_state.ai_problems = None # 既存の問題をリセット
                    st.session_state.ai_idx = 0
                    
                    with st.spinner("🚀 AIが問題を作成中..."):
                        problems = generate_ai_problems(st.session_state.pdf_content, num_questions=5)
                        st.session_state.ai_problems = problems
                        
                    if st.session_state.ai_problems:
                        st.session_state.pdf_uploaded_key += 1 # アップローダーをリセットして次のアップロードに備える
                        st.rerun() # タブ2に移動してもらうため再実行

            else:
                 # PDF処理が失敗した場合のメッセージはload_and_process_pdf内で表示済み
                 pass

        else:
            # 既にデータがある場合は表示
            if st.session_state.pdf_content:
                st.info(f"現在、資料「**{st.session_state.file_name}**」がセットされています。")
                if st.button("資料をクリアして新しいファイルをアップロード"):
                    st.session_state.pdf_content = None
                    st.session_state.file_name = None
                    st.session_state.ai_problems = None
                    st.session_state.ai_idx = 0
                    st.session_state.pdf_uploaded_key += 1
                    st.rerun()

    # --- Tab 2: 問題演習 ---
    with tab2:
        st.header("ステップ2: 問題演習")
        
        if not st.session_state.pdf_content:
            st.warning("先に「資料設定」タブで学習資料（PDF）をアップロードしてください。")
        elif not st.session_state.ai_problems:
            st.warning("資料がセットされました。「資料設定」タブで「AI問題を生成する」ボタンを押してください。")
        else:
            # AI生成問題の表示と解答
            problems = st.session_state.ai_problems
            current_index = st.session_state.ai_idx
            total_problems = len(problems)

            if current_index < total_problems:
                st.subheader(f"AI生成問題 {current_index + 1} / {total_problems}")
                
                q = problems[current_index]
                
                st.markdown(f"**問題:** {q['question']}")
                
                key_suffix = f"{current_index}"
                with st.form(key=f"ai_question_form_{key_suffix}"):
                    user_answer = st.text_area("あなたの解答を入力してください", key=f"user_answer_{key_suffix}")
                    submitted = st.form_submit_button("解答をチェック")

                if submitted:
                    # AIに採点させる（簡易的に正解と一致するかで判断）
                    if 'is_correct' not in st.session_state or st.session_state.ai_idx != current_index:
                        
                        # 厳密なAI採点ロジックは省略し、今回はユーザーに自己採点させるか、簡易的に一致確認
                        
                        # 簡易的な正否判定（今回はヒントとして正解を表示）
                        st.markdown(f"**💡 正解:** `{q['answer']}`")

                        # ユーザーによる採点ボタン
                        col_correct, col_incorrect = st.columns(2)
                        if col_correct.button("⭕ 正解だった", key=f"btn_correct_{key_suffix}"):
                            st.session_state.is_correct = True
                            st.success("🎉 正解です！")
                            log_result("AI生成問題", 1)
                        if col_incorrect.button("❌ 不正解だった", key=f"btn_incorrect_{key_suffix}"):
                            st.session_state.is_correct = False
                            st.error("❌ 不正解です。")
                            log_result("AI生成問題", 0)
                        
                        st.info("💡 **解説**")
                        st.markdown(q['explanation'], unsafe_allow_html=True)
                        
                        # 正否判定が終わったら次の問題ボタンを表示
                        if st.session_state.get('is_correct') is not None:
                            if st.button("次の問題へ", key=f"ai_next_{key_suffix}"):
                                st.session_state.ai_idx += 1
                                st.session_state.is_correct = None # 状態リセット
                                st.rerun()
                    
            else:
                st.success("全てのAI生成問題が終了しました！")
                if st.button("新しい問題を生成する"):
                    del st.session_state.ai_problems
                    st.rerun()

    # --- Tab 3: Stats & Coaching ---
    with tab3:
        st.header("学習履歴とAIコーチング")
        
        # DataFrameの再取得（最新のログを反映）
        df = get_stats() 
        
        if df.empty:
            st.warning("まだ学習データがありません。「問題演習」タブで問題を解いてみましょう！")
        else:
            col1, col2 = st.columns([2, 1])
            
            # --- 統計情報 ---
            with col1:
                st.subheader("分野別 正答率")
                stats = df.groupby('topic').agg(
                    正解数=('is_correct', 'sum'),
                    回答数=('id', 'count')
                )
                stats['正答率'] = (stats['正解数'] / stats['回答数']).map('{:.1%}'.format)
                
                try:
                    # to_markdown()にはtabulateが必要 (requirements.txtで追加済み)
                    st.dataframe(stats.style.background_gradient(cmap='RdYlGn', subset=['正答率'], vmin=0, vmax=1)) 
                except Exception:
                    # tabulateがまだインストールされていない場合のフォールバック
                    st.dataframe(stats)

            # --- AIコーチング ---
            with col2:
                st.subheader("AIコーチングメッセージ")
                # コーチングメッセージの生成/更新
                if st.button("AIコーチングを更新", key="update_coaching"):
                    with st.spinner("AIコーチが分析中..."):
                         st.session_state.coaching_message = get_ai_coaching_message(df)

                st.info(st.session_state.coaching_message)
                
    
if __name__ == '__main__':
    main()
