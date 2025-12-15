import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import os
import google.generativeai as genai
import json
import pypdf
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import time

# --- Configuration ---
# Gemini APIキーを環境変数から取得
# st.secretsやos.environを使うのが一般的ですが、ここでは仮のAPIキー設定とします
# 実際の環境に合わせて変更してください。
if 'GEMINI_API_KEY' in os.environ:
    genai.configure(api_key=os.environ['GEMINI_API_KEY'])
elif 'GEMINI_API_KEY' in st.secrets:
    genai.configure(api_key=st.secrets['GEMINI_API_KEY'])
else:
    # APIキーが設定されていない場合の代替（開発用）
    # 実際のAPIキーを設定してください
    st.error("Gemini APIキーが設定されていません。環境変数またはst.secretsを設定してください。")
    # genai.configure(api_key="YOUR_API_KEY") 
    pass 

# --- Database Setup ---
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
    """データベースから全ログを取得し、Pandas DataFrameとして返す"""
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM logs", conn)
    conn.close()
    return df

# --- Google Drive & PDF Handling ---
CREDENTIALS_FILE = 'service_account.json'

def authenticate_drive():
    """サービスアカウントキーを使用してGoogle Driveに認証する"""
    try:
        if not os.path.exists(CREDENTIALS_FILE):
             st.error(f"認証ファイル '{CREDENTIALS_FILE}' が見つかりません。")
             return None
             
        creds = service_account.Credentials.from_service_account_file(
            CREDENTIALS_FILE,
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        service = build('drive', 'v3', credentials=creds)
        return service
    except Exception as e:
        st.error(f"Google Drive認証エラー: {e}")
        return None

def download_pdf_from_drive(service, file_id):
    """Google DriveからPDFをダウンロードし、テキストを抽出する"""
    try:
        # ダウンロード
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        with st.spinner("Google Driveからファイルをダウンロード中..."):
            while done is False:
                status, done = downloader.next_chunk()
        
        # テキスト抽出
        pdf_reader = pypdf.PdfReader(fh)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or "" # extract_textがNoneを返す可能性に対応
        
        if not text.strip():
            st.error("PDFからテキストを抽出できませんでした。ファイル形式を確認してください。")
            return None
        
        return text.strip()
        
    except Exception as e:
        st.error(f"ファイルのダウンロードまたは抽出中にエラーが発生しました。ファイルIDとアクセス権を確認してください: {e}")
        return None

# --- Gemini API Functions ---
def generate_problems_from_text(document_text):
    """ドキュメントテキストを元に問題セットを生成する"""
    
    # JSONスキーマを定義 (Geminiの構造化出力機能を使用)
    problem_schema = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "question": {"type": "STRING", "description": "PDFの内容に基づいた、学習者向けの短い一問一答形式の質問。"},
                "answer": {"type": "STRING", "description": "質問に対する正確な正解。"},
                "explanation": {"type": "STRING", "description": "正解の根拠と、PDFのどの部分に対応するかを説明する丁寧な解説。"}
            },
            "required": ["question", "answer", "explanation"]
        }
    }
    
    system_prompt = (
        "あなたはプロの教育者であり、生成AIです。提供されたドキュメントのテキストを完全に理解し、"
        "その内容に基づいた、学生が確実に理解すべき重要事項に関する一問一答形式の問題を5問生成してください。 "
        "問題、正解、解説を必ず日本語で、指定されたJSONスキーマに従って出力してください。 "
        "解説は、なぜその答えになるのか、ドキュメントの内容に言及して詳細に記述してください。"
    )

    user_query = f"このドキュメントのテキストから、学生向けの一問一答形式の問題を5問、JSON形式で生成してください:\n\n---\n{document_text[:15000]}" # 15000文字に制限
    
    try:
        model_name = 'gemini-2.5-flash'
        client = genai.Client()
        
        # NOTE: プレビューのため、ここではダミーの応答を使用します
        # 実際の動作では、このダミー応答はコメントアウトされます。
        # with st.spinner("🧠 AIが問題を生成中... (数秒〜数十秒かかります)"):
        #     response = client.models.generate_content(
        #         model=model_name,
        #         contents=user_query,
        #         config={
        #             "system_instruction": system_prompt,
        #             "response_mime_type": "application/json",
        #             "response_schema": problem_schema,
        #         },
        #     )
        # json_text = response.text.strip()
        
        # --- プレビュー用ダミー応答 ---
        dummy_problems = [
            {"question": "Streamlitアプリケーションのレイアウトを構築するために使用されるPythonライブラリは何ですか？", "answer": "Streamlit", "explanation": "このアプリはStreamlitフレームワークを使っており、UIの構築に不可欠です。"},
            {"question": "学習結果を永続的に記録するために使用されるデータベースファイルの名前は何ですか？", "answer": "pk_study_log.db", "explanation": "init_db関数でこの名前のSQLiteファイルが初期化され、学習履歴が記録されます。"},
            {"question": "Google Driveからファイルをダウンロードするために必要な認証ファイルの名称は何ですか？", "answer": "service_account.json", "explanation": "Google Drive認証にはサービスアカウントキーが必要です。"},
            {"question": "AIコーチングメッセージを生成するのに使われているGoogleの生成AIモデルは何ですか？", "answer": "gemini-2.5-flash", "explanation": "get_ai_coaching_message関数内で指定されています。"},
            {"question": "ユーザーの正誤判定で、正解が記録される際のログテーブルのis_correct列の値は何ですか？", "answer": "1", "explanation": "log_result関数内で、正解の場合に1が、不正解の場合に0が記録されます。"}
        ]
        return dummy_problems
        # -----------------------------

        # problems = json.loads(json_text)
        # return problems
        
    except Exception as e:
        # st.error(f"Gemini APIによる問題生成エラー: {e}")
        st.error(f"プレビューモード: Gemini API接続エラーを無視し、ダミーデータを使用します。")
        return dummy_problems # 失敗時もダミーを返す

def get_ai_coaching_message(stats_df):
    """学習履歴データからAIコーチングメッセージを生成する"""
    
    if stats_df.empty:
        return "まだ学習データがないため、一般的な学習アドバイスを提供します。まずは問題を解いてみましょう！"
    
    # 統計情報の整形
    stats = stats_df.groupby('topic').agg(
        正解数=('is_correct', 'sum'),
        回答数=('id', 'count')
    )
    stats['正答率'] = stats['正解数'] / stats['回答数']
    stats_markdown = stats.to_markdown()
    
    system_prompt = (
        "あなたは専門のAI学習コーチです。提供された学習履歴の統計データ（Markdown形式）を分析し、"
        "この学生の学習傾向、強み、弱点を特定し、今後の学習で最も効果的な改善点を、親身になって日本語で、"
        "3つの具体的なアドバイスにまとめて提供してください。"
    )

    user_query = f"以下の学習履歴統計を分析し、個別指導メッセージを生成してください:\n\n---\n{stats_markdown}"
    
    try:
        model_name = 'gemini-2.5-flash'
        client = genai.Client()
        
        # NOTE: プレビューのため、ここではダミーの応答を使用します
        # with st.spinner("🗣️ AIコーチが分析中..."):
        #     response = client.models.generate_content(
        #         model=model_name,
        #         contents=user_query,
        #         config={"system_instruction": system_prompt},
        #     )
        # return response.text
        
        # --- プレビュー用ダミー応答 ---
        time.sleep(1) # スピナー表示のため
        return (
            "**個別指導メッセージ (ダミー)**\n\n"
            "1. **基礎固め:** 全体の正答率がまだ50%を下回っているようです。まずは、最も正答率の低いトピックに焦点を当て、その基礎概念をPDF資料で再確認しましょう。\n"
            "2. **キーワード学習:** 一問一答形式で不正解が多い場合、キーとなる用語や定義の理解が不十分かもしれません。問題の答えとなるキーワードを紙に書き出し、視覚的に覚えることを試みてください。\n"
            "3. **時間配分:** 連続で問題を解く時間を決め、集中力を高める練習をしましょう。短時間で集中して取り組むことで、記憶の定着が促進されます。"
        )
        # -----------------------------
        
    except Exception as e:
        # st.error(f"AIコーチングメッセージの生成エラー: {e}")
        return "AIコーチングメッセージの生成中にエラーが発生しました。"


# --- Streamlit Application Main ---
def main():
    """Streamlitアプリケーションのメイン関数"""
    
    init_db()
    st.set_page_config(page_title="PKラーニングAIコーチ", layout="wide")
    st.title("📚 PKラーニング AIコーチ")

    # --- 共通のデータベース情報取得 ---
    df = get_stats()

    # --- Session State 初期化 ---
    if 'ai_problems' not in st.session_state:
        st.session_state.ai_problems = None
    if 'ai_idx' not in st.session_state:
        st.session_state.ai_idx = 0
    if 'pdf_name' not in st.session_state:
        st.session_state.pdf_name = "未設定"
    if 'show_result' not in st.session_state:
        st.session_state.show_result = False

    # --- 開発者/管理者用サイドバー設定 ---
    st.sidebar.header("管理者設定 (資料切り替え)")
    
    # 問題が未生成、または新しい資料に切り替えたい場合
    is_problems_generated = st.session_state.ai_problems is not None
    
    if not is_problems_generated:
        st.sidebar.warning("現在、問題が設定されていません。資料を読み込んでください。")

    
    with st.sidebar.form("admin_form"):
        st.subheader("Google Drive 資料設定")
        # 開発者様がPDFのファイルIDを入力
        new_file_id = st.text_input(
            "PDFファイルID (Google Drive)",
            key="file_id_input",
            placeholder="例: 1a2b3c4d5e6f..."
        )
        submitted = st.form_submit_button("この資料を読み込み、問題を作成する")
    
        if submitted and new_file_id:
            # 1. 認証とダウンロード
            drive_service = authenticate_drive()
            
            # --- プレビュー用ダミー分岐 ---
            if new_file_id == "dummy-id-for-preview":
                 text_content = "プレビュー用のダミーテキストです。Streamlit、SQLite、Gemini APIの3つの技術を組み合わせています。"
            elif drive_service:
            # ---------------------------
                text_content = download_pdf_from_drive(drive_service, new_file_id)
                
            if text_content:
                # 2. 問題生成
                problems = generate_problems_from_text(text_content)
                
                if problems:
                    # 3. 成功したら状態を更新
                    st.session_state.ai_problems = problems
                    st.session_state.ai_idx = 0 # インデックスをリセット
                    st.session_state.pdf_name = f"資料ID: {new_file_id[:8]}..."
                    st.sidebar.success(f"✅ 問題 {len(problems)} 問を正常に生成しました。")
                    # 成功したため、メインコンテンツを再描画
                    st.rerun() 
                else:
                    st.sidebar.error("問題生成に失敗しました。Gemini APIの設定を確認してください。")


    st.sidebar.markdown("---")
    st.sidebar.info(f"💡 **現在の学習資料:** {st.session_state.pdf_name}")
    st.sidebar.markdown(
        """
        このサイドバーは、教材の切り替えを行う管理者（開発者）向けです。
        学生ユーザーは、メイン画面のタブのみを使用します。
        """
    )


    # --- Main Content Tabs (学生ユーザー向け) ---
    tab1, tab2 = st.tabs(["🧠 問題演習", "📊 学習履歴とAIコーチング"])


    # --- Tab 1: Problem Solving ---
    with tab1:
        st.header("🧠 AI生成問題演習")
        
        if st.session_state.ai_problems is None:
            st.warning("現在、学習資料が設定されていません。管理者がサイドバーから資料を読み込むまでお待ちください。")
            st.session_state.pdf_name = "未設定"
            
        else:
            total_problems = len(st.session_state.ai_problems)
            current_idx = st.session_state.ai_idx
            
            if current_idx < total_problems:
                q = st.session_state.ai_problems[current_idx]
                key_suffix = f"{current_idx}"

                st.subheader(f"問題 {current_idx + 1} / {total_problems}")
                st.markdown(f"**テーマ:** {st.session_state.pdf_name}")
                
                # 問題表示とフォーム
                with st.form(key=f"ai_question_form_{key_suffix}"):
                    st.markdown(f"#### 質問:")
                    st.markdown(f"**{q['question']}**")

                    user_answer = st.text_input(
                        "あなたの答えを入力してください",
                        key=f"ai_answer_input_{key_suffix}",
                        label_visibility="collapsed"
                    )
                    
                    submitted = st.form_submit_button("解答する")
                    
                    if submitted:
                        # 正誤判定 (大文字・小文字、前後のスペースを無視)
                        correct_answer = q['answer'].strip().lower()
                        user_input = user_answer.strip().lower()
                        
                        # 簡易的な判定（完全一致）
                        is_correct = (user_input == correct_answer)
                        
                        if is_correct:
                            st.success("🎉 正解です！")
                            log_result("AI生成問題", 1)
                        else:
                            st.error(f"❌ 不正解... 正解は「{q['answer']}」です。")
                            log_result("AI生成問題", 0)
                        
                        st.info("💡 **解説**")
                        st.markdown(q['explanation'], unsafe_allow_html=True)

                if submitted:
                     # 解答済みの場合は「次の問題へ」ボタンを表示
                     if st.button("次の問題へ", key=f"ai_next_{key_suffix}"):
                         st.session_state.ai_idx += 1
                         st.rerun()

            else:
                st.success("全てのAI生成問題が終了しました！お疲れ様でした。")
                if st.button("もう一度最初から解く"):
                    st.session_state.ai_idx = 0
                    st.rerun()


    # --- Tab 2: Stats & Coaching ---
    with tab2:
        st.header("📊 学習履歴とAIコーチング")
        
        if df.empty:
            st.warning("まだ学習データがありません。「問題演習」タブで問題を解いてみましょう！")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("分野別 正答率")
                stats = df.groupby('topic').agg(
                    正解数=('is_correct', 'sum'),
                    回答数=('id', 'count')
                )
                stats['正答率'] = stats['正解数'] / stats['回答数']
                stats['正答率(%)'] = (stats['正答率'] * 100).round(1)
                
                # 正答率の棒グラフ表示
                st.bar_chart(stats, y='正答率(%)', color='#4CAF50')

                # 詳細データフレーム
                st.markdown("##### 詳細データ")
                st.dataframe(stats[['回答数', '正解数', '正答率(%)']].sort_values(by='正答率(%)', ascending=False), 
                             use_container_width=True)

            with col2:
                st.subheader("全体統計")
                total_correct = df['is_correct'].sum()
                total_attempts = len(df)
                overall_rate = (total_correct / total_attempts) if total_attempts > 0 else 0

                st.metric(
                    label="全体正答率", 
                    value=f"{overall_rate * 100:.1f}%", 
                    delta=f"{total_correct} 問 / {total_attempts} 問"
                )
                
                if st.button("AIコーチングを更新"):
                    st.session_state.coaching_message = None
                    
                # AIコーチングの生成と表示
                if 'coaching_message' not in st.session_state or st.session_state.coaching_message is None:
                    st.session_state.coaching_message = get_ai_coaching_message(df)

                st.subheader("AIコーチからのアドバイス")
                st.info(st.session_state.coaching_message)


if __name__ == "__main__":
    main()