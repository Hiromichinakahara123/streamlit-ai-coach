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
# Gemini APIキーを環境変数またはst.secretsから取得
try:
    if 'GEMINI_API_KEY' in os.environ:
        genai.configure(api_key=os.environ['GEMINI_API_KEY'])
    elif 'GEMINI_API_KEY' in st.secrets:
        genai.configure(api_key=st.secrets['GEMINI_API_KEY'])
    else:
        st.error("Gemini APIキーが設定されていません。環境変数またはst.secretsを設定してください。")
except Exception as e:
    st.error(f"Gemini API設定エラー: {e}")

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
def get_drive_credentials():
    """Streamlit secretsまたはローカルファイルから認証情報を取得する"""
    # 既存のロジックを保持
    if 'service_account_json' in st.secrets:
        sa_json_str = st.secrets['service_account_json']
        return service_account.Credentials.from_service_account_info(
            json.loads(sa_json_str),
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
    elif os.path.exists('service_account.json'):
        return service_account.Credentials.from_service_account_file(
            'service_account.json',
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
    else:
        # デプロイ環境では必須
        return None

def authenticate_drive():
    """サービスアカウントキーを使用してGoogle Driveに認証する"""
    try:
        creds = get_drive_credentials()
        if creds:
            service = build('drive', 'v3', credentials=creds)
            return service
        else:
            return None
    except Exception as e:
        st.error(f"Google Drive認証エラー: {e}")
        return None

@st.cache_data(show_spinner=False)
def download_single_pdf(service, file_id):
    """単一のPDFをダウンロードし、テキストを抽出する（キャッシュ対応）"""
    try:
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        
        while done is False:
            status, done = downloader.next_chunk()
        
        pdf_reader = pypdf.PdfReader(fh)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        
        return text.strip()
        
    except Exception as e:
        # エラーは親関数で処理するため、ここではNoneを返す
        st.warning(f"ファイルのダウンロードまたは抽出中にエラーが発生しました（ID: {file_id}）。権限を確認してください。")
        return None

def list_pdf_files_in_folder(service, folder_id):
    """指定されたGoogle Driveフォルダ内のPDFファイルの一覧を取得する"""
    # Google Drive APIの検索クエリを使用: 'フォルダID' in parents AND mimeType='application/pdf'
    query = (
        f"'{folder_id}' in parents and "
        "mimeType='application/pdf' and "
        "trashed=false"
    )
    try:
        results = service.files().list(
            q=query,
            fields="files(id, name)",
            pageSize=100  # 一度に取得するファイル数 (最大100)
        ).execute()
        items = results.get('files', [])
        return items
    except Exception as e:
        st.error(f"フォルダ内のファイル一覧取得エラー: {e}")
        return []

def process_folder_files(service, folder_id):
    """フォルダ内の全PDFを処理し、テキストを結合する"""
    files = list_pdf_files_in_folder(service, folder_id)
    if not files:
        return None, None

    combined_text = ""
    
    with st.spinner(f"📁 フォルダ内の {len(files)} 件のPDFファイルを処理中..."):
        for i, file in enumerate(files):
            file_id = file['id']
            file_name = file['name']
            st.info(f"({i+1}/{len(files)}) ファイル '{file_name}' のテキストを抽出中...")

            # 単一ファイルのテキストをダウンロード
            text = download_single_pdf(service, file_id)
            
            if text:
                # ファイルごとに区切り文字を入れてテキストを結合
                combined_text += f"\n\n--- DOCUMENT START: {file_name} ---\n\n{text}"
            else:
                st.warning(f"ファイル '{file_name}' のテキスト抽出をスキップしました。")
            
    if not combined_text.strip():
        st.error("すべてのPDFからテキストを抽出できませんでした。フォルダの内容と権限を確認してください。")
        return None, None
    
    # 資料名として、フォルダIDとファイル数を使用
    folder_name = f"資料フォルダ: {folder_id[:8]}... ({len(files)}ファイル)"
    
    return combined_text.strip(), folder_name

# --- Gemini API Functions (変更なし) ---
def generate_problems_from_text(document_text):
    """ドキュメントテキストを元に問題セットを生成する"""
    # ... (既存の generate_problems_from_text 関数は変更なし) ...
    problem_schema = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "question": {"type": "STRING", "description": "PDFの内容に基づいた、学習者向けの短い一問一答形式の質問。"},
                "answer": {"type": "STRING", "description": "質問に対する正確な正解。"},
                "explanation": {"type": "STRING", "description": "正解の根拠と、ドキュメントのどの部分に対応するかを説明する丁寧な解説。"}
            },
            "required": ["question", "answer", "explanation"]
        }
    }
    
    system_prompt = (
        "あなたはプロの教育者であり、生成AIです。提供されたドキュメントのテキストを完全に理解し、"
        "その内容に基づいた、学生が確実に理解すべき重要事項に関する一問一答形式の問題を5問生成してください。 "
        "問題、正解、解説を必ず日本語で、指定されたJSONスキーマに従って出力してください。 "
        "解説は、なぜその答えになるのか、ドキュメントの内容に言及して詳細に記述してください。"
        "入力テキストが複数のファイルから結合されている場合、それが一体の資料であると見なして問題を生成してください。"
    )

    # 15000文字に制限（Geminiの入力制限とパフォーマンスのため）
    user_query = f"このドキュメントのテキストから、学生向けの一問一答形式の問題を5問、JSON形式で生成してください:\n\n---\n{document_text[:15000]}"
    
    try:
        model_name = 'gemini-2.5-flash'
        client = genai.Client()
        
        # APIキーが設定されていない場合の代替
        if not ('GEMINI_API_KEY' in os.environ or 'GEMINI_API_KEY' in st.secrets):
            dummy_problems = [
                {"question": "フォルダIDで複数のPDFを読み込む機能を追加するために使ったAPIは何ですか？", "answer": "Google Drive API", "explanation": "Google Drive APIの`files().list`メソッドを使用して、フォルダ内のファイルを検索しています。"},
                {"question": "複数のPDFから抽出したテキストは、どのように結合されますか？", "answer": "区切り文字を挟んで結合される", "explanation": "コードでは、`--- DOCUMENT START: [ファイル名] ---`という区切り文字を使って、ファイルごとのテキストを一つにまとめています。"},
                {"question": "Streamlit Cloudに設定が必要な2つのSecretsは何ですか？", "answer": "GEMINI_API_KEYとservice_account_json", "explanation": "これらはAPIアクセスとGoogle Driveアクセスに必要な機密情報です。"}
            ]
            return dummy_problems
        
        with st.spinner("🧠 AIが問題を生成中... (数秒〜数十秒かかります)"):
            response = client.models.generate_content(
                model=model_name,
                contents=user_query,
                config={
                    "system_instruction": system_prompt,
                    "response_mime_type": "application/json",
                    "response_schema": problem_schema,
                },
            )
            
        json_text = response.text.strip()
        problems = json.loads(json_text)
        return problems
        
    except Exception as e:
        st.error(f"Gemini APIによる問題生成エラー: {e}")
        return None

def get_ai_coaching_message(stats_df):
    """学習履歴データからAIコーチングメッセージを生成する (変更なし)"""
    # ... (既存の get_ai_coaching_message 関数は変更なし) ...
    if stats_df.empty:
        return "まだ学習データがないため、一般的な学習アドバイスを提供します。まずは問題を解いてみましょう！"
    
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
        
        with st.spinner("🗣️ AIコーチが分析中..."):
            response = client.models.generate_content(
                model=model_name,
                contents=user_query,
                config={"system_instruction": system_prompt},
            )
            
        return response.text
        
    except Exception as e:
        st.error(f"AIコーチングメッセージの生成エラー: {e}")
        return "AIコーチングメッセージの生成中にエラーが発生しました。"

# --- Streamlit Application Main ---
def main():
    """Streamlitアプリケーションのメイン関数"""
    
    init_db()
    st.set_page_config(page_title="PKラーニングAIコーチ", layout="wide")
    st.title("📚 PKラーニング AIコーチ")

    df = get_stats()

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
    
    is_problems_generated = st.session_state.ai_problems is not None
    
    if not is_problems_generated:
        st.sidebar.warning("現在、問題が設定されていません。資料フォルダを読み込んでください。")

    
    with st.sidebar.form("admin_form"):
        st.subheader("Google Drive 資料フォルダ設定")
        # フォルダIDを入力するように変更
        new_folder_id = st.text_input(
            "フォルダID (Google Drive)",
            key="folder_id_input",
            placeholder="例: 1fX8Y..."
        )
        submitted = st.form_submit_button("この資料フォルダを読み込み、問題を作成する")
    
        if submitted and new_folder_id:
            drive_service = authenticate_drive()
            
            if drive_service:
                # フォルダ内の全PDFを処理する関数を呼び出す
                text_content, folder_name = process_folder_files(drive_service, new_folder_id)
                
                if text_content and folder_name:
                    problems = generate_problems_from_text(text_content)
                    
                    if problems:
                        st.session_state.ai_problems = problems
                        st.session_state.ai_idx = 0
                        st.session_state.pdf_name = folder_name
                        st.sidebar.success(f"✅ フォルダ内の資料から問題 {len(problems)} 問を正常に生成しました。")
                        st.rerun() 
                    else:
                        st.sidebar.error("問題生成に失敗しました。Gemini APIの設定を確認してください。")
            else:
                 st.sidebar.error("Google Drive認証が確立できませんでした。Secretsを確認してください。")


    st.sidebar.markdown("---")
    st.sidebar.info(f"💡 **現在の学習資料:** {st.session_state.pdf_name}")
    st.sidebar.markdown(
        """
        **重要:** ここには**フォルダ ID**を入力してください。
        このフォルダ内の**すべてのPDFファイル**からテキストが抽出され、AIが問題を生成します。
        """
    )


    # --- Main Content Tabs (学生ユーザー向け) ---
    tab1, tab2 = st.tabs(["🧠 問題演習", "📊 学習履歴とAIコーチング"])


    # --- Tab 1: Problem Solving ---
    with tab1:
        st.header("🧠 AI生成問題演習")
        
        if st.session_state.ai_problems is None:
            st.warning("現在、学習資料が設定されていません。管理者がサイドバーから資料フォルダを読み込むまでお待ちください。")
            st.session_state.pdf_name = "未設定"
            
        else:
            total_problems = len(st.session_state.ai_problems)
            current_idx = st.session_state.ai_idx
            
            if current_idx < total_problems:
                q = st.session_state.ai_problems[current_idx]
                key_suffix = f"{current_idx}"

                st.subheader(f"問題 {current_idx + 1} / {total_problems}")
                st.markdown(f"**テーマ:** {st.session_state.pdf_name}")
                
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
                        # 簡易的なチェックロジック（厳密な正誤判定は別途AIで実装可能だが、ここでは単純な一致を使用）
                        correct_answer = q['answer'].strip().lower()
                        user_input = user_answer.strip().lower()
                        
                        # 完全に一致しなくても、答えが主要なキーワードを含んでいれば正解と見なす方が親切だが、
                        # 現状は厳密一致(スペース・大文字小文字無視)
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
                     if st.button("次の問題へ", key=f"ai_next_{key_suffix}"):
                         st.session_state.ai_idx += 1
                         st.rerun()

            else:
                st.success(f"全てのAI生成問題が終了しました！({st.session_state.pdf_name})の学習完了です。お疲れ様でした。")
                if st.button("もう一度最初から解く"):
                    st.session_state.ai_idx = 0
                    st.rerun()
                if st.button("新しいフォルダを読み込む"):
                    st.session_state.ai_problems = None
                    st.session_state.pdf_name = "未設定"
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
                
                st.bar_chart(stats, y='正答率(%)', color='#4CAF50')

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
                    
                if 'coaching_message' not in st.session_state or st.session_state.coaching_message is None:
                    st.session_state.coaching_message = get_ai_coaching_message(df)

                st.subheader("AIコーチからのアドバイス")
                st.info(st.session_state.coaching_message)


if __name__ == "__main__":
    main()
