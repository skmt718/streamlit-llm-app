# app.py
# Streamlit + LangChain simple LLM web app
# Python 3.11
import os
import streamlit as st

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# ------------------------------
# App metadata
# ------------------------------
APP_TITLE = "LLM Expert Switcher (A/B)"
APP_DESC = """
このアプリは、テキスト入力と「A/B」専門家モードの選択に基づいて、LLM に質問できるシンプルなデモです。  
左上の *A/B ラジオボタン* で専門家の種類を選択し、テキストを入力して **送信** を押すと回答が表示されます。

- **A**: 経営・事業戦略の専門家として回答  
- **B**: データ分析・機械学習の専門家として回答

※ 本アプリの実行には OpenAI API キーが必要です。Streamlit Community Cloud では `secrets` に `OPENAI_API_KEY` を設定してください。
"""

# ------------------------------
# Expert system prompts
# ------------------------------
EXPERT_SYSTEM_PROMPTS = {
    "A": (
        "あなたは一流の経営コンサルタントです。"
        "ユーザーの課題に対し、目的→洞察→施策→リスク→次の一手の順で、"
        "簡潔かつ実務的に提案してください。専門用語は初出で短く定義し、"
        "箇条書きを効果的に使い、無根拠な断定を避けます。"
    ),
    "B": (
        "あなたは一流のデータサイエンティストです。"
        "ユーザーの課題に対し、問題定義→データ前提→手法候補→評価指標→実装の留意点→代替案の順で説明します。"
        "式やパラメータ名は必要最小限とし、初学者にも伝わるよう比喩を織り交ぜます。"
    ),
}

# ------------------------------
# Core LLM function (Required by spec)
# ------------------------------
def ask_llm(user_text: str, expert_choice: str) -> str:
    """
    入力テキストとラジオボタンの選択値（A/B）を受け取り、
    LangChain を用いて LLM からの回答テキストを返す。
    """
    if not user_text:
        return "入力テキストが空です。質問を入力してください。"

    # 安全な選択値正規化（A/B のみを許容）
    choice = expert_choice.strip().upper()
    if choice not in EXPERT_SYSTEM_PROMPTS:
        choice = "A"

    system_msg = EXPERT_SYSTEM_PROMPTS[choice]

    # LangChain prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_msg),
            (
                "human",
                "次の内容に専門家として回答してください。\n"
                "【質問】{question}\n"
                "制約: 300〜600文字で、根拠と注意点を1つずつ含める。",
            ),
        ]
    )

    # モデルの準備（環境変数か Streamlit secrets から取得）
    # secrets が優先、なければ環境変数
    api_key = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    if not api_key:
        return "OpenAI APIキーが設定されていません。secrets または環境変数に OPENAI_API_KEY を設定してください。"

    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        openai_api_key=api_key,
        max_retries=2,
    )

    chain = prompt | model | StrOutputParser()

    # 実行
    try:
        result = chain.invoke({"question": user_text})
    except Exception as e:
        return f"エラーが発生しました: {e}"

    return result

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="centered")
st.title(APP_TITLE)
st.caption("Python 3.11 / Streamlit + LangChain デモ")

with st.expander("アプリの概要・使い方", expanded=True):
    st.markdown(APP_DESC)

# サイドバー：専門家モード選択
st.sidebar.header("専門家モード")
expert_choice = st.sidebar.radio("専門家を選択", ["A", "B"], index=0, horizontal=True)

# 入力フォーム
with st.form("query_form", clear_on_submit=False):
    user_text = st.text_area(
        "質問や依頼内容を入力してください",
        placeholder="例：新規D2Cの立ち上げで、初月の集客戦略を考えてください（A）／\n社内FAQ検索の精度を改善したい。どの指標を追えばいい？（B）",
        height=160,
    )
    submitted = st.form_submit_button("送信")

# 実行と結果表示
if submitted:
    with st.spinner("LLMに問い合わせ中..."):
        answer = ask_llm(user_text, expert_choice)
    st.markdown("### 回答")
    st.write(answer)

# フッター
st.markdown("---")
st.caption("© 2024 Hiro_1. Built with Streamlit and LangChain.")
