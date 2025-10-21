import os
import streamlit as st

# LangChainのインポート（0.3.x対応）
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

APP_TITLE = "LLM Expert Switcher (A/B)"
APP_DESC = """
このアプリは、テキスト入力と「A/B」専門家モードの選択に基づいて、LLM に質問できるシンプルなデモです。  
左上の *A/B ラジオボタン* で専門家の種類を選択し、テキストを入力して **送信** を押すと回答が表示されます。
"""

EXPERT_SYSTEM_PROMPTS = {
    "A": "あなたは経営戦略の専門家です。実務的で要点を押さえた提案を行ってください。",
    "B": "あなたはデータ分析の専門家です。問題定義から評価方法まで論理的に説明してください。",
}

def ask_llm(user_text: str, expert_choice: str) -> str:
    if not user_text:
        return "入力が空です。質問を入力してください。"

    choice = expert_choice.strip().upper()
    system_msg = EXPERT_SYSTEM_PROMPTS.get(choice, EXPERT_SYSTEM_PROMPTS["A"])

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", "次の内容に回答してください:\n{question}")
    ])

    api_key = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY"))
    if not api_key:
        return "APIキーが設定されていません。"

    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        openai_api_key=api_key
    )

    chain = prompt | model | StrOutputParser()

    try:
        return chain.invoke({"question": user_text})
    except Exception as e:
        return f"エラー: {e}"

st.set_page_config(page_title=APP_TITLE, page_icon="🧠")
st.title(APP_TITLE)
st.write(APP_DESC)

st.sidebar.header("専門家モード")
expert_display = st.sidebar.radio("選択してください", ["経営戦略", "データ分析"], horizontal=True)

# 表示名をA/Bに変換
expert_choice = "A" if expert_display == "経営戦略" else "B"

user_text = st.text_area("質問を入力", height=150)
if st.button("送信"):
    with st.spinner("処理中..."):
        answer = ask_llm(user_text, expert_choice)
    st.subheader("回答")
    st.write(answer)
