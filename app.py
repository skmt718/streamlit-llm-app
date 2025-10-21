from dotenv import load_dotenv
load_dotenv()

# app.py
# Streamlit + LangChain simple LLM web app
# Python 3.11
import os
from typing import Dict, Optional
import streamlit as st

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


class AppConfig:
    """アプリケーションの設定クラス"""
    
    TITLE = "LLM Expert Switcher (A/B)"
    ICON = "🧠"
    LAYOUT = "centered"
    
    DESCRIPTION = """
このアプリは、テキスト入力と「A/B」専門家モードの選択に基づいて、LLM に質問できるシンプルなデモです。  
左上の *A/B ラジオボタン* で専門家の種類を選択し、テキストを入力して **送信** を押すと回答が表示されます。

- **A**: 経営・事業戦略の専門家として回答  
- **B**: データ分析・機械学習の専門家として回答


    PLACEHOLDER_TEXT = (
        "例：新規D2Cの立ち上げで、初月の集客戦略を考えてください（A）／\n"
        "社内FAQ検索の精度を改善したい。どの指標を追えばいい？（B）"
    )

    # LLMの設定
    LLM_MODEL = "gpt-4o-mini"
    LLM_TEMPERATURE = 0.3
    LLM_MAX_RETRIES = 2
    
    TEXT_AREA_HEIGHT = 160


class ExpertPrompts:
    """専門家プロンプトの管理クラス"""
    
    PROMPTS: Dict[str, str] = {
        "A": (
            "あなたは一流の経営コンサルタントです。"
            "ユーザーの課題に対し、目的→洞察→施策→リスク→次の一手の順で、"
            "簡潔かつ実務的に提案してください。専門用語は初出で短く定義し、"
            "箇条書きを効果的に使い、無根拠な断定を避けます。"
        ),
        "B": (
            "あなたは一流のデータサイエンティストです。"
            "ユーザーの課題に対し、問題定義→データ前提→手法候補→評価指標→実装の留意点→代替案の順で説明します。"
            "式やパラメータ名は必要最小限とし、初学者にも伝わるよう比喻を織り交ぜます。"
        ),
    }
    
    @classmethod
    def get_prompt(cls, expert_choice: str) -> str:
        """専門家の選択に応じたプロンプトを取得"""
        choice = expert_choice.strip().upper()
        return cls.PROMPTS.get(choice, cls.PROMPTS["A"])
    
    @classmethod
    def get_available_experts(cls) -> list:
        """利用可能な専門家の一覧を取得"""
        return list(cls.PROMPTS.keys())


class LLMService:
    """LLMサービスを管理するクラス"""
    
    def __init__(self):
        self.api_key = self._get_api_key()
    
    def _get_api_key(self) -> Optional[str]:
        """OpenAI APIキーを取得"""
        return st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY"))
    
    def _create_prompt(self, system_message: str) -> ChatPromptTemplate:
        """プロンプトテンプレートを作成"""
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            (
                "human",
                "次の内容に専門家として回答してください。\n"
                "【質問】{question}\n"
                "制約: 300〜600文字で、根拠と注意点を1つずつ含める。",
            ),
        ])
    
    def _create_model(self) -> ChatOpenAI:
        """LLMモデルを作成"""
        return ChatOpenAI(
            model=AppConfig.LLM_MODEL,
            temperature=AppConfig.LLM_TEMPERATURE,
            openai_api_key=self.api_key,
            max_retries=AppConfig.LLM_MAX_RETRIES,
        )
    
    def ask_llm(self, user_text: str, expert_choice: str) -> str:
        """
        入力テキストとラジオボタンの選択値（A/B）を受け取り、
        LangChain を用いて LLM からの回答テキストを返す。
        """
        # 入力検証
        if not user_text or not user_text.strip():
            return "入力テキストが空です。質問を入力してください。"
        
        # APIキーの検証
        if not self.api_key:
            return "OpenAI APIキーが設定されていません。secrets または環境変数に OPENAI_API_KEY を設定してください。"
        
        try:
            # プロンプトの準備
            system_msg = ExpertPrompts.get_prompt(expert_choice)
            prompt = self._create_prompt(system_msg)
            
            # モデルとチェーンの作成
            model = self._create_model()
            chain = prompt | model | StrOutputParser()
            
            # 実行
            result = chain.invoke({"question": user_text.strip()})
            return result
            
        except Exception as e:
            return f"エラーが発生しました: {str(e)}"


class UIComponents:
    """Streamlit UIコンポーネントを管理するクラス"""
    
    @staticmethod
    def setup_page_config() -> None:
        """ページの設定を行う"""
        st.set_page_config(
            page_title=AppConfig.TITLE,
            page_icon=AppConfig.ICON,
            layout=AppConfig.LAYOUT
        )
    
    @staticmethod
    def render_header() -> None:
        """ヘッダー部分をレンダリング"""
        st.title(AppConfig.TITLE)
        st.caption("Python 3.11 / Streamlit + LangChain デモ")
        
        with st.expander("アプリの概要・使い方", expanded=True):
            st.markdown(AppConfig.DESCRIPTION)
    
    @staticmethod
    def render_sidebar() -> str:
        """サイドバーをレンダリングして専門家の選択を取得"""
        st.sidebar.header("専門家モード")
        available_experts = ExpertPrompts.get_available_experts()
        return st.sidebar.radio(
            "専門家を選択",
            available_experts,
            index=0,
            horizontal=True
        )
    
    @staticmethod
    def render_input_form() -> tuple[str, bool]:
        """入力フォームをレンダリングして入力内容と送信状態を取得"""
        with st.form("query_form", clear_on_submit=False):
            user_text = st.text_area(
                "質問や依頼内容を入力してください",
                placeholder=AppConfig.PLACEHOLDER_TEXT,
                height=AppConfig.TEXT_AREA_HEIGHT,
            )
            submitted = st.form_submit_button("送信")
        return user_text, submitted
    
    @staticmethod
    def render_answer(answer: str) -> None:
        """回答を表示"""
        st.markdown("### 回答")
        st.write(answer)
    
    @staticmethod
    def render_footer() -> None:
        """フッターをレンダリング"""
        st.markdown("---")
        st.caption(AppConfig.FOOTER_TEXT)


def main() -> None:
    """メイン関数"""
    # ページ設定
    UIComponents.setup_page_config()
    
    # LLMサービスの初期化
    llm_service = LLMService()
    
    # UI要素のレンダリング
    UIComponents.render_header()
    expert_choice = UIComponents.render_sidebar()
    user_text, submitted = UIComponents.render_input_form()
    
    # 実行と結果表示
    if submitted:
        with st.spinner("LLMに問い合わせ中..."):
            answer = llm_service.ask_llm(user_text, expert_choice)
        UIComponents.render_answer(answer)
    
    # フッター
    UIComponents.render_footer()


# ------------------------------
# 後方互換性のための関数（Required by spec）
# ------------------------------
def ask_llm(user_text: str, expert_choice: str) -> str:
    """
    後方互換性のためのラッパー関数
    入力テキストとラジオボタンの選択値（A/B）を受け取り、
    LangChain を用いて LLM からの回答テキストを返す。
    """
    llm_service = LLMService()
    return llm_service.ask_llm(user_text, expert_choice)


if __name__ == "__main__":
    main()
