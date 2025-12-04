import os
import sys
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
# Lesson7の解説に合わせて、LangChain Coreからメッセージクラスをインポートします。
from langchain_core.messages import SystemMessage, HumanMessage 

# Windows環境でのエンコードエラー ('ascii' codec can't encode) 対策
# Pythonの標準I/OエンコーディングをUTF-8に設定します。
if sys.platform.startswith('win'):
    os.environ["PYTHONIOENCODING"] = "utf-8"

# 1. 環境変数をロードする
# .envファイルからAPIキーを読み込みます
load_dotenv()

# 2. Streamlitアプリのタイトルとレイアウトを設定
st.set_page_config(page_title="専門家LLMアプリ", layout="centered")
st.title("👨‍💼 専門家パーソナリティ選択型LLMアプリ")

# 3. 専門家定義とシステムメッセージの設定
# 専門家の種類と、それに対応するシステムメッセージを定義します
# システムメッセージに「回答は日本語で行うこと」という指示を追記します。
EXPERT_PROFILES = {
    "テック系ジャーナリスト": "あなたは最新のITトレンド、ガジェット、科学技術に非常に詳しいテック系ジャーナリストです。回答は最新の情報に基づいて、エキサイティングで簡潔な口調で提供してください。回答は必ず日本語で行ってください。",
    "歴史学者": "あなたは古代から現代までの歴史、文化、年号に精通した厳格な歴史学者です。回答は客観的な事実に基づき、情報の出典（例：紀元前100年頃、〇〇史によると）を明確にしながら、正確で丁寧な口調で提供してください。回答は必ず日本語で行ってください。",
    "ビジネスコンサルタント": "あなたは市場戦略、組織改革、効率化の専門知識を持つビジネスコンサルタントです。回答は構造化され（箇条書きなどを活用）、実用的で、ビジネス課題の解決に役立つ具体的なアクションプランを中心に提供してください。回答は必ず日本語で行ってください。",
}

# 4. LLMモデルの初期化
try:
    # APIキーのチェックを兼ねる
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
except Exception as e:
    st.error(f"OpenAIモデルの初期化に失敗しました。APIキーを確認してください: {e}")
    st.stop()

# ----------------------------------------------------
# ① 関数を定義し、利用する
# ----------------------------------------------------
# 「入力テキスト」と「ラジオボタンでの選択値」を引数として受け取り、
# LLMからの回答を戻り値として返す関数
@st.cache_data
def get_llm_response(user_question: str, expert_profile_key: str) -> str:
    """
    ユーザーの質問と専門家の選択に基づいてLLMから回答を取得する関数。
    
    Args:
        user_question (str): ユーザーからの入力テキスト。
        expert_profile_key (str): 選択された専門家のキー (例: "歴史学者")。
        
    Returns:
        str: LLMからの回答テキスト。
    """
    # 選択された専門家に対応するシステムメッセージを取得
    system_message = EXPERT_PROFILES[expert_profile_key]
    
    # ----------------------------------------------------
    # 【エンコーディング対策の簡素化】
    # 複雑なエスケープを削除し、純粋な文字列を渡します。
    # 必要なエンコーディング回避は、起動コマンドと環境変数に任せます。
    # ----------------------------------------------------

    # Lesson7の解説にある「llm(messages)」形式を、より安定した「llm.invoke(messages)」形式に修正
    messages = [
        # SystemMessageとHumanMessageに元の文字列をそのまま渡す
        SystemMessage(content=system_message),
        HumanMessage(content=user_question),
    ]
    
    # ChatOpenAIのインスタンスのinvokeメソッドにメッセージリストを渡して実行
    result = llm.invoke(messages)
    
    # result.contentで回答結果を取り出す
    return result.content


# ----------------------------------------------------
# ② Webアプリの概要や操作方法をユーザーに明示するためのテキストを表示
# ----------------------------------------------------
st.markdown("""
### アプリケーションの概要
このWebアプリは、ユーザーが選択した**専門家（ペルソナ）**の視点に基づいて、質問に回答するLLM（大規模言語モデル）インターフェースです。質問の内容に合わせて専門家を切り替えることで、多角的な視点からの情報を得ることができます。

### 操作方法
1.  **左側のサイドバー**で、回答してほしい専門家（テック系ジャーナリスト、歴史学者、ビジネスコンサルタント）をラジオボタンで選択します。
2.  中央のテキストエリアに質問を入力します。
3.  「**回答を生成**」ボタンをクリックすると、選択した専門家のシステムメッセージがLLMに渡され、そのペルソナに基づいた回答が表示されます。
""")

# ----------------------------------------------------
# 画面の構成
# ----------------------------------------------------

# ① ラジオボタンでLLMに振る舞わせる専門家の種類を選択できるようにする
st.sidebar.header("専門家の選択")
selected_expert = st.sidebar.radio(
    "LLMに誰の振る舞いをさせますか？",
    list(EXPERT_PROFILES.keys()),
    key="expert_radio"
)
st.sidebar.info(f"選択中の専門家: **{selected_expert}**")

# 5. 入力フォームの用意と処理
user_input = st.text_area("ここに質問を入力してください：", height=150, key="input_area")

# ボタンが押されたら処理を実行
if st.button("回答を生成"):
    if user_input:
        
        # 関数を利用してLLMを実行
        with st.spinner(f"AI ({selected_expert}) が回答を考え中です..."):
            try:
                # 定義した関数を呼び出し、回答を取得
                llm_response_content = get_llm_response(user_input, selected_expert)
            except Exception as e:
                # エラー発生時に詳細な情報を表示するように修正
                st.error(f"LLMの実行中にエラーが発生しました: {type(e).__name__} - {e}")
                llm_response_content = "回答の取得に失敗しました。詳細なエラーは上記メッセージを確認してください。"

        # ③回答結果が画面上に表示されるようにする
        st.subheader(f"🤖 AIの回答 ({selected_expert})")
        st.info(llm_response_content)

    else:
        st.warning("質問内容を入力してください。")