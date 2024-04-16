import streamlit as st
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI

st.set_page_config(page_title="🦜🔗 뭐든지 질문하세요~ ")
st.title('🦜🔗 뭐든지 질문하세요~ ')

load_dotenv()


# os.environ["OPENAI_API_KEY"] = "sk-"  # openai 키 입력

def generate_response(input_text):  # llm이 답변 생성
    # # 창의성 0으로 설정, # 모델명
    llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')
    st.info(llm.predict(input_text))


with st.form('Question'):
    text = st.text_area('질문 입력:', 'What types of text models does OpenAI provide?')  # 첫 페이지가 실행될 때 보여줄 질문
    submitted = st.form_submit_button('보내기')
    generate_response(text)

# streamlit run 5_1_chatbot.py
