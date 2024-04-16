import pandas as pd  # 파이썬 언어로 작성된 데이터를 분석 및 조작하기 위한 라이브러리
from dotenv import load_dotenv
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

load_dotenv()
# csv 파일을 데이터프레임으로 가져오기
df = pd.read_csv('booksv_02.csv')  # booksv_02.csv 파일이 위치한 경로 지정
df.head()
print(df.head())

# 에이전트 생성
agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, model='gpt-3.5-turbo'),  # gpt-3.5-turbo 사용
    df,  # 데이터가 담긴 곳
    verbose=False,  # 추론 과정을 출력하지 않음
    agent_type=AgentType.OPENAI_FUNCTIONS,
)

result1 = agent.invoke('어떤 제품의 ratings_count가 제일 높아?')  # ratings_count가 가장 높은 제품을 찾기 위한 질의
print(result1)
print('result1: ' + result1.get('output'))  # 결과 출력
result2 = agent.invoke('가장 최근에 출간된 책은?')  # 가장 최근에 출간된 책을 찾기 위한 질의
print('result2: ' + result2.get('output'))  # 결과 출력
