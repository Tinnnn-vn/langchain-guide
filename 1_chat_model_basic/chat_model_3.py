from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

messages = [
    SystemMessage(content="Bạn là một gia sư Toán."),
    HumanMessage(content="Tại sao số âm nhân với số âm lại thành số dương?"),
]

# Tạo model openai
model = ChatOpenAI(model="gpt-4o-mini")
result = model.invoke(messages)
print(f"OpenAI: {result.content}")

# Tạo model Anthropic
model = ChatAnthropic(model="claude-3-opus-20240229")
result = model.invoke(messages)
print(f"Anthropic: {result.content}")

# Tạo model Google
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
result = model.invoke(messages)
print(f"Answer from Google: {result.content}")

"""
Trong hướng dẫn này chúng ta gọi nhiều model khác nhau để tạo ra các câu trả lời riêng của từng model.
Tuy nhiên AI vẫn chưa nhớ lịch sử trò chuyện trước đó, trong phần tiếp theo chúng ta sẽ xây dựng thêm
lịch sử trò chuyện (AIMessage).
"""