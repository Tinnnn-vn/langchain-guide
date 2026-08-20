from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=1.2,
    max_tokens=100
)

messages = [
    ("system", "Bạn là một chuyên gia {topic}."),
    ("human", "Hãy kể một câu chuyện về {concept} bằng {language}.")
]

prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({
    "topic": "Kể chuyện hài hước",
    "concept": "Conan",
    "language": "Tiếng Việt"
})

result = model.invoke(prompt)
print(result.content)

"""
Trong hướng dẫn này chúng ta sẽ xem cách ChatPromptTemplate tạo ra các lời nhắc (prompt) cho AI một cách chuẩn xác, linh hoạt
và tái sử dụng được trong LangChain.
Khi xây dựng ứng dụng thực tế, bạn không thể ngồi gõ cứng (hardcode) từng câu hỏi hay vai trò của AI. Mà bạn sẽ:
- Định hình sẵn cấu trúc hội thoại (Tin nhắn Hệ thống, Tin nhắn Người dùng...)
- Thay thế các biến (như tên topic, concept, language,...) vào câu hỏi một cách tự động.
"""