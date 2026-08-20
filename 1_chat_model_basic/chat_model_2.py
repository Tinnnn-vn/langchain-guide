from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=1.0
)

# Tạo danh sách messages và vai trò của AI
messages = [
    SystemMessage("Bạn là một chuyên gia sáng tạo nội dung."),
    HumanMessage("Tạo một nội dung ngắn về bài đăng quảng cáo cho thương hiệu cà phê.")
]

result = model.invoke(messages)
print(result.content)

"""
Hãy tưởng tượng việc giao tiếp với AI giống như một cuộc trò chuyện  giữa người với người, có nhiều loại thông điệp khác nhau:
SystemMessage: "Thiết lập những quy tắc cơ bản" - Giống như việc nói với một người với vai trò nào đó trước khi bắt đầu cuộc trò chuyện.
HumanMessage: "Thông điệp từ con người" - Những gì bạn sẽ nói.
AIMessage: "Thông điệp từ AI" - AI sẽ phản hồi như thế nào?
"""
