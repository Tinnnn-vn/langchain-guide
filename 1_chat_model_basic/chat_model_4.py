from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5
)

messages = [
    SystemMessage("Bạn là một chuyên gia AI."),
    HumanMessage("Gradient Descent là gì?")
]

# Gửi lần 1
response1 = model.invoke(messages)
print("AI:", response1.content)

# Để AI nhớ: Thêm câu trả lời cũ của AI (ở đây response1 chính là một AIMessage)
messages.append(response1) 

# Thêm câu hỏi tiếp theo của người dùng
messages.append(HumanMessage("Tôi vừa chat với bạn những gì bạn có nhớ không?"))

# Gửi lần 2 (lúc này 'messages' đã chứa toàn bộ lịch sử)
response2 = model.invoke(messages)
print("\nAI:", response2.content)

"""
Trong hướng dẫn này chúng ta đã thêm lịch sử trò chuyện cho AI bằng cách sử dụng AIMessage, nó đại diện
cho câu trả lời của AI trước đó.
Muốn AI nhớ: Bạn phải duy trì một danh sách Python (messages = [...]) và dùng lệnh .append() để liên tục
cập nhật cả HumanMessage lẫn AIMessage vào danh sách đó trước khi gọi llm.invoke(messages)
Trong hướng dẫn tiếp theo chúng ta sẽ nhận đầu vào từ người dùng và lưu lịch sử chat.
"""