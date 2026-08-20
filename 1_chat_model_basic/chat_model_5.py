from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

# Khởi tạo model
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5
)

# Tạo một danh sách rỗng để lưu lịch sử trò chuyện
chat_history = []

# Thiết lập hệ thống prompt cho model rồi lưu vào lịch sử chat.
chat_history.append(
    SystemMessage(content="Bạn là một trợ lý AI thân thiện và thông minh.")
)

print("Bắt đầu trò chuyện với AI. Gõ 'exit hoặc quit' để thoát.")

# Vòng lặp chat
while True:
    user_input = input("You: ")
    if user_input.strip().lower() in ["quit", "exit"]:
        break

    # Đóng gói câu hỏi của người dùng vào nhãn HumanMessage rồi thêm câu hỏi của người dùng vào lịch sử chat.
    chat_history.append(HumanMessage(content=user_input))

    # Đưa lich sử chat cho AI đọc và trả lời.
    result = model.invoke(chat_history)

    # Ghi câu trả lời của AI vào lại lịch sử chat.
    chat_history.append(result)

    # In câu trả lời ra màn hình
    print(f"AI: {result.content}")

# In lịch sử chat khi người dùng gõ 'quit hoặc exit'
print("\n---- Lịch sử trò chuyện ----")
for msg in chat_history:
    if isinstance(msg, SystemMessage):
        print(f"[System]: {msg.content}")
    elif isinstance(msg, HumanMessage):
        print(f"[You]: {msg.content}")
    elif isinstance(msg, AIMessage):
        print(f"[AI]: {msg.content}")
    print("-" * 30)

"""
Trong hướng dẫn này chúng ta nhận đầu vào trực tiếp từ người dùng và mới chỉ lưu lịch sử trò chuyện vào bộ nhớ RAM của máy tính.
Tuy nhiên khi tắt chương trình, dữ liệu lịch sử chat sẽ bị mất.
Trong hướng dẫn tiếp chúng ta sẽ lưu trữ dữ liệu sang đám mây với Google Firestore.
"""