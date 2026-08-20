from dotenv import load_dotenv
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_openai import ChatOpenAI

load_dotenv()

# Cấu hình biến định danh Firebase
PROJECT_ID = "du-an1-455313"
SESSION_ID = "user_session_new"
COLLECTION_NAME = "chat_history"

# Khởi tạo kết nối và quản lý bộ nhớ đám mây
print("Đang khởi tạo Client Firestore...")
client = firestore.Client(project=PROJECT_ID)

# Khởi tạo lịch sử tin nhắn trò chuyện trên Firestore
print("Đang khởi tạo lịch sử tin nhắn trò chuyện Firestore...")
chat_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID,
    collection=COLLECTION_NAME,
    client=client,
)

# chat_history.messages là thuộc tính chứa danh sách các tin nhắn ([HumanMessage, AIMessage...]) đã lấy từ cơ sở dữ liệu về.
# Dòng này giúp bạn kiểm tra xem hiện tại đã có lịch sử cũ nào được nạp lên chưa.
print("Lịch sử trò chuyện đã được khởi tạo.")
print("Lịch sử trò chuyện hiện tại:", chat_history.messages)

# Khởi tạo model
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7
)

print("\nBắt đầu trò chuyện với AI. Gõ 'exit hoặc quit' để thoát.")

while True:
    user_input = input("\nUser: ")
    if user_input.strip().lower() in ["quit", "exit"]:
        break

    # Đẩy tin nhắn lên cơ sở dữ liệu Firestore trên đám mây
    chat_history.add_user_message(user_input)

    # Lấy toàn bộ lịch sử trong quá khứ + tin nhắn mới truyền cho LLM để sinh ra câu trả lời
    ai_response = model.invoke(chat_history.messages)

    # Lưu câu trả lời của AI
    chat_history.add_ai_message(ai_response.content)

    print(f"\nAI: {ai_response.content}")

"""
Trong hướng dẫn này chúng ta sử dụng cơ sở dữ liệu đám mây (Google Firestore) để lưu lại toàn bộ lịch sử chat của user.
Ngay khi bạn tắt chương trình, toàn bộ lịch sử chat vẫn được lưu lại trên cơ sở dữ liệu.
Khi bạn bật lại, chương trình sẽ tự động đọc lại toàn bộ lịch sử cũ, chat tiếp như chưa từng dừng.

Tuy nhiên có một vấn đề đó là khi triển khai ứng dụng thực tế các cuộc gọi API bị gián đoạn là điều khó tránh khỏi.
Trong hướng dẫn tiếp theo chúng ta sẽ tích hợp thêm tính năng Xử lý lỗi (Error Handling) và Tự động thử lại (Retry Mechanism),
giúp chương trình không bị gián đoạn đột ngột (crash) khi gặp sự cố, đồng thời tự khôi phục trạng thái hoạt động ngay khi
dịch vụ ổn định trở lại.
"""