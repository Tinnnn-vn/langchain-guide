from dotenv import load_dotenv
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_openai import ChatOpenAI

load_dotenv()

PROJECT_ID = "du-an1-455313"
SESSION_ID = "user_session_new"
COLLECTION_NAME = "chat_history"

print("Đang khởi tạo Client Firestore...")
client = firestore.Client(project=PROJECT_ID)

print("Đang khởi tạo lịch sử tin nhắn trò chuyện Firestore...")
chat_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID,
    collection=COLLECTION_NAME,
    client=client,
)

print("Lịch sử trò chuyện đã được khởi tạo.")
print("Lịch sử trò chuyện hiện tại:", chat_history.messages)

base_model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7
)

# 2. Tạo mô hình có tính năng TỰ ĐỘNG THỬ LẠI (Retry Mechanism)
# Nếu gặp lỗi Rate Limit hoặc lỗi mạng chập chờn, LangChain sẽ tự động thử lại tối đa 3 lần
model = base_model.with_retry(
    stop_after_attempt=3,        # Thử lại tối đa 3 lần
    wait_exponential_jitter=True # Thời gian trễ ngẫu nhiên tăng dần giữa các lần thử
)

print("\nBắt đầu trò chuyện với AI. Gõ 'exit' hoặc 'quit' để thoát.")

while True:
    user_input = input("\nUser: ")
    
    if not user_input.strip():
        continue
        
    if user_input.strip().lower() in ["quit", "exit"]:
        break

    # 3. Sử dụng khối try - except để bắt các lỗi phát sinh
    try:
        # Đẩy tin nhắn của người dùng lên cơ sở dữ liệu Firestore
        chat_history.add_user_message(user_input)
        # Gửi toàn bộ lịch sử cho AI thông qua mô hình đã cài tính năng Retry
        ai_response = model.invoke(chat_history.messages)
        # Lưu câu trả lời của AI vào Firestore
        chat_history.add_ai_message(ai_response.content)

        print(f"AI: {ai_response.content}")

    except Exception as error:
        # Trong trường hợp thử lại 3 lần vẫn thất bại (hoặc mất mạng hoàn toàn)
        print(f"\n⚠️ Có lỗi xảy ra khi kết nối với AI hoặc Firestore: {error}")
        print("Vui lòng kiểm tra lại kết nối mạng hoặc thử lại sau ít phút.")

"""
Trong hướng dẫn này chúng ta đã thêm tính năng xử lý lỗi (Error Handling) và tự động thử lại (Retry Mechanism).
Sử dụng phương thức .with_retry() để đơn giản và gọn nhất.
"""