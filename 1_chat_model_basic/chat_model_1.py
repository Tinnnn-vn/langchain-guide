from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load biến môi trường .env
load_dotenv()

# Khởi tạo model
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.9,
    max_tokens=50
)

# Tạo nội dung chat và trả về kết quả
result = model.invoke("Hãy tạo tên cho một công ty thiết kế.")
print(result.content)

"""
temperature (0.0 - 2.0): Kiểm soát tính ngẫu nhiên và khả năng sáng tạo của model:
temperature (0.0 - 0.3): Thường dùng để giải toán, tạo mã, cho ra câu trả lời logic - chính xác.
temperature (0.4 - 0.7): Cho câu trả lời nhất quán và đa dạng, thường dùng cho các cuộc trò chuyện thông thường.
temperature (0.8 - 1.5): Cho câu trả lời ngẫu nhiên và sáng tạo hơn, thường sử dụng khi cần sáng tạo nội dung, ý tưởng, động não.
temperature (1.5 - 2.0): Cho câu trả lời ngẫu nhiên tối đa, dùng để thử nghiệm.

Bạn cũng có thể thiết lập max_tokens để giới hạn số lượng token tối đa mà model được phép tạo ra trong câu trả lời.
Mục đích của việc này dùng để tối ưu chi phí.

model.invoke(): AI sẽ xử lý xong toàn bộ mới trả về 1 kết quả duy nhất. Người dùng phải đợi vài giây màn hình "đứng yên"
một lúc mới thấy kết quả.
Tuy nhiên chúng ta cũng có thể sử dụng model.stream(), nghĩa là AI sinh ra đến đâu, trả về ngay từng chunk đến đó.
Người dùng thấy chữ xuất hiện dần dần ngay lập tức (giống ChatGPT).
Lợi ích của model.stream() giúp trải nghiệm người dùng tốt hơn (phản hồi tức thì). Người dùng có thể bắt đầu đọc trong khi AI tạo ra phần còn lại.

Bạn có thể sử dụng đoạn mã sau để tạo từng chunk trả về ngay lập tức trên màn hình thay vì phải chờ đợi AI trả về toàn bộ kết quả:

result = model.stream("Hãy tạo tên cho một công ty thiết kế.")
for chunk in result:
    print(chunk.content, end="", flush=True)  # flush=True giúp màn hình terminal in ra ngay lập tức không bị hoãn
"""