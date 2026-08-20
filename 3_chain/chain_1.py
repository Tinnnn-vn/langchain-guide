from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model="gpt-4o-mini",
    max_tokens=250
)

# Định nghĩa prompt templates (không cần các chuỗi Runnable riêng biệt)
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Bạn là một chuyên gia am hiểu về {animal}."),
        ("human", "Hãy kể cho tôi {fact_count} sự thật.")
    ]
)

# Tạo chuỗi kết hợp sử dụng LangChain Expression Language (LCEL)
chain = prompt_template | model | StrOutputParser()

# Chạy chuỗi
result = chain.invoke({"animal": "con rắn", "fact_count": 1})

print(result)

"""
Trong hướng dẫn này chúng ta sẽ sử dụng toán tử: Pipe |
Có nghĩa là lấy đầu ra của bên trái truyền làm đầu vào cho bên phải.
Giả sử bạn đang làm việc ở một dây chuyền sản xuất. Đầu vào đi qua Công đoạn A, kết quả của A tự động chuyển sang Công đoạn B,
rồi xuất ra Sản phẩm hoàn chỉnh.
Trong LangChain, chain chính là dây chuyền đó, nó giúp bạn nối các thành phần (Prompt + Model + Output Parser) lại với nhau
thành một luồng xử lý tự động từ đầu đến cuối. Giúp mã gọn gàng hơn và dễ mở rộng hơn.
Thay vì phải tự tay chuyển dữ liệu qua lại giữa các biến, chain giúp bạn gắn các linh kiện vào nhau thành một cỗ máy tự động.
Bạn chỉ cần bấm nút chain.invoke() là xong.

Trong hướng dẫn tiếp theo, chúng ta sẽ cùng xem cách hoạt động bên dưới băng chuyền này của langchain nhé.
"""