from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model="gpt-4o-mini",
)

# Mẫu 1: Dùng khi phản hồi Tích cực -> Tạo thư cảm ơn
positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Bạn là một trợ lý hữu ích."),
        ("human",
         "Hãy soạn một lời cảm ơn cho phản hồi tích cực này: {feedback}."),
    ]
)

# Mẫu 2: Dùng khi phản hồi Tiêu cực -> Tạo thư xin lỗi / giải quyết khiếu nại
negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Bạn là một trợ lý hữu ích."),
        ("human",
         "Hãy soạn thảo phản hồi để giải quyết ý kiến tiêu cực này: {feedback}."),
    ]
)

# Mẫu 3: Dùng khi phản hồi Trung tính -> Hỏi thêm thông tin chi tiết
neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Bạn là một trợ lý hữu ích."),
        (
            "human",
            "Soạn yêu cầu cung cấp thêm thông tin chi tiết cho phản hồi trung lập này: {feedback}.",
        ),
    ]
)

# Mẫu 4: Dùng khi tình huống phức tạp -> Tạo tin nhắn báo chuyển cho nhân viên (Human Agent)
escalate_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Bạn là một trợ lý hữu ích."),
        (
            "human",
            "Tạo tin nhắn để chuyển tiếp phản hồi này đến nhân viên hỗ trợ: {feedback}.",
        ),
    ]
)

# Định nghĩa Prompt Phân loại Cảm xúc. Prompt này yêu cầu AI đọc đoạn {feedback} và chỉ trả ra đúng 1 trong 4 từ:
# positive, negative, neutral, hoặc escalate.
classification_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Bạn là một trợ lý hữu ích."),
        ("human",
         "Hãy phân loại sắc thái của phản hồi này là tích cực, tiêu cực, trung lập hoặc cần chuyển lên cấp cao hơn: {feedback}."),
    ]
)

# Xác định các nhánh điều kiện `RunnableBranch` có thể thực thi để xử lý phản hồi.
# RunnableBranch hoạt động theo cấu trúc: RunnableBranch((điều_kiện_1, nhánh_1), (điều_kiện_2, nhánh_2), ..., nhánh_mặc_định_else)
branches = RunnableBranch(
    # NHÁNH 1: Nếu trong câu trả lời phân loại có chứa chữ "positive"
    (
        lambda x: "positive" in x,
        positive_feedback_template | model | StrOutputParser()
    ),
    # NHÁNH 2: Nếu trong câu trả lời phân loại có chứa chữ "negative"
    (
        lambda x: "negative" in x,
        negative_feedback_template | model | StrOutputParser()
    ),
    # NHÁNH 3: Nếu trong câu trả lời phân loại có chứa chữ "neutral"
    (
        lambda x: "neutral" in x,
        neutral_feedback_template | model | StrOutputParser()
    ),
    # NHÁNH MẶC ĐỊNH (ELSE): Nếu không rơi vào 3 trường hợp trên (ví dụ: "escalate")
    escalate_feedback_template | model | StrOutputParser()
)

# Ghép nối thành Chuỗi hoàn chỉnh (Combined Chain):
# 1: Tạo chuỗi phân loại cảm xúc
classification_chain = classification_template | model | StrOutputParser()

# 2. Nối chuỗi phân loại vào các nhánh điều kiện
chain = classification_chain | branches

customer_review = "Sản phẩm này chưa đủ tốt, tôi chưa thích cho lắm."
result = chain.invoke({"feedback": customer_review})
print(result)

"""
Đoạn mã này giới thiệu một khái niệm cực kỳ quan trọng trong LangChain:
Phân nhánh điều kiện (Conditional Routing/Branching) sử dụng `RunnableBranch`.

Nếu `RunnableParallel` ở bài trước giống như chia nhánh ra cho nhiều người cùng làm một lúc, thì `RunnableBranch` 
lại giống như câu lệnh if - elif - else: Nó sẽ kiểm tra điều kiện và chỉ chọn duy nhất 1 nhánh phù hợp nhất để chạy.

Trong hướng dẫn này chúng ta sẽ xây dựng một hệ thống xử lý Feedback tự động:
Bước 1: Đọc đánh giá của khách hàng và phân loại cảm xúc thành 1 trong 4 loại:
positive (tích cực), negative (tiêu cực), neutral (trung tính), hoặc escalate (cần chuyển người thật xử lý).
Bước 2 (Phân nhánh): Dựa vào kết quả phân loại ở Bước 1, chọn đúng mẫu câu phản hồi tương ứng để gửi cho khách hàng.
"""