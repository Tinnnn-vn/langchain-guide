from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model="gpt-4o-mini",
    max_tokens=500
)

prompt_template = ChatPromptTemplate.from_messages(
     [
        ("system", "Bạn là một chuyên gia am hiểu về {animal}."),
        ("human", "Hãy kể cho tôi {fact_count} sự thật.")
    ]
)

# Tạo các runnable riêng lẻ (các bước trong chuỗi)
format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content)

# Tạo RunnableSequence (tương đương với chuỗi LCEL)
chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)

# Chạy chuỗi
result = chain.invoke({"animal": "Chim yến", "fact_count": 2})

print(result)

"""
Ở bài trước, chúng ta dùng cú pháp ngắn gọn với toán tử pipe: prompt | model | Parser. Đoạn mã trong bài này chúng ta cùng
mổ xẻ chi tiết 3 công đoạn đó ra bằng các lớp nguyên bản của LangChain để bạn hiểu bản chất.
- format_prompt (Công đoạn 1): Nhận đầu vào x (ví dụ: {"animal": "Chim yến"}) và điền dữ liệu vào prompt_template để tạo ra
  một đối tượng Prompt hoàn chỉnh.
- invoke_model (Công đoạn 2): Chuyển Prompt thành danh sách tin nhắn (.to_messages()), sau đó gửi cho model xử lý và nhận về
  đối tượng AIMessage.
- parse_output (Công đoạn 3): Nhận đối tượng AIMessage từ công đoạn 2 và bóc tách riêng phần nội dung chữ (.content).

RunnableSequence: chính là sợi dây chuyền nối các công đoạn trên lại với nhau theo đúng thứ tự.
first: Công đoạn đầu tiên (format_prompt).
middle: Danh sách các công đoạn ở giữa (ở đây có 1 bước là invoke_model).
last: Công đoạn cuối cùng (parse_output).
"""