from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model="gpt-4o-mini",
    max_tokens=500
)

# Mẫu 1: Yêu cầu AI kể sự thật về động vật
animal_facts_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You like telling facts and you tell facts about {animal}."),
        ("human", "Tell me {count} facts."),
    ]
)

# Mẫu 2: Yêu cầu AI dịch văn bản sang ngôn ngữ được chỉ định
translation_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a translator and convert the provided text into {language}."),
        ("human", "Translate the following text to {language}: {text}"),
    ]
)

# Chuẩn bị dữ liệu cho công đoạn dịch
prepare_for_translation = RunnableLambda(lambda output: {"text": output, "language": "vietnamese"})

# Tạo chuỗi kết hợp giữa các mẫu trên
chain = animal_facts_template | model | StrOutputParser() | prepare_for_translation | translation_template | model | StrOutputParser()

result = chain.invoke({"animal": "cat", "count": 2})

print(result)

"""
Trong đoạn mã này chúng ta gộp 2 tác vụ nối tiếp nhau (Sequential Chain) và tự động biến đổi dữ liệu giữa các bước.
Đây là một dây chuyền tự động thực hiện 2 công việc liên tiếp:
- Công đoạn 1: Nhờ AI nghĩ ra các sự thật về một loài động vật.
- Công đoạn 2: Tự động lấy các sự thật đó đem đi dịch sang tiếng Việt.
prepare_for_translation: nhận đầu vào output (chuỗi văn bản sự thật về động vật do AI ở Công đoạn 1 tạo ra).
Nó đóng gói chuỗi đó thành một Python Dictionary: {"text": output, "language": "vietnamese"}.
Tại sao phải làm vậy? Vì translation_template ở Công đoạn 2 bắt buộc cần đúng 2 biến là {text} và {language}.
prepare_for_translation: Nhận chuỗi tiếng Anh ➔ Biến đổi thành {"text": "chuỗi tiếng Anh...", "language": "vietnamese"}.
"""