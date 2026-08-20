from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model="gpt-4o-mini",
    max_tokens=1000
)

# Định nghĩa prompt templates cho phần tóm tắt phim
summary_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Bạn là một nhà phê bình phim."),
        ("human", "Hãy tóm tắt ngắn gọn về bộ phim {movie_name}."),
    ]
)

# Nhận bản tóm tắt ➔ Tạo prompt hỏi về Ưu/Nhược điểm của cốt truyện.
def analyze_plot(plot):
    plot_template = ChatPromptTemplate.from_messages(
        [
            ("system", "Bạn là một nhà phê bình phim."),
            ("human", "Phân tích cốt truyện: {plot}. Những điểm mạnh và điểm yếu của nó là gì?"),
        ]
    )
    return plot_template.format_prompt(plot=plot)

# Nhận bản tóm tắt ➔ Tạo prompt hỏi về Ưu/Nhược điểm của Nhân vật.
def analyze_characters(characters):
    character_template = ChatPromptTemplate.from_messages(
        [
            ("system", "Bạn là một nhà phê bình phim."),
            ("human", "Phân tích nhân vật: {characters}. Điểm mạnh và điểm yếu của họ là gì?"),
        ]
    )
    return character_template.format_prompt(characters=characters)

# Hàm nối 2 đoạn văn phân tích trên thành 1 văn bản duy nhất để đưa ra kết luận cuối cùng.
def combine_verdicts(plot_analysis, character_analysis):
    return f"Phân tích cốt truyện:\n{plot_analysis}\n\nPhân tích nhân vật:\n{character_analysis}"

# Định nghĩa 2 nhánh xử lý (plot_branch_chain & character_branch_chain) với LCEL
# Mỗi nhánh là một chuỗi xử lý độc lập: Nhận bản tóm tắt ➔ Tạo Prompt ➔ Gọi Model ➔ Bóc tách lấy văn bản dạng chuỗi.
plot_branch_chain = (
    RunnableLambda(lambda x: analyze_plot(x)) | model | StrOutputParser()
)

character_branch_chain = (
    RunnableLambda(lambda x: analyze_characters(x)) | model | StrOutputParser()
)

# Tạo chuỗi kết hợp sử dụng LangChain Expression Language (LCEL)
chain = (
    summary_template
    | model
    | StrOutputParser()
    | RunnableParallel(branches={"plot": plot_branch_chain, "characters": character_branch_chain})
    | RunnableLambda(lambda x: combine_verdicts(x["branches"]["plot"], x["branches"]["characters"]))
)

result = chain.invoke({"movie_name": "Interstellar"})

print(result)

"""
Đoạn mã này nâng cấp từ chuỗi nối tiếp (Sequential) sang xử lý song song (Parallel Processing).
Giả sử bạn đưa bộ phim Interstellar cho 2 nhà phê bình: một người chuyên phân tích Cốt truyện (Plot) và một người
chuyên phân tích Nhân vật (Characters). Thay vì bắt từng người làm lần lượt, bạn giao việc cho cả 2 người làm
cùng một lúc để tiết kiệm thời gian. Đó chính là vai trò của `RunnableParallel`.

summary_template | model | StrOutputParser(): Gửi tên phim (ví dụ: "Inception") ➔ AI tạo ra một đoạn văn bản tóm tắt.
RunnableParallel(branches={...}): Nhận đoạn văn bản tóm tắt ở trên rồi chia đôi dữ liệu và gửi cùng lúc (song song)
vào plot_branch_chain và character_branch_chain.

Kết quả đầu ra của bước này: Là một Python Dictionary chứa đáp án của cả 2 nhánh:
{
    "branches": {
        "plot": "Phân tích cốt truyện...",
        "characters": "Phân tích nhân vật..."
    }
}

RunnableLambda(lambda x: combine_verdicts(...)): Lấy dữ liệu từ Dictionary trên truyền vào hàm combine_verdicts
để gộp 2 bài phân tích thành 1 bài hoàn chỉnh.

Tóm lại RunnableParallel giúp tiết kiệm thời gian nếu bạn gọi nối tiếp theo chuỗi, nhờ xử lý song song nên các
nhánh chạy cùng lúc.
"""