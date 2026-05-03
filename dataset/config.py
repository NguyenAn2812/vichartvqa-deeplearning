import os

OUTPUT_DIR   = "./vqa_flat_dataset"
IMG_DIR      = os.path.join(OUTPUT_DIR, "images")
MD_DIR       = os.path.join(OUTPUT_DIR, "markdown")
DATASET_FILE = os.path.join(OUTPUT_DIR, "vqa_dataset.json")

for d in [IMG_DIR, MD_DIR]:
    os.makedirs(d, exist_ok=True)

API_KEY    = ""
MODEL_NAME = "openai/gpt-oss-20b:free"

PROMPT_CHART_CONFIG = """
Bạn là chuyên gia dữ liệu. Sinh ra cấu hình cho 1 biểu đồ duy nhất loại: {forced_type}.

YÊU CẦU BẮT BUỘC CHO LOẠI {forced_type}:
- Nếu là "pie": Chỉ có 1 cột dữ liệu trong y_cols. x_cats là các thành phần (tên danh mục, không phải thời gian).
- Nếu là "line" hoặc "area": x_col PHẢI là chuỗi thời gian (tháng, quý, năm). y_cols có đúng 2 cột. Dữ liệu phải có xu hướng rõ ràng (tăng dần, giảm dần, hoặc biến động có đỉnh/đáy).
- Nếu là "bar": x_col là các thực thể so sánh (quốc gia, sản phẩm, khu vực...). y_cols có đúng 2 cột. Các giá trị phải tạo ra sự chênh lệch rõ ràng giữa các thực thể.

YÊU CẦU VỀ DỮ LIỆU:
- Dữ liệu phải thực tế, có câu chuyện (không phải số ngẫu nhiên).
- Phải có ít nhất 1 xu hướng hoặc đặc điểm nổi bật để có thể đặt câu hỏi phân tích.
- min và max phải tạo ra sự biến động đủ lớn (chênh lệch ít nhất 30% so với max).

Trả về JSON Object theo cấu trúc mẫu sau (Lưu ý: Chỉ trả về JSON, không giải thích):
{{
  "theme": "Chủ đề cụ thể và thực tế",
  "title": "Tiêu đề mô tả rõ nội dung biểu đồ",
  "type": "{forced_type}",
  "y_label": "Đơn vị đo lường",
  "x_col": "Tên trục X",
  "x_cats": ["Mốc 1", "Mốc 2", "Mốc 3", "Mốc 4", "Mốc 5"],
  "y_cols": [
    {{
      "name": "Tên đại lượng 1 rõ ràng",
      "min": 10,
      "max": 100,
      "color": "#3b82f6"
    }},
    {{
      "name": "Tên đại lượng 2 rõ ràng",
      "min": 10,
      "max": 100,
      "color": "#ef4444"
    }}
  ]
}}
"""

PROMPT_GEN_QUESTIONS = """
Dựa vào bảng dữ liệu Markdown sau:
{markdown_data}

NHIỆM VỤ: Sinh ra đúng 4 câu hỏi tiếng Việt về biểu đồ, tập trung vào PHÂN TÍCH và SUY LUẬN.

ĐỊNH HƯỚNG CÂU HỎI (chọn 4 hướng khác nhau):
1. Xu hướng tổng thể: Hỏi về chiều hướng thay đổi trong một khoảng (tăng/giảm/ổn định).
   Ví dụ mẫu: "Từ [mốc A] đến [mốc B], [đại lượng] thay đổi như thế nào?"

2. So sánh tương quan giữa 2 đại lượng: Hỏi mối quan hệ giữa 2 cột dữ liệu.
   Ví dụ mẫu: "Khi [đại lượng 1] tăng, [đại lượng 2] có xu hướng ra sao?"

3. Xác định đỉnh/đáy hoặc giai đoạn nổi bật: Hỏi về thời điểm/thực thể có đặc điểm đặc biệt.
   Ví dụ mẫu: "Giai đoạn nào [đại lượng] đạt mức cao nhất?" hoặc "Mục nào dẫn đầu về [đại lượng]?"

4. So sánh hai nhóm/giai đoạn: Hỏi về sự khác biệt giữa nửa đầu và nửa cuối, hoặc giữa 2 thực thể cụ thể.
   Ví dụ mẫu: "Nửa đầu và nửa cuối [kỳ] có sự khác biệt gì về [đại lượng]?"

5. Đánh giá mức độ biến động: Hỏi về sự ổn định hay biến động của dữ liệu.
   Ví dụ mẫu: "[Đại lượng] trong giai đoạn này biến động nhiều hay ít?"

LUẬT BẮT BUỘC:
- TUYỆT ĐỐI KHÔNG hỏi giá trị tuyệt đối của một điểm dữ liệu cụ thể (không hỏi "bao nhiêu", "là mấy", "bằng bao nhiêu" tại một mốc đơn lẻ).
- Câu hỏi phải yêu cầu đọc NHIỀU điểm dữ liệu hoặc so sánh để trả lời.
- Câu hỏi phải tự nhiên, giống ngôn ngữ người dùng thực tế khi xem biểu đồ.
- Trả về DUY NHẤT một mảng JSON chứa 4 chuỗi câu hỏi.
- Không giải thích gì thêm.
"""

PROMPT_ANSWER_QUESTION = """
Bảng dữ liệu:
{markdown_data}

Câu hỏi: {question}

Hãy trả lời câu hỏi trên dựa vào bảng dữ liệu. Yêu cầu:
- Trả lời bằng NHẬN XÉT hoặc MÔ TẢ XU HƯỚNG, không đọc số liệu cụ thể.
  Ví dụ đúng: "Tăng dần đều qua các tháng", "Giảm mạnh sau đỉnh giữa kỳ", "Biến động không ổn định"
  Ví dụ sai: "Đạt 85 triệu vào tháng 3", "Tăng thêm 20 đơn vị"
- Độ dài: 5 đến 12 từ, tự nhiên như người xem biểu đồ nhận xét.
- Ưu tiên dùng các từ mô tả xu hướng: tăng dần, giảm mạnh, ổn định, biến động, đạt đỉnh, chạm đáy, vượt trội, bắt kịp, phân hóa rõ rệt...
- Trả về ĐÚNG chuỗi câu trả lời, KHÔNG bọc trong JSON, KHÔNG giải thích.
"""