import os
import google.generativeai as genai
from dotenv import load_dotenv

# 1. Mở két sắt lấy chìa khóa API của anh Long
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ LỖI: Không tìm thấy chìa khóa. Hãy kiểm tra lại file .env!")
    exit()

genai.configure(api_key=api_key)

print("\n🔍 ĐANG DÒ TÌM DANH SÁCH BỘ NÃO GOOGLE ĐƯỢC CẤP PHÉP RIÊNG CHO TÀI KHOẢN CỦA ANH...")
print("=" * 80)

try:
    valid_models = []
    # 2. Quét thẳng vào máy chủ Google
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            # Lọc bỏ ký tự thừa để in ra tên gốc
            clean_name = m.name.replace('models/', '')
            valid_models.append(clean_name)
            print(f"👉 Tên phiên bản chính xác: {clean_name}")
            
    print("=" * 80)
    if valid_models:
        print("✅ HOÀN TẤT! Anh hãy bôi đen và COPY MỘT TÊN trong danh sách trên (Ưu tiên tên có chữ 'flash' hoặc 'pro').")
except Exception as e:
    print(f"❌ LỖI KẾT NỐI API: {e}")