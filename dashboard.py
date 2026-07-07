# -*- coding: utf-8 -*-
"""
dashboard.py
============
Giao diện Streamlit cho "Meta Agent" — Nhà máy sản xuất phần mềm tự động.

Chạy:
    streamlit run dashboard.py

Luồng vận hành:
    1. Nhập Google API Key + mô tả phần mềm → bấm "Khởi động nhà máy".
    2. Đồ thị chạy: Supervisor → Architect → Supervisor → DỪNG (interrupt_before)
       ngay trước node Developer để chờ duyệt bản vẽ.
    3. Tại Cổng kiểm duyệt:
       - ✅ Duyệt      → Developer chạy → QA exec() kiểm thử → tự chữa lành tối đa 3 vòng.
       - ♻️ Yêu cầu sửa → update_state(as_node="supervisor") ép next_agent="architect"
                          → conditional_edge của Supervisor tự lái ngược về Architect.
    4. Kết thúc: hiển thị code cuối cùng + kết quả đóng gói .exe, hoặc thông báo
       cầu dao an toàn đã ngắt sau 3 lần lỗi.

Lưu ý kỹ thuật quan trọng (self-healing không bị kẹt ở cổng duyệt):
    interrupt_before=["developer"] sẽ kích hoạt MỖI LẦN đồ thị chuẩn bị chạy
    Developer — kể cả khi QA fail và vòng ngược về Developer để sửa lỗi.
    Hàm resume_until_gate() phân biệt 2 trường hợp bằng test_result:
      - test_result == "FAILED"  → đây là vòng self-healing → TỰ ĐỘNG chạy tiếp.
      - test_result == ""        → đây là cổng duyệt bản vẽ thật → DỪNG chờ người dùng.
"""

import uuid

import streamlit as st

from meta_team_config import MAX_RETRIES, build_meta_team, make_initial_state

# ============================================================
# CẤU HÌNH TRANG
# ============================================================

st.set_page_config(
    page_title="Meta Agent Factory",
    page_icon="🏭",
    layout="wide",
)

st.title("🏭 Meta Agent — Nhà máy sản xuất phần mềm tự động")
st.caption(
    "LangGraph + Streamlit + Google Gemini  |  "
    "Supervisor (CEO) • Architect • Developer • QA • DevOps"
)

# ============================================================
# KHỞI TẠO SESSION STATE
# ============================================================

_DEFAULTS = {
    "phase": "idle",            # idle | awaiting_approval | finished
    "thread_id": str(uuid.uuid4()),
    "app": None,                # Đồ thị LangGraph đã compile
}
for _key, _value in _DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _value


def get_config() -> dict:
    """Config gắn với thread_id — MemorySaver dùng nó để nhớ state giữa các lần invoke."""
    return {"configurable": {"thread_id": st.session_state.thread_id}}


def resume_until_gate(payload) -> None:
    """
    Chạy đồ thị đến khi: (1) kết thúc, hoặc (2) dừng ở cổng duyệt bản vẽ thật sự.

    interrupt_before=["developer"] kích hoạt cả trong vòng self-healing
    (QA fail → quay lại Developer). Khi đó test_result == "FAILED" → đây KHÔNG
    phải cổng duyệt bản vẽ → tự động invoke(None) chạy tiếp, không hỏi lại
    người dùng. Chỉ khi test_result == "" (lần code đầu sau khi có bản vẽ mới)
    thì mới dừng thật để chờ duyệt.
    """
    app = st.session_state.app
    config = get_config()

    app.invoke(payload, config)

    while True:
        snapshot = app.get_state(config)

        if not snapshot.next:
            # Không còn node chờ chạy → đồ thị đã đến END
            st.session_state.phase = "finished"
            return

        if "developer" in snapshot.next and snapshot.values.get("test_result") == "FAILED":
            # Vòng self-healing: QA vừa fail, Developer cần chạy lại NGAY → tự resume
            app.invoke(None, config)
            continue

        # Cổng duyệt bản vẽ thật sự (Human-in-the-loop)
        st.session_state.phase = "awaiting_approval"
        return


# ============================================================
# SIDEBAR — CẤU HÌNH
# ============================================================

with st.sidebar:
    st.header("⚙️ Cấu hình")
    api_key = st.text_input("Google API Key (Gemini)", type="password")
    model_name = st.text_input("Model Gemini", value="gemini-2.5-flash")
    st.divider()
    st.markdown(
        f"**Cầu dao an toàn:** hệ thống tự ngắt sau **{MAX_RETRIES} lần** QA báo lỗi "
        f"để tránh cháy API."
    )
    st.divider()
    if st.button("🔄 Reset phiên làm việc", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ============================================================
# KHU VỰC NHẬP NHIỆM VỤ
# ============================================================

task = st.text_area(
    "📝 Mô tả phần mềm cần sản xuất",
    height=120,
    placeholder="Ví dụ: Viết ứng dụng máy tính bỏ túi bằng tkinter, có cộng trừ nhân chia và nút xóa...",
)

start_disabled = st.session_state.phase == "awaiting_approval"
if st.button("🚀 Khởi động nhà máy", type="primary", disabled=start_disabled):
    if not api_key:
        st.error("Vui lòng nhập Google API Key ở thanh bên trái.")
    elif not task.strip():
        st.error("Vui lòng mô tả phần mềm cần sản xuất.")
    else:
        # Mỗi lần khởi động là một thread mới → state sạch hoàn toàn
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.app = build_meta_team(api_key, model_name)
        with st.spinner("Supervisor đang điều phối, Architect đang vẽ bản kiến trúc..."):
            resume_until_gate(make_initial_state(task.strip()))
        st.rerun()

# ============================================================
# HIỂN THỊ TRẠNG THÁI NHÀ MÁY
# ============================================================

if st.session_state.phase != "idle" and st.session_state.app is not None:
    snapshot = st.session_state.app.get_state(get_config())
    values = snapshot.values

    # ---- Nhật ký các Agent ----
    with st.expander("📜 Nhật ký nhà máy (log các Agent)", expanded=True):
        for line in values.get("messages", []):
            st.markdown(f"- {line}")

    # ---- Thanh trạng thái retry ----
    retry_count = values.get("retry_count", 0)
    st.progress(
        min(retry_count / MAX_RETRIES, 1.0),
        text=f"Cầu dao an toàn: {retry_count}/{MAX_RETRIES} lần lỗi",
    )

    # ========================================================
    # CỔNG KIỂM DUYỆT — HUMAN-IN-THE-LOOP
    # ========================================================
    if st.session_state.phase == "awaiting_approval":
        st.divider()
        st.subheader("🛑 Cổng kiểm duyệt — Human-in-the-loop")
        st.info(
            "Đồ thị đang **dừng (interrupt_before)** ngay trước node **Developer**. "
            "Hãy duyệt bản vẽ để Developer bắt đầu code, hoặc gửi yêu cầu chỉnh sửa "
            "để hệ thống lái ngược về Architect."
        )

        st.markdown("#### 📐 Bản vẽ kiến trúc chờ duyệt")
        with st.container(border=True):
            st.markdown(values.get("architecture_plan", "_(bản vẽ trống)_"))

        col_approve, col_reject = st.columns(2)

        with col_approve:
            st.markdown("**Phương án 1 — Đồng ý:**")
            if st.button("✅ Duyệt bản vẽ — cho Developer chạy", type="primary", use_container_width=True):
                with st.spinner(
                    f"Developer đang code, QA đang exec() kiểm thử "
                    f"(tự chữa lành tối đa {MAX_RETRIES} vòng)..."
                ):
                    resume_until_gate(None)
                st.rerun()

        with col_reject:
            st.markdown("**Phương án 2 — Yêu cầu sửa bản vẽ:**")
            feedback = st.text_area(
                "Nội dung cần Architect chỉnh sửa",
                key="feedback_box",
                height=110,
                placeholder="Ví dụ: Đổi giao diện từ console sang tkinter, thêm chức năng lưu lịch sử...",
            )
            if st.button("♻️ Gửi yêu cầu sửa — quay lại Architect", use_container_width=True):
                if not feedback.strip():
                    st.warning("Hãy nhập nội dung cần sửa trước khi gửi.")
                else:
                    app = st.session_state.app
                    # HUMAN REJECTION LOGIC:
                    # Ghi đè state NHƯ THỂ chính Supervisor vừa ra quyết định
                    # (as_node="supervisor") → LangGraph lập tức chạy lại
                    # conditional_edge của Supervisor với next_agent="architect"
                    # → đồ thị lái ngược về node Architect kèm human_feedback.
                    app.update_state(
                        get_config(),
                        {
                            "next_agent": "architect",
                            "human_feedback": feedback.strip(),
                        },
                        as_node="supervisor",
                    )
                    with st.spinner("Architect đang sửa bản vẽ theo phản hồi của bạn..."):
                        resume_until_gate(None)
                    st.rerun()

    # ========================================================
    # KẾT QUẢ CUỐI CÙNG
    # ========================================================
    if st.session_state.phase == "finished":
        st.divider()

        if values.get("test_result") == "PASSED":
            st.success("🎉 Sản phẩm đã vượt qua kiểm thử QA!")

            st.markdown("#### 💻 Mã nguồn cuối cùng")
            st.code(values.get("source_code", ""), language="python")
            st.download_button(
                "⬇️ Tải mã nguồn (.py)",
                data=values.get("source_code", ""),
                file_name="generated_app.py",
                mime="text/x-python",
            )

            exe_status = values.get("exe_status", "")
            if exe_status.startswith("SUCCESS::"):
                exe_path = exe_status.split("::", 1)[1]
                st.success(f"📦 DevOps đã đóng gói thành công. File thực thi: `{exe_path}`")
            elif exe_status == "FAILED":
                st.error(
                    "📦 Code chạy tốt nhưng DevOps đóng gói .exe thất bại — "
                    "xem chi tiết trong Nhật ký nhà máy phía trên."
                )
        else:
            st.error(
                f"⛔ CẦU DAO AN TOÀN ĐÃ NGẮT: QA báo lỗi đủ "
                f"{values.get('retry_count', 0)}/{MAX_RETRIES} lần. "
                f"Hệ thống dừng để tránh cháy API."
            )
            if values.get("error_logs"):
                with st.expander("🧨 Traceback cuối cùng (để bạn tự chẩn đoán)"):
                    st.code(values.get("error_logs", ""), language="text")
            if values.get("source_code"):
                with st.expander("💻 Phiên bản code cuối cùng trước khi ngắt"):
                    st.code(values.get("source_code", ""), language="python")

# ============================================================
# TRẠNG THÁI CHỜ
# ============================================================

if st.session_state.phase == "idle":
    st.info(
        "Nhập mô tả phần mềm phía trên rồi bấm **🚀 Khởi động nhà máy**. "
        "Đồ thị sẽ dừng lại tại Cổng kiểm duyệt để bạn duyệt bản vẽ kiến trúc "
        "trước khi Developer được phép viết code."
    )