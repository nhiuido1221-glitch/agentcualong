# -*- coding: utf-8 -*-
"""
meta_team_config.py
===================
Bộ não của "Meta Agent" — Nhà máy sản xuất phần mềm tự động.

Kiến trúc: LangGraph StateGraph với 5 đặc vụ:
    - Supervisor (CEO)  : điều phối, giữ cầu dao an toàn retry_count.
    - Architect         : thiết kế bản vẽ kiến trúc (hỗ trợ sửa theo phản hồi người duyệt).
    - Developer         : viết code Python từ bản vẽ ĐÃ ĐƯỢC DUYỆT (interrupt_before tại node này).
    - QA                : dùng exec() chạy thử code, bắt Traceback, tự cài thư viện thiếu
                          bằng `sys.executable -m pip install` (vá lỗi Windows).
    - DevOps            : đóng gói .exe bằng PyInstaller.

Quản lý State: TypedDict. Mọi trường đều GHI ĐÈ (overwrite) giá trị mới nhất —
duy nhất `messages` dùng reducer `operator.add` để tích lũy nhật ký.

Cài đặt phụ thuộc:
    pip install -U langgraph langchain-google-genai streamlit pyinstaller
"""

import importlib.util
import operator
import re
import subprocess
import sys
import time
import traceback
import uuid
from functools import partial
from pathlib import Path
from typing import Annotated, List, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

# ============================================================
# HẰNG SỐ TOÀN CỤC
# ============================================================

MAX_RETRIES = 3                      # Cầu dao an toàn: ngắt sau đúng 3 lần lỗi
DEFAULT_MODEL = "gemini-2.5-flash"   # Model Gemini mặc định
BUILD_DIR = Path("meta_build")       # Thư mục trung gian khi đóng gói
DIST_DIR = Path("meta_dist")         # Thư mục chứa file .exe đầu ra
APP_BASENAME = "MetaApp"             # Tên file thực thi đầu ra
MAX_AUTO_INSTALLS = 5                # QA chỉ tự cài tối đa 5 thư viện thiếu / 1 lượt test


# ============================================================
# 1. QUẢN LÝ STATE (TypedDict)
# ============================================================
# LƯU Ý VỀ REDUCER:
#   - Trong LangGraph, trường KHÔNG có Annotated reducer sẽ mặc định OVERWRITE
#     (giá trị mới nhất từ node ghi đè giá trị cũ). Đây chính là hành vi ta muốn
#     cho source_code / architecture_plan / error_logs / v.v.
#   - DUY NHẤT `messages` dùng operator.add để nhật ký các Agent được nối dài,
#     không bị mất khi đi qua nhiều node.

class AgentState(TypedDict):
    task: str                                       # Yêu cầu gốc của người dùng (bất biến)
    architecture_plan: str                          # Bản vẽ của Architect (OVERWRITE)
    source_code: str                                # Code Python của Developer (OVERWRITE)
    error_logs: str                                 # Nguyên văn Traceback từ QA (OVERWRITE)
    test_result: str                                # "", "PASSED" hoặc "FAILED" (OVERWRITE)
    retry_count: int                                # Bộ đếm cầu dao an toàn (OVERWRITE)
    next_agent: str                                 # Quyết định điều phối của Supervisor (OVERWRITE)
    human_feedback: str                             # Phản hồi sửa bản vẽ của người duyệt (OVERWRITE)
    exe_status: str                                 # Kết quả đóng gói của DevOps (OVERWRITE)
    messages: Annotated[List[str], operator.add]    # Nhật ký nhà máy (TÍCH LŨY duy nhất)


def make_initial_state(task: str) -> AgentState:
    """Khởi tạo state đầy đủ mọi trường để không node nào bị KeyError."""
    return {
        "task": task,
        "architecture_plan": "",
        "source_code": "",
        "error_logs": "",
        "test_result": "",
        "retry_count": 0,
        "next_agent": "",
        "human_feedback": "",
        "exe_status": "",
        "messages": [f"🎬 [System] Nhận nhiệm vụ: {task}"],
    }


# ============================================================
# 2. TIỆN ÍCH CHỐNG LỖI ÉP KIỂU + BÓC CODE + PIP AN TOÀN
# ============================================================

def extract_text(response) -> str:
    """
    Chống lỗi ép kiểu LangChain/Gemini: response.content có thể là str
    HOẶC là List (gồm str / dict {'type': 'text', 'text': ...}).
    Hàm này luôn trả về một String duy nhất — mọi Regex chỉ chạy SAU hàm này.
    """
    content = getattr(response, "content", response)
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(getattr(item, "text", "") or ""))
        return "\n".join(p for p in parts if p).strip()
    return str(content).strip()


CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """Bóc code Python khỏi khối ```python ... ```. Lấy khối dài nhất nếu có nhiều khối."""
    matches = CODE_FENCE_RE.findall(text)
    if matches:
        return max(matches, key=len).strip()
    return text.strip()


def pip_install(package: str):
    """
    Vá lỗi môi trường Windows: gọi pip qua chính trình thông dịch đang chạy
    (`sys.executable -m pip install ...`) thay vì lệnh `pip` trực tiếp,
    tránh lỗi "'pip' is not recognized as an internal or external command".
    Trả về (ok: bool, log: str).
    """
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", package],
        capture_output=True,
        text=True,
        timeout=600,
    )
    return result.returncode == 0, (result.stdout or "") + (result.stderr or "")


# ============================================================
# 3. CÁC NODE ĐẶC VỤ
# ============================================================

# ---------- 3.1 SUPERVISOR (CEO) — điều phối thuần logic, không tốn API ----------

def supervisor_node(state: AgentState) -> dict:
    """
    CEO đọc state và ghi next_agent. Thứ tự kiểm tra là RÀNG BUỘC CHẾT:
    cầu dao an toàn retry_count được kiểm tra TRƯỚC mọi logic khác.
    """
    if state["retry_count"] >= MAX_RETRIES:
        return {
            "next_agent": "END",
            "messages": [
                f"🔴 [Supervisor] CẦU DAO AN TOÀN: đã lỗi {state['retry_count']}/{MAX_RETRIES} lần. "
                f"Ngắt hệ thống để tránh cháy API."
            ],
        }
    if not state["architecture_plan"]:
        return {
            "next_agent": "architect",
            "messages": ["🧭 [Supervisor] Chưa có bản vẽ → giao việc cho Architect."],
        }
    if state["test_result"] == "PASSED":
        return {
            "next_agent": "devops",
            "messages": ["🧭 [Supervisor] Code đã PASS kiểm thử → giao DevOps đóng gói .exe."],
        }
    return {
        "next_agent": "developer",
        "messages": ["🧭 [Supervisor] Bản vẽ sẵn sàng → giao Developer viết code (chờ duyệt tại cổng kiểm duyệt)."],
    }


# ---------- 3.2 ARCHITECT — thiết kế / sửa bản vẽ theo phản hồi người duyệt ----------

ARCHITECT_SYSTEM = (
    "Bạn là Kiến trúc sư phần mềm cấp cao (Software Architect). "
    "Nhiệm vụ: viết BẢN VẼ KIẾN TRÚC ngắn gọn, rõ ràng bằng tiếng Việt cho một ứng dụng Python "
    "chạy trong MỘT FILE duy nhất, gồm: (1) Mô tả tổng quan, (2) Danh sách hàm/lớp chính kèm chữ ký, "
    "(3) Luồng xử lý chính, (4) Thư viện cần dùng (ưu tiên thư viện chuẩn của Python). "
    "KHÔNG viết code chi tiết, chỉ viết bản vẽ."
)


def architect_node(state: AgentState, llm) -> dict:
    """Nếu có human_feedback (người duyệt từ chối bản vẽ) → chế độ SỬA. Ngược lại → thiết kế mới."""
    if state["human_feedback"]:
        user_prompt = (
            f"Nhiệm vụ gốc: {state['task']}\n\n"
            f"BẢN VẼ CŨ:\n{state['architecture_plan']}\n\n"
            f"YÊU CẦU CHỈNH SỬA TỪ NGƯỜI DUYỆT:\n{state['human_feedback']}\n\n"
            f"Hãy viết lại TOÀN BỘ bản vẽ kiến trúc đã cập nhật theo yêu cầu trên."
        )
        note = "♻️ [Architect] Sửa lại bản vẽ theo phản hồi của người duyệt."
    else:
        user_prompt = (
            f"Nhiệm vụ: {state['task']}\n\n"
            f"Hãy viết bản vẽ kiến trúc cho ứng dụng này."
        )
        note = "📐 [Architect] Thiết kế bản vẽ kiến trúc lần đầu."

    response = llm.invoke([SystemMessage(content=ARCHITECT_SYSTEM), HumanMessage(content=user_prompt)])
    plan = extract_text(response)

    return {
        "architecture_plan": plan,   # OVERWRITE bản vẽ mới nhất
        "human_feedback": "",        # Xóa phản hồi đã xử lý (overwrite bằng chuỗi rỗng)
        "messages": [note],
    }


# ---------- 3.3 DEVELOPER — viết code / tự chữa lành theo Traceback ----------

DEVELOPER_SYSTEM = (
    "Bạn là Senior Python Developer. Chỉ trả về ĐÚNG MỘT khối ```python ... ``` chứa toàn bộ code "
    "của MỘT file hoàn chỉnh, chạy được ngay. RÀNG BUỘC BẮT BUỘC: "
    "(1) TUYỆT ĐỐI không dùng input() hay bất kỳ lệnh chặn luồng chờ người dùng nào ở cấp module; "
    "(2) Mọi lệnh khởi chạy giao diện / vòng lặp chính (vd: mainloop(), app.run()) phải đặt dưới "
    "if __name__ == \"__main__\": để bộ phận QA có thể chạy kiểm thử an toàn; "
    "(3) KHÔNG bao giờ cắt bớt code hoặc dùng comment kiểu '# ... existing code'; luôn trả file toàn vẹn."
)


def developer_node(state: AgentState, llm) -> dict:
    """Nếu error_logs có Traceback → chế độ SỬA LỖI (self-healing). Ngược lại → viết mới từ bản vẽ."""
    if state["error_logs"]:
        user_prompt = (
            f"Nhiệm vụ gốc: {state['task']}\n\n"
            f"BẢN VẼ KIẾN TRÚC:\n{state['architecture_plan']}\n\n"
            f"CODE CŨ BỊ LỖI:\n```python\n{state['source_code']}\n```\n\n"
            f"TRACEBACK KHI QA CHẠY THỬ:\n{state['error_logs']}\n\n"
            f"Hãy phân tích Traceback, sửa triệt để lỗi và trả về TOÀN BỘ file code mới (không cắt bớt)."
        )
        note = f"🔧 [Developer] Tự chữa lành: sửa code theo Traceback (vòng lỗi {state['retry_count']}/{MAX_RETRIES})."
    else:
        user_prompt = (
            f"Nhiệm vụ: {state['task']}\n\n"
            f"BẢN VẼ KIẾN TRÚC ĐÃ ĐƯỢC NGƯỜI DÙNG DUYỆT:\n{state['architecture_plan']}\n\n"
            f"Hãy viết toàn bộ code cho ứng dụng theo đúng bản vẽ trên."
        )
        note = "💻 [Developer] Viết code theo bản vẽ đã được duyệt."

    response = llm.invoke([SystemMessage(content=DEVELOPER_SYSTEM), HumanMessage(content=user_prompt)])
    code = extract_code(extract_text(response))  # Luôn extract_text TRƯỚC khi Regex

    return {
        "source_code": code,   # OVERWRITE code mới nhất
        "messages": [note],
    }


# ---------- 3.4 QA — exec() chạy thử, bắt Traceback, tự cài thư viện thiếu ----------

def qa_node(state: AgentState) -> dict:
    """
    Kiểm thử THỰC TẾ bằng exec():
      - Chạy code trong scope có __name__ = "__meta_qa__" → khối
        if __name__ == "__main__": của code sinh ra sẽ KHÔNG chạy (an toàn,
        không bị treo bởi mainloop), nhưng mọi lỗi cú pháp / import / logic
        cấp module vẫn bị bắt trọn.
      - Gặp ModuleNotFoundError → tự cài bằng `sys.executable -m pip install`
        (vá lỗi Windows) rồi chạy lại, tối đa MAX_AUTO_INSTALLS lần.
      - Gặp Traceback bất kỳ → test_result = FAILED, retry_count + 1,
        error_logs = nguyên văn Traceback để Developer tự chữa lành.
    """
    code = state["source_code"]
    fail_count = state["retry_count"] + 1
    install_logs: List[str] = []

    if not code.strip():
        return {
            "test_result": "FAILED",
            "error_logs": "EmptyCodeError: Developer trả về code rỗng.",
            "retry_count": fail_count,
            "messages": [f"❌ [QA] Code rỗng (lần lỗi {fail_count}/{MAX_RETRIES}) → trả ngược về Developer."],
        }

    for _ in range(MAX_AUTO_INSTALLS):
        try:
            exec_scope = {"__name__": "__meta_qa__"}
            exec(compile(code, "<generated_app>", "exec"), exec_scope)
            return {
                "test_result": "PASSED",
                "error_logs": "",
                "messages": install_logs + ["✅ [QA] exec() chạy sạch, không có Traceback → PASSED."],
            }
        except ModuleNotFoundError as exc:
            missing = exc.name or "unknown"
            ok, log = pip_install(missing)
            if ok:
                install_logs.append(
                    f"📦 [QA] Thiếu thư viện '{missing}' → đã cài bằng "
                    f"`{Path(sys.executable).name} -m pip install {missing}` và chạy thử lại."
                )
                continue
            return {
                "test_result": "FAILED",
                "error_logs": traceback.format_exc() + "\n\n[PIP LOG]\n" + log[-1000:],
                "retry_count": fail_count,
                "messages": install_logs + [
                    f"❌ [QA] Không cài được thư viện '{missing}' "
                    f"(lần lỗi {fail_count}/{MAX_RETRIES}) → trả ngược về Developer."
                ],
            }
        except SystemExit as exc:
            if exc.code in (None, 0):
                return {
                    "test_result": "PASSED",
                    "error_logs": "",
                    "messages": install_logs + ["✅ [QA] Code kết thúc bằng SystemExit(0) → PASSED."],
                }
            return {
                "test_result": "FAILED",
                "error_logs": f"SystemExit với mã lỗi {exc.code}:\n{traceback.format_exc()}",
                "retry_count": fail_count,
                "messages": install_logs + [
                    f"❌ [QA] Code thoát với mã lỗi {exc.code} "
                    f"(lần lỗi {fail_count}/{MAX_RETRIES}) → trả ngược về Developer."
                ],
            }
        except Exception:
            tb = traceback.format_exc()
            return {
                "test_result": "FAILED",
                "error_logs": tb,   # OVERWRITE Traceback mới nhất — Developer sẽ đọc nguyên văn
                "retry_count": fail_count,
                "messages": install_logs + [
                    f"❌ [QA] Bắt được Traceback (lần lỗi {fail_count}/{MAX_RETRIES}) → trả ngược về Developer."
                ],
            }

    return {
        "test_result": "FAILED",
        "error_logs": f"DependencyError: Quá {MAX_AUTO_INSTALLS} lần tự cài thư viện mà code vẫn thiếu module.",
        "retry_count": fail_count,
        "messages": install_logs + [
            f"❌ [QA] Quá nhiều thư viện thiếu (lần lỗi {fail_count}/{MAX_RETRIES}) → trả ngược về Developer."
        ],
    }


# ---------- 3.5 DEVOPS — đóng gói .exe bằng PyInstaller ----------

def devops_node(state: AgentState) -> dict:
    """
    Đóng gói .exe bằng PyInstaller — phiên bản chống WinError 5 (Access is denied):
      1. KHÔNG dùng cờ --clean → PyInstaller không gọi shutil.rmtree xóa thư mục
         cũ, tránh đụng File Lock của antivirus / IDE trên Windows.
      2. Mỗi lần build sinh một build_tag duy nhất (timestamp + uuid) gắn vào
         --workpath, --distpath và --specpath → luôn ghi vào thư mục mới toanh,
         không bao giờ đụng độ file cũ đang bị khóa.
      3. Toàn bộ lời gọi PyInstaller bọc trong try/except Exception → dù lỗi gì
         node vẫn trả về dict hợp lệ cho Supervisor, KHÔNG làm sập luồng LangGraph.
    """
    msgs: List[str] = []

    # Đảm bảo PyInstaller tồn tại — cài bằng sys.executable -m pip (vá lỗi Windows)
    if importlib.util.find_spec("PyInstaller") is None:
        msgs.append("📦 [DevOps] Chưa có PyInstaller → cài bằng `sys.executable -m pip install pyinstaller`.")
        ok, log = pip_install("pyinstaller")
        if not ok:
            return {
                "exe_status": "FAILED",
                "messages": msgs + ["❌ [DevOps] Cài PyInstaller thất bại:\n" + log[-1000:]],
            }

    # Build tag duy nhất cho mỗi lần đóng gói: timestamp + 6 ký tự ngẫu nhiên
    build_tag = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    work_dir = BUILD_DIR / f"work_{build_tag}"
    spec_dir = BUILD_DIR / f"spec_{build_tag}"
    dist_dir = DIST_DIR / f"build_{build_tag}"

    try:
        for folder in (work_dir, spec_dir, dist_dir):
            folder.mkdir(parents=True, exist_ok=True)

        src_path = BUILD_DIR / f"generated_app_{build_tag}.py"
        src_path.write_text(state["source_code"], encoding="utf-8")

        cmd = [
            sys.executable, "-m", "PyInstaller",
            "--onefile",
            "--noconfirm",
            "--name", APP_BASENAME,
            "--distpath", str(dist_dir),
            "--workpath", str(work_dir),
            "--specpath", str(spec_dir),
            str(src_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

    except subprocess.TimeoutExpired:
        return {
            "exe_status": "FAILED",
            "messages": msgs + ["❌ [DevOps] PyInstaller chạy quá 30 phút — hủy đóng gói."],
        }
    except Exception as e:
        # Không làm sập đồ thị — trả cảnh báo về cho Supervisor / dashboard
        return {
            "exe_status": "FAILED",
            "messages": msgs + [
                f"⚠️ [DevOps] Đóng gói thất bại do lỗi cấp quyền Windows. "
                f"Vui lòng tắt các ứng dụng đang khóa thư mục và thử lại. Chi tiết: {e}"
            ],
        }

    exe_name = APP_BASENAME + (".exe" if sys.platform.startswith("win") else "")
    exe_path = dist_dir / exe_name

    if result.returncode == 0 and exe_path.exists():
        return {
            "exe_status": f"SUCCESS::{exe_path.resolve()}",
            "messages": msgs + [f"🚀 [DevOps] Đóng gói thành công: {exe_path.resolve()}"],
        }

    # PyInstaller tự thoát với mã lỗi — soi log xem có phải lỗi cấp quyền không
    tail_log = ((result.stderr or result.stdout) or "")[-1500:]
    if "WinError 5" in tail_log or "PermissionError" in tail_log or "Access is denied" in tail_log:
        return {
            "exe_status": "FAILED",
            "messages": msgs + [
                f"⚠️ [DevOps] Đóng gói thất bại do lỗi cấp quyền Windows. "
                f"Vui lòng tắt các ứng dụng đang khóa thư mục và thử lại. Chi tiết: {tail_log}"
            ],
        }
    return {
        "exe_status": "FAILED",
        "messages": msgs + ["❌ [DevOps] PyInstaller báo lỗi:\n" + tail_log],
    }


# ============================================================
# 4. LUỒNG ĐIỀU KIỆN (conditional_edges)
# ============================================================

def route_from_supervisor(state: AgentState) -> str:
    """Điểm rẽ 1: đi theo next_agent do Supervisor ghi (hoặc do người duyệt ép qua update_state)."""
    return state["next_agent"]


def route_from_qa(state: AgentState) -> str:
    """
    Điểm rẽ 2 — biến kiểm soát: test_result + retry_count.
      - PASSED                      → về Supervisor (CEO ra lệnh cho DevOps).
      - FAILED & retry_count >= 3   → về Supervisor (CEO ngắt cầu dao, trả END).
      - FAILED & retry_count < 3    → vòng ngược về Developer (self-healing).
    """
    if state["test_result"] == "PASSED":
        return "supervisor"
    if state["retry_count"] >= MAX_RETRIES:
        return "supervisor"
    return "developer"


# ============================================================
# 5. LẮP RÁP ĐỒ THỊ
# ============================================================

def build_meta_team(google_api_key: str, model_name: str = DEFAULT_MODEL):
    """
    Lắp ráp và compile đồ thị Meta Agent.
      - Dùng add_edge cho các cạnh cứng, add_conditional_edges cho 2 điểm rẽ.
      - MemorySaver + interrupt_before=["developer"]: đồ thị DỪNG trước node
        Developer để người dùng duyệt bản vẽ trên Streamlit (Human-in-the-loop).
    """
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=google_api_key,
        temperature=0.2,
    )

    graph = StateGraph(AgentState)

    # --- Các node đặc vụ ---
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("architect", partial(architect_node, llm=llm))
    graph.add_node("developer", partial(developer_node, llm=llm))
    graph.add_node("qa", qa_node)
    graph.add_node("devops", devops_node)

    # --- Cạnh cứng (add_edge) ---
    graph.add_edge(START, "supervisor")
    graph.add_edge("architect", "supervisor")   # Architect luôn báo cáo lại CEO
    graph.add_edge("developer", "qa")           # Code viết xong bắt buộc qua kiểm thử
    graph.add_edge("devops", END)               # Đóng gói xong là hoàn tất

    # --- Điểm rẽ 1: Supervisor điều phối ---
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "architect": "architect",
            "developer": "developer",
            "devops": "devops",
            "END": END,
        },
    )

    # --- Điểm rẽ 2: QA quyết định self-healing hay báo cáo CEO ---
    graph.add_conditional_edges(
        "qa",
        route_from_qa,
        {
            "developer": "developer",
            "supervisor": "supervisor",
        },
    )

    checkpointer = MemorySaver()
    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["developer"],   # Cổng kiểm duyệt Human-in-the-loop
    )