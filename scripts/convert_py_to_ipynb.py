#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
KỊCH BẢN CHUYỂN ĐỔI PYTHON SCRIPT (.PY) SANG JUPYTER NOTEBOOK (.IPYNB)
Hỗ trợ chuẩn hóa các khối Interactive Window (# %% hoặc # %% [Cell N]: Tiêu đề).
Tương thích 100% chuẩn Jupyter Notebook nbformat v4.5 không cần cài thêm thư viện phụ thuộc.

Vị trí lưu: D:\DATN\City2Graph-Hanoi\scripts\convert_py_to_ipynb.py

Cách sử dụng:
1. Chuyển đổi một file cụ thể:
   python scripts/convert_py_to_ipynb.py notebooks/hanoi/05_visualization_hanoi.py
2. Chuyển đổi file với tên file xuất tùy chọn:
   python scripts/convert_py_to_ipynb.py file_input.py file_output.ipynb
3. Chuyển đổi toàn bộ các file .py trong thư mục notebooks/hanoi/:
   python scripts/convert_py_to_ipynb.py notebooks/hanoi/
4. Tự động chuyển đổi toàn bộ các file chính của đề tài Hà Nội:
   python scripts/convert_py_to_ipynb.py --all
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

CWD = Path.cwd().resolve()
PROJECT_ROOT = CWD
while not (PROJECT_ROOT / "data").exists() and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent


def build_notebook_skeleton() -> Dict[str, Any]:
    """Khởi tạo cấu trúc từ điển tiêu chuẩn của Jupyter Notebook (nbformat 4.5)."""
    return {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


def format_source_lines(lines: List[str]) -> List[str]:
    """
    Chuẩn hóa các dòng mã nguồn cho cell Jupyter Notebook:
    Mỗi dòng kết thúc bằng '\n', ngoại trừ dòng cuối cùng không có ký tự xuống dòng dư.
    """
    if not lines:
        return []
    formatted = [line + "\n" for line in lines]
    formatted[-1] = formatted[-1].rstrip("\r\n")
    return formatted


def make_code_cell(lines: List[str]) -> Dict[str, Any]:
    """Tạo một cell code chuẩn Jupyter Notebook."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": format_source_lines(lines)
    }


def make_markdown_cell(lines: List[str]) -> Dict[str, Any]:
    """Tạo một cell markdown chuẩn Jupyter Notebook."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": format_source_lines(lines)
    }


def extract_top_docstring(lines: List[str]) -> Tuple[List[str], int]:
    """
    Bóc tách khối docstring mở đầu tệp (nếu có) để chuyển thành cell Markdown tiêu đề.
    Trả về (danh sách dòng tiêu đề markdown, chỉ số dòng tiếp theo cần đọc).
    """
    i = 0
    doc_lines: List[str] = []
    in_doc = False
    quote_mark = ""

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Bỏ qua các dòng shebang và khai báo encoding
        if i == 0 and (stripped.startswith("#!") or stripped.startswith("# -*- coding")):
            i += 1
            continue

        if not in_doc and (stripped.startswith('"""') or stripped.startswith("'''")):
            quote_mark = stripped[:3]
            in_doc = True
            content = stripped[3:].strip()
            if content:
                doc_lines.append(content)

            # Trường hợp docstring đóng ngay trên cùng một dòng
            if len(stripped) > 3 and stripped.endswith(quote_mark):
                if content.endswith(quote_mark):
                    doc_lines[-1] = content[:-3].strip()
                in_doc = False
                i += 1
                break

            i += 1
            continue

        if in_doc:
            if quote_mark in line:
                content = line.split(quote_mark)[0].strip()
                if content:
                    doc_lines.append(content)
                in_doc = False
                i += 1
                break
            else:
                doc_lines.append(line.rstrip("\r\n"))
            i += 1
            continue

        # Nếu gặp dòng code thực tế đầu tiên mà không phải docstring
        if stripped and not stripped.startswith("#"):
            break

        i += 1

    return doc_lines, i


def parse_py_script_to_notebook(py_text: str) -> Dict[str, Any]:
    """
    Phân tích cú pháp toàn diện tệp Python script thành notebook:
    - Nhận diện các điểm ngắt # %%, # %% [Cell N], # %% [markdown].
    - Bóc tách ghi chú và phân biệt giữa code cell và markdown cell.
    """
    nb_dict = build_notebook_skeleton()
    all_lines = py_text.splitlines()

    # 1. Trích xuất docstring đầu tệp
    top_doc_lines, start_idx = extract_top_docstring(all_lines)
    if top_doc_lines:
        clean_doc = [l for l in top_doc_lines if l.strip()]
        if clean_doc:
            md_header = [f"# {clean_doc[0]}"] + clean_doc[1:]
            nb_dict["cells"].append(make_markdown_cell(md_header))

    # 2. Duyệt qua toàn bộ các dòng còn lại
    current_cell_type = "code"
    current_cell_lines: List[str] = []

    def commit_current_cell():
        nonlocal current_cell_lines, current_cell_type
        # Loại bỏ các dòng trống dư thừa ở đầu và cuối cell
        while current_cell_lines and not current_cell_lines[0].strip():
            current_cell_lines.pop(0)
        while current_cell_lines and not current_cell_lines[-1].strip():
            current_cell_lines.pop()

        if current_cell_lines:
            if current_cell_type == "markdown":
                nb_dict["cells"].append(make_markdown_cell(current_cell_lines))
            else:
                nb_dict["cells"].append(make_code_cell(current_cell_lines))
        current_cell_lines = []

    idx = start_idx
    while idx < len(all_lines):
        line = all_lines[idx]
        stripped = line.strip()

        # Nhận diện điểm ngắt cell Interactive (# %%)
        if stripped.startswith("# %%"):
            commit_current_cell()

            # Phân định kiểu cell kế tiếp
            if "[markdown]" in stripped.lower() or "[md]" in stripped.lower():
                current_cell_type = "markdown"
            else:
                current_cell_type = "code"

            # Giữ lại chú thích tiêu đề khối (# %% [Cell N]: ...)
            header_match = re.search(r"# %%\s*(?:\[Cell\s*\d+\]:?\s*)?(.*)", stripped)
            if header_match:
                header_title = header_match.group(1).strip()
                if header_title and not stripped.startswith("# %% [markdown]"):
                    current_cell_lines.append(f"# {header_title}")

            idx += 1
            continue

        current_cell_lines.append(line)
        idx += 1

    commit_current_cell()

    # Nếu file không chứa marker # %%, gom toàn bộ thành 1 cell duy nhất
    if not nb_dict["cells"]:
        clean_lines = [l for l in all_lines if l.strip()]
        if clean_lines:
            nb_dict["cells"].append(make_code_cell(all_lines))

    return nb_dict


def convert_single_file(py_path: Path, ipynb_path: Path | None = None) -> bool:
    """Chuyển đổi một tệp Python script sang Jupyter Notebook."""
    if not py_path.exists():
        print(f"✗ Không tìm thấy tệp: {py_path.resolve()}")
        return False

    if ipynb_path is None:
        ipynb_path = py_path.with_suffix(".ipynb")

    try:
        with open(py_path, "r", encoding="utf-8") as f:
            py_text = f.read()

        nb_dict = parse_py_script_to_notebook(py_text)

        ipynb_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ipynb_path, "w", encoding="utf-8") as f:
            json.dump(nb_dict, f, ensure_ascii=False, indent=1)

        cell_count = len(nb_dict["cells"])
        print(f"✓ [{cell_count:2d} cells] {py_path.name:<45} ──► {ipynb_path.name}")
        return True
    except Exception as e:
        print(f"✗ Lỗi khi chuyển đổi {py_path.name}: {e}")
        return False


def get_default_project_files() -> List[Path]:
    """Tìm kiếm danh sách các tệp mã nguồn chính thức của đề tài Hà Nội."""
    search_dirs = [
        PROJECT_ROOT / "notebooks" / "hanoi",
        PROJECT_ROOT / "notebooks",
        Path(r"D:\DATN\City2Graph-Hanoi\notebooks\hanoi"),
        Path(r"D:\DATN\City2Graph-Hanoi\notebooks"),
    ]

    target_names = [
        "01_data_processing_hanoi.py",
        "02_graph_construction_hanoi.py",
        "03_model_training_hanoi.py",
        "04_evaluation_hanoi.py",
        "05_visualization_hanoi.py",
        "appendix_evaluation_hdbscan_hanoi.py",
    ]

    found_files = []
    for name in target_names:
        for d in search_dirs:
            p = d / name
            if p.exists() and p not in found_files:
                found_files.append(p)
                break

    return found_files


def main():
    parser = argparse.ArgumentParser(
        description="Chuyển đổi tệp mã nguồn Python (.py) sang Jupyter Notebook (.ipynb)"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Đường dẫn tệp .py hoặc thư mục chứa các tệp .py cần chuyển đổi"
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Đường dẫn tệp .ipynb đầu ra tùy chọn (chỉ áp dụng khi chuyển đổi 1 tệp)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Tự động quét và chuyển đổi toàn bộ 6 file chính của dự án Hà Nội"
    )

    args = parser.parse_args()

    print("=" * 90)
    print("CÔNG CỤ CHUYỂN ĐỔI PYTHON SCRIPT (.PY) ──► JUPYTER NOTEBOOK (.IPYNB)")
    print(f"Thư mục làm việc: {PROJECT_ROOT.resolve()}")
    print("=" * 90)

    # Trường hợp 1: Kích hoạt cờ --all hoặc chạy không đối số
    if args.all or (args.input is None):
        project_files = get_default_project_files()
        if not project_files:
            print("! Không tự động dò thấy các file .py của đề tài trong notebooks/hanoi/.")
            print("  Vui lòng chỉ định đường dẫn cụ thể. Ví dụ:")
            print("  python scripts/convert_py_to_ipynb.py notebooks/hanoi/05_visualization_hanoi.py")
            return

        print(f"\n-> Đang chuyển đổi danh sách {len(project_files)} tệp mã nguồn đề tài Hà Nội sang .ipynb:")
        print("-" * 90)
        success_cnt = 0
        for py_f in project_files:
            if convert_single_file(py_f):
                success_cnt += 1
        print("-" * 90)
        print(f"✓ HOÀN TẤT: Đã chuyển đổi thành công {success_cnt}/{len(project_files)} notebook(s)!")
        print("=" * 90)
        return

    input_path = Path(args.input).resolve()

    # Trường hợp 2: Đường dẫn truyền vào là một thư mục
    if input_path.is_dir():
        py_files = sorted(list(input_path.glob("*.py")))
        if not py_files:
            print(f"! Không tìm thấy tệp .py nào trong thư mục: {input_path}")
            return

        print(f"\n-> Tìm thấy {len(py_files)} tệp .py trong {input_path.name}/:")
        print("-" * 90)
        success_cnt = 0
        for py_f in py_files:
            if convert_single_file(py_f):
                success_cnt += 1
        print("-" * 90)
        print(f"✓ HOÀN TẤT: Đã chuyển đổi thành công {success_cnt}/{len(py_files)} notebook(s)!")
        print("=" * 90)
        return

    # Trường hợp 3: Đường dẫn truyền vào là một tệp đơn lẻ
    if input_path.is_file():
        output_path = Path(args.output).resolve() if args.output else None
        print("\n-> Đang thực hiện chuyển đổi tệp đơn lẻ:")
        print("-" * 90)
        if convert_single_file(input_path, output_path):
            print("-" * 90)
            print("✓ CHUYỂN ĐỔI THÀNH CÔNG!")
        print("=" * 90)
        return

    print(f"✗ Đường dẫn không hợp lệ: {input_path}")


if __name__ == "__main__":
    main()