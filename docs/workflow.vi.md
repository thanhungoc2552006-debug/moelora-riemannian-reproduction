# Làm việc với repo theo trọng tâm LLM

Phần chính là fine-tune Llama trên ScienceQA. Phần synthetic hỗ trợ kiểm chứng
cơ chế và được gom trọn trong [experiments/synthetic/](../experiments/synthetic/README.md).

## Tìm và sửa đúng chỗ

| Bạn cần làm gì? | Mở ở đâu? |
| --- | --- |
| Sửa model, optimizer hoặc training LLM | `src/moelora_repro/scienceqa/` |
| Thay K, seed hoặc ngân sách chạy LLM | `configs/scienceqa/` |
| Xem kết quả LLM đã lưu | `results/scienceqa/` |
| Đọc, chạy hoặc sửa thí nghiệm synthetic | `experiments/synthetic/` |
| Xem các lần chạy mới trên máy | `outputs/` |
| Hiểu protocol và giới hạn so với paper | `docs/reproduction.md` |

Trong folder synthetic đã có code, `configs/`, `results/`, README và
`mechanism.md`. Bạn không cần tìm phần này ở nhiều nhánh thư mục khác nhau.
Các kiểm thử vẫn ở `tests/` để chạy chung bằng một lệnh.

## 1. Cài đặt

Từ thư mục gốc repo, với Python 3.10 trở lên:

```bash
python -m pip install -e ".[scienceqa]"
```

Nếu chỉ chạy synthetic trên CPU, dùng `python -m pip install -e .`.
Nếu đã cài bản cấu trúc trước, chạy lại lệnh cài để Python nhận vị trí package mới.
Hai file `requirements*.txt` trung gian đã bỏ; các dependency nằm trong
`pyproject.toml`.

Với Colab/Jupyter, chuyển thư mục tới repo rồi dùng `%pip install -e ".[scienceqa]"`.
ScienceQA cần GPU phù hợp và quyền tải model từ Hugging Face.

## 2. Chọn cấu hình rồi chạy

Ví dụ xem cấu hình gRSGD dài hạn ở K=10:

```bash
python -m moelora_repro.run --config configs/scienceqa/long_grsgd_k10.json --dry-run
```

Lệnh này chỉ hiện cấu hình. Bỏ `--dry-run` để thực sự tải model/dữ liệu và train.
Đọc [các giới hạn protocol](reproduction.md), đặc biệt vấn đề cắt mất token đáp án,
trước khi dùng lần chạy mới để kết luận về chất lượng reproduce.

Đổi K và seed của một pilot mà không tạo thêm file train:

```bash
python -m moelora_repro.run --config configs/scienceqa/pilot_grsgd_k4.json --set top_k=8 --set seed=43
```

Nếu muốn lưu thí nghiệm lâu dài, copy recipe sang tên mới, đổi trường `name` và
commit code/cấu hình trước khi chạy. Chỉ sửa những tham số driver có hỗ trợ;
rank, số expert và learning rate ScienceQA hiện vẫn là các hằng số trong code.

Runner tạo thư mục mới dưới `outputs/`, lưu `recipe.json`, `manifest.json`,
`console.log` và các kết quả của driver. Tên thư mục dựa trên tên recipe; đọc
`recipe.json` để biết tham số thực tế sau khi áp dụng `--set`.
`--out outputs/my_run_01` cho phép đặt tên riêng nhưng thư mục phải chưa tồn tại.

## 3. Kiểm tra và lưu kết quả

Kiểm tra trạng thái trong manifest, loss/metric và diagnostics. Khi một lần chạy
đã được kiểm tra, copy cả kết quả lẫn cấu hình/log sang đúng nơi:

| Loại thí nghiệm | Nơi lưu kết quả được đưa lên GitHub |
| --- | --- |
| LLM/ScienceQA | `results/scienceqa/<tên-thí-nghiệm>/` |
| Synthetic | `experiments/synthetic/results/<tên-thí-nghiệm>/` |

Thêm README theo [mẫu thí nghiệm](experiment-template.md), ghi câu hỏi nghiên cứu,
setting, seed, kết quả và giới hạn. Cập nhật đúng danh mục:
[kết quả LLM](../results/README.md) hoặc
[kết quả synthetic](../experiments/synthetic/results/README.md).

Không tạo thêm `train_final.py` hay `train_v2.py` khi chỉ đổi K/seed. Dataset,
cache và weights lớn để ngoài Git. CSV/log đã lưu không phải checkpoint để
resume model nếu Colab ngắt kết nối.

## 4. Chạy kiểm tra nhẹ

```bash
python -m unittest discover -s tests -v
python -m moelora_repro.run --config experiments/synthetic/configs/smoke.json
```

Lệnh thứ hai chỉ chạy synthetic nhỏ trên CPU, không chạy Llama.

## 5. Những file cũ đã bỏ

`train.py`, `experiment.py` ở ngoài cùng và folder `real_task/` từng là các file
gọi trung gian. Chúng đã được bỏ; toàn bộ code chính vẫn còn ở các vị trí mới.
Dùng runner như các ví dụ trên hoặc xem [bảng lệnh module](structure.md).

Các đường dẫn trước khi tổ chức lại repo và ghi chú tích hợp code cũ nằm trong
[lịch sử thay đổi](history.md).
