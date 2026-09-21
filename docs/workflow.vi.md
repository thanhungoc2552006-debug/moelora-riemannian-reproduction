# Cách làm việc với repo từ bây giờ

## Quy ước chính

| Bạn muốn thêm gì? | Đặt ở đâu? |
| --- | --- |
| Sửa model, optimizer, cách train synthetic | `src/moelora_repro/synthetic/` |
| Sửa code ScienceQA | `src/moelora_repro/scienceqa/` |
| Thử K, seed hoặc số bước khác | Tạo/sửa recipe trong `configs/`, hoặc dùng `--set` |
| Kết quả vừa chạy | `outputs/`; runner tự tạo thư mục riêng |
| Kết quả đã kiểm tra và muốn đưa lên GitHub | `results/<track>/<tên-thí-nghiệm>/` |
| Giải thích cơ chế hoặc ghi chú protocol | `docs/` |

Không cần tạo thêm `train_final.py`, `train_new.py`, `train_v2.py` mỗi khi đổi
K hay seed. Thay các tham số đã được hỗ trợ trong recipe, giữ code chạy ở một
chỗ. Khi đổi hẳn protocol, đặt tên thí nghiệm mới và ghi rõ điểm khác biệt.

## 1. Cài và kiểm tra

Từ thư mục repo, dùng Python 3.10 trở lên:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m moelora_repro.run --config configs/synthetic/smoke.json
```

Với Colab/Jupyter, dùng `%pip install -e .` sau khi chuyển thư mục tới repo.
ScienceQA cần thêm `%pip install -e ".[scienceqa]"`, GPU phù hợp và tài khoản
Hugging Face đã được cấp quyền tải model. Hướng dẫn này không chạy huấn luyện
ScienceQA giúp bạn trên Colab.

## 2. Thêm một thí nghiệm

Ví dụ muốn chạy gRSGD với K=8, seed=43 trên pilot ScienceQA:

```bash
python -m moelora_repro.run --config configs/scienceqa/pilot_grsgd_k4.json --set top_k=8 --set seed=43 --dry-run
```

Lệnh trên chỉ hiển thị cấu hình. Bỏ `--dry-run` để thực sự tải dữ liệu/model và
train. Nếu đây là thí nghiệm bạn định giữ lâu dài, hãy copy recipe sang tên mới,
đổi cả trường `name`, rồi commit code và cấu hình trước khi chạy.

Runner tạo một thư mục mới, ví dụ `outputs/scienceqa-pilot-grsgd-k4/<run-id>/`.
`recipe.json` ghi tham số thực tế sau khi áp dụng `--set`; `manifest.json` ghi
commit, môi trường và trạng thái; `console.log` ghi log. Tên thư mục tự động dựa
trên tên recipe, nên khi dùng override cần đọc recipe đi kèm để biết đúng cấu hình.
Bạn cũng có thể dùng `--out outputs/scienceqa_k8_seed43_run01`; thư mục này phải
chưa tồn tại để tránh ghi đè kết quả.

## 3. Đưa kết quả lên GitHub

Kiểm tra `manifest.json` có trạng thái `completed`, xem loss/metric có hợp lệ
không, và đọc diagnostics trước khi kết luận. Với ScienceQA, cần xử lý hạn chế
tokenization đã ghi trong [protocol](reproduction.md) trước các kết luận mới.

Copy toàn bộ thư mục chạy đã kiểm tra sang `results/`, giữ cả cấu hình và log.
Thêm README theo [mẫu thí nghiệm](experiment-template.md): câu hỏi nghiên cứu,
setting, seed, kết quả, diễn giải và giới hạn. Thêm dòng tương ứng vào
[danh mục kết quả](../results/README.md).

Chỉ stage những đường dẫn có chủ đích. Ví dụ:

```bash
git add configs/scienceqa/my_experiment.json results/scienceqa/my_experiment results/README.md
git commit -m "results: add ScienceQA K8 paired-seed comparison"
```

Thay các tên ví dụ bằng file thực tế của bạn. Dataset, cache, checkpoint và
weights lớn để ngoài Git. CSV đã lưu trong lúc train không phải checkpoint để
resume model khi Colab ngắt kết nối.

## 4. Làm việc với nhánh và PR

Đợt tổ chức này nằm trên nhánh `refactor/research-layout`, xuất phát từ `main`
có các kết quả dài hạn mới nhất tại thời điểm kiểm tra. Mở nhánh này trên GitHub
để xem cấu trúc mới; `main` chỉ đổi khi bạn quyết định merge PR.

PR #7 và #8 là các thay đổi từ trước. Đợt này không merge hay đóng chúng.
Do nhiều file được chuyển chỗ, cần đưa các sửa lỗi của chúng sang đường dẫn
mới trước khi kết hợp. Xem [bảng chuyển đường dẫn và quan hệ PR](structure.md).
