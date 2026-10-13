# Cách làm việc với repo

## Bố cục

| Việc cần làm | Thư mục |
| --- | --- |
| Sửa model / optimizer / training ScienceQA | `src/moelora_repro/scienceqa/` |
| Chọn seed, bước train, batch và recipe | `configs/scienceqa/` |
| Xem hai run mới | `results/scienceqa/runs/` |
| Xem diagnostic | `results/scienceqa/diagnostics/` |
| Xem đúng code tạo checkpoint | `results/scienceqa/source_snapshot/` |
| Đọc/chạy/sửa toàn bộ synthetic | `synthetic/` |
| Hiểu protocol | [reproduction.md](reproduction.md) |

## Chạy ScienceQA mới

Từ thư mục gốc repo:

```bash
python -m pip install -e ".[scienceqa]"
python -m moelora_repro.run --config configs/scienceqa/grsgd_k10_seed42.json --dry-run
python -m moelora_repro.run --config configs/scienceqa/grsgd_k10_seed42.json
```

Colab: chuyển thư mục tới repo rồi dùng `%pip install -e ".[scienceqa]"`.
Recipe `configs/scienceqa/smoke.json` kiểm tra nhanh trên GPU.
Đổi seed bằng `--set seed=43`. Tạo recipe có tên riêng khi muốn lưu một thí nghiệm.
Runner tạo output mới, ghi recipe/manifest/log và không ghi đè run cũ.

## Tiếp tục một run bị ngắt

Dùng trực tiếp driver với đúng tham số và đúng output directory:

```bash
python -m moelora_repro.scienceqa.train --mode moe-riemannian --out outputs/my_run --revisions configs/scienceqa/revisions.json --steps 800 --batch-size 2 --grad-accum 40 --eval-every 100 --save-every 50 --seed 42
```

Ví dụ trên dành cho run mới của phiên bản hiện tại. Driver kiểm tra config,
code hash và dependency trước khi nạp checkpoint. Runner chỉ tạo directory mới,
không dùng để resume một directory đã tồn tại.

Checkpoint cũ trên Drive cần đúng source/package/environment trong
[source snapshot](../results/scienceqa/source_snapshot/README.md).
Không sửa metadata hoặc bỏ config guard để ép nạp vào bản mới.

## Synthetic

Toàn bộ code, configs, results, notes và mechanism tests nằm trong
[synthetic/](../synthetic/README.md).

```bash
python -m pip install -e .
python -m unittest discover -s synthetic/tests -v
python -m moelora_repro.run --config synthetic/configs/smoke.json
```

## Lưu và kiểm tra kết quả

Run mới ở `outputs/` cho đến khi được kiểm tra. Lưu ScienceQA đã kiểm tra vào
`results/scienceqa/runs/<tên-run>/`, cùng config, revisions, audit, environment,
metrics và predictions. Synthetic lưu trong `synthetic/results/`.
Weights/checkpoint vẫn ở Drive hoặc nơi lưu weights, ngoài Git.
Dùng [mẫu thí nghiệm](experiment-template.md).

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s synthetic/tests -v
python scripts/summarize_scienceqa.py
```

Số liệu ScienceQA cũ đã bỏ khỏi phần đang dùng. Truy lại tại
[nhánh archive](https://github.com/thanhungoc2552006-debug/moelora-riemannian-reproduction/tree/archive/before-scienceqa-cleanup).
