# Synthetic mechanism notes: RSGD và gRSGD

Phần synthetic của project PyTorch độc lập để kiểm tra cơ chế MoE-LoRA + Riemannian SGD +
gate gradient rescaling. Không dùng HuggingFace/PEFT, model pretrained hay dataset tải ngoài.
Đây là reproduction cơ chế trên synthetic regression, không phải reproduction điểm benchmark paper.

Đọc [README](../README.md) để xem cả hai track và [workflow](workflow.vi.md)
để dùng recipe runner có ghi provenance. Các lệnh dưới đây là lệnh trực tiếp,
chạy từ thư mục gốc repo; chúng không tự tạo manifest như recipe runner.

## Chạy

Python 3.10+; dependencies: PyTorch và Matplotlib. Từ thư mục project:

```powershell
python -m pip install -e .
python -m unittest discover -s tests -v
python -m moelora_repro.synthetic.experiment --steps 300 --seeds 0 1 2 --device auto --out outputs/synthetic/main
```

Trên RTX 4050 với PyTorch bản CUDA đã được cài:

```powershell
python -m moelora_repro.synthetic.experiment --device cuda --steps 300 --seeds 0 1 2 --out outputs/synthetic/cuda
```

`auto` dùng CUDA nếu khả dụng, nếu không dùng CPU. `cuda` báo lỗi rõ nếu không khả dụng.
Mô hình rất nhỏ nên overhead GPU có thể lớn hơn lợi ích tính toán; không cần tăng model size.
Không có mixed precision để tránh làm nhiễu các phép solve và so sánh gradient.

Chạy một setting hoặc giảm tải:

```powershell
python -m moelora_repro.synthetic.train --mode moe-riemannian --top-k 4 --steps 300 --device cpu --out outputs/synthetic/single
python -m moelora_repro.synthetic.experiment --steps 200 --seeds 0 --hidden-dim 16 --rank 2 --batch-size 32 --out outputs/synthetic/small
```

## File

| File | Vai trò |
|---|---|
| `src/moelora_repro/synthetic/model.py` | Frozen Linear, LoRA experts, Top-K router, hai backward |
| `src/moelora_repro/synthetic/optimizer.py` | Hai batched solve độc lập theo expert, simultaneous SGD |
| `src/moelora_repro/synthetic/train.py` | Synthetic teacher, paired RNG, train/validation, CSV/JSON |
| `src/moelora_repro/synthetic/experiment.py` | Sweep Top-K × mode × seed, aggregate, ba plot |
| `tests/test_mechanism.py` | Kiểm tra Jacobian, routing, optimizer, finite update |
| `pyproject.toml` | Dependency cơ bản: torch và matplotlib |
| `results/synthetic/main/` | Kết quả 24 runs đã lưu; môi trường từng run nằm trong JSON |

## Kiến trúc và task

Input có dimension `hidden_dim` (mặc định 32), frozen base `Linear(32,8)`.
Mỗi expert có `A_i[rank,32]`, `B_i[8,rank]`; mặc định 8 experts, rank 4.
Không có hệ số alpha/rank phụ: LoRA scale bằng 1.
Router `Linear(32,8)` học bằng SGD thường. Mô hình hỗ trợ tensor `[batch,d]`
và `[batch,tokens,d]`, routing độc lập ở từng vị trí.

Adapter output là tổng các expert `B_i A_i x` được nhân với gate sau Top-K;
base `Wx` được giữ frozen. Router nhận cùng input và chọn expert theo từng token.

Forward: `y = Wx + sum_i g_i B_i A_i x`. Top-K chọn theo softmax trên toàn bộ
experts rồi chuẩn hóa lại trên tập được chọn. Implementation dùng softmax selected
logits, tương đương mask-and-renormalize nhưng ổn định số học hơn.
Các expert nhỏ được tính dạng dense rồi mask; Top-K đúng về ngữ nghĩa nhưng
không đo lợi ích sparse dispatch. B khởi tạo zero, A random độc lập theo expert.
Base không có gradient. Tổng 1,792 parameter: 1,536 trainable và 256 frozen.

Synthetic regression: 2,048 train + 512 validation, Gaussian clusters chồng lấn.
Teacher là frozen base cộng một mixture mềm của low-rank maps phụ thuộc input,
với noise Gaussian std 0.02. Student được dùng chung base W, nhưng không biết
teacher router/experts/cluster labels. Teacher cố định giữa các K và hai mode
trong cùng seed. Task phi tuyến theo x nên các experts có gradient khác nhau.

Mặc định: 300 steps, batch 64, expert LR 0.1, router SGD LR 0.01,
damping 0.01, temperature 1, CPU threads 1. Không scheduler, weight decay,
momentum, clipping hay auxiliary router loss. Tất cả hyperparameter giống nhau
giữa hai mode và các K. Seed 0/1/2 tạo ba task/initialization độc lập;
mỗi cặp dùng cùng initialization và chuỗi minibatch. Không tune riêng để ưu ái mode nào.

## Backward và ý nghĩa của g²

RSGD dùng `g * expert_output`. gRSGD dùng chính trick được yêu cầu:

```python
sqrt_g_const = torch.sqrt(g).detach()
expert_const = expert_output.detach()
weighted_output = (
    sqrt_g_const * expert_output
    + (g - sqrt_g_const) * expert_const
)
```

Hai forward bằng nhau tới sai số floating point. Với cùng state và upstream
gradient, derivative về expert đổi từ g sang sqrt(g); derivative về gate vẫn
bằng expert output. Vì vậy hệ số khuếch đại cục bộ là `1/sqrt(g)` với g>0.
Không chia gradient đã cộng cả batch cho căn của mean gate: đó là một thuật toán khác.
Sau nhiều bước state khác nhau, không thể đòi gradient router của hai runs vẫn giống nhau.

**Phân biệt gradient và thay đổi forward thực tế:** xét một gate cố định và một
upstream gradient cố định. RSGD có g từ backward; khi thay đổi expert quay lại
mixture, thêm một g từ forward. Do đó thay đổi output bậc một theo LR mang g².
Trick detach giảm suy hao gradient, nhưng là **surrogate backward**:
thay đổi forward thực tế sau cập nhật parameter mang `g * sqrt(g) = g^(3/2)`
trong phép phân tích cục bộ này, không tự động là g. Nếu dùng chính surrogate
Jacobian để đo bước đi, bình phương sqrt(g) là g, nhưng đó là phép đo khác.
Unit test finite-update kiểm tra sự phân biệt này. Mini experiment kiểm chứng
engineering trick, không khẳng định nó tương đương hoàn toàn preconditioner lý thuyết chia g.

Ở batch nhiều sample, các gate khác nhau cùng đóng góp vào gradient mỗi expert;
norm tổng còn phụ thuộc residual, load, và sự triệt tiêu vector. Không có một
gate scalar chung để suy ra toàn bộ norm gradient từ mean gate.

Riemannian preconditioner:

```text
dA_i = solve(B_i^T B_i + lambda I, grad_A_i)
dB_i = solve(A_i A_i^T + lambda I, grad_B_i^T)^T
A_i <- A_i - lr*dA_i
B_i <- B_i - lr*dB_i
```

Cả hai solve dùng A/B trước update. Damping dương giữ hệ khả nghịch khi B=0;
với damping, các toán tử hình học là projector được regularize, không phải
projector lý tưởng chính xác. Không silently fallback khi gradient không finite.

**Top-1 đối chứng:** sau renormalize, selected g=1, hai mode giống hệt nhau.
Hard Top-1 có derivative router bằng 0 gần như mọi nơi; không có loss phụ nên
router không học trong cấu hình này. Đây là hệ quả có chủ đích của routing đã chọn.

## CSV và plots

Mỗi run có `{mode}_k{K}_seed{seed}.csv` và JSON lưu config, runtime, device,
PyTorch/Python version và summary. Log tại step 1, mỗi 10 steps, và step cuối;
đổi bằng `--log-every 1` nếu muốn ghi mỗi bước.

- `train_loss`: minibatch MSE trước update. `val_loss`: toàn validation sau update.
- `mean_selected_gate`: trung bình gate sau Top-K trên tất cả vị trí được chọn.
  Giá trị này luôn gần `1/K`; không dùng nó làm thước đo độ đồng đều.
- `gate_entropy`: entropy sau Top-K, nats; `softmax_entropy`: trước Top-K.
- `expert_i_selected_count`: số lần chọn trong minibatch được log.
- `expert_i_selected_cumulative`: tổng từ step 1, bao gồm cả steps không log.
- `expert_i_mean_gate`: mean gate khi expert đó được chọn, 0 nếu không được chọn.
- `expert_i_grad_raw`, `expert_i_grad_preconditioned`: norm kết hợp
  `sqrt(||grad_A_i||_F² + ||grad_B_i||_F²)` trước/sau solve, trước nhân LR.

`summary.csv` lưu mọi seed, gồm full-train MSE cuối, mean minibatch loss 50 steps
cuối và validation MSE. `aggregate.csv` lưu chênh lệch ghép cặp và sample SD;
SD không phải confidence interval. Phần trăm cải thiện là trung bình phần trăm
trong từng cặp seed, không phải phần trăm tính từ hai mean loss.

1. `training_loss.png`: loss vs step cho 4 K, mean ± SD qua seeds, không smooth.
2. `top_k_gap.png`: gap MSE cuối `RSGD - gRSGD` và entropy thực đo theo K.
3. `gradient_probe.png` + CSV: cố định expert output/upstream gradient, quét g;
   norm gradient chuẩn hóa theo gradient ungated có slope log-log 1 và 1/2.
   Đây là probe có kiểm soát, không phải scatter từ training.

## Kết quả đã chạy

Bảng dưới lấy từ `results/synthetic/main/aggregate.csv` đã commit: 3 seeds × 4 K × 2 modes × 300 steps.
Các JSON hiện có ghi PyTorch 2.14.0+cu126, Python 3.13.5 và NVIDIA GeForce RTX 4050 Laptop GPU.
Đây là metadata lịch sử trong file; đợt tổ chức repo không chạy lại hay xác nhận môi trường đó.

| K | Validation RSGD | Validation gRSGD | Gap ± SD | Giảm MSE theo cặp |
|---|---:|---:|---:|---:|
| 1 | 0.73915 | 0.73915 | 0.00000 ± 0.00000 | 0.00% |
| 2 | 0.57654 | 0.54514 | 0.03139 ± 0.01071 | 5.35% |
| 4 | 0.64738 | 0.58914 | 0.05823 ± 0.01031 | 8.98% |
| 8 | 0.78362 | 0.74844 | 0.03518 ± 0.00960 | 4.44% |

gRSGD tốt hơn trong cả 9 cặp có K>1; K=1 trùng hoàn toàn.
Entropy sau routing tăng khoảng 0 → 0.682 → 1.368 → 2.057 nats ở RSGD,
gần log(K). Tuy nhiên lợi ích lớn nhất ở K=4, giảm ở K=8: **không tăng đơn điệu**.
Điều này phân biệt khuếch đại gradient cục bộ (g nhỏ làm tỷ lệ 1/sqrt(g) lớn)
với chất lượng cuối cùng của training. Ở gate gần đều, tỷ lệ cục bộ khoảng sqrt(K),
nhưng nó không đảm bảo loss-gap tăng theo K.

Chỉ 3 synthetic seeds và 300 steps, chưa chứng minh cải thiện hội tụ dài hạn
hay độ tổng quát sang LLM. Thay đổi K đồng thời thay đổi tập expert hoạt động,
khả năng biểu diễn và routing; chưa cô lập entropy như một nguyên nhân riêng.
Init router nhỏ cũng chủ đích tạo gate phân tán. Không có sweep LR để tách lợi ích
rescaling khỏi tăng effective step size.

## Hướng research tiếp theo

1. **Entropy × exponent:** cố định K, sweep temperature và backward g^alpha
   (alpha=1, 0.75, 0.5), so sánh router frozen/learned. Kiểm tra alpha thích ứng
   theo entropy hoặc load có giúp expert ít được chọn mà không tăng variance.
2. **Đối chứng learning rate:** so RSGD có LR nhân sqrt(K) với gRSGD, cùng ngân
   sách tune LR/damping. Nếu gRSGD vẫn tốt khi gate không đều, nghiên cứu lợi ích
   phân bổ cập nhật giữa experts thay vì chỉ tăng tốc cập nhật chung.
3. **Geometry và gradient cancellation:** thay rank, damping, cluster overlap,
   đo condition number Gram matrices và per-sample gradient cosine. Kiểm tra
   khi nào sửa gate bị lấn át bởi conditioning hoặc gradient triệt tiêu.

## Nguồn đối chiếu

- [Repo gốc](https://github.com/THUDM/MoELoRA_Riemannian), nhánh main đã đọc ngày 2026-09-10.
- [Routing và detach trick](https://github.com/THUDM/MoELoRA_Riemannian/blob/main/moe_lora.py).
- [SGDr của tác giả](https://github.com/THUDM/MoELoRA_Riemannian/blob/main/custom_optimizer.py).
- [Paper, sections 3.1–3.3](https://arxiv.org/html/2502.15828v1).

Implementation được viết mới, giữ equations và routing rule; không copy module
PEFT của tác giả. Các phần giản lược: một frozen Linear, synthetic task,
constant LR, damping lớn hơn, không clip router, không LoRA dropout.

## Expert similarity experiment (K=4 versus K=8)

Run the unchanged baseline model while measuring three diagnostics every 20 steps
on the same 256-example validation batch:

```powershell
python -m moelora_repro.synthetic.experiment --steps 300 --seeds 0 1 2 3 4 --top-ks 4 8 `
  --similarity-every 20 --diagnostic-size 256 --device cuda `
  --out outputs/synthetic/expert_similarity
```

Outputs:

- `similarity.csv`: every checkpoint and seed.
- `expert_similarity.png`: rows are raw expert-output, `B_i A_i`, and raw-gradient
  cosine; columns are RSGD and gRSGD; lines compare K=4 and K=8.
- `similarity_tail_summary.csv`: mean of the last five finite checkpoints and the
  paired `K8 - K4` difference. Positive output/BA differences support greater
  expert redundancy at K=8; a negative gradient cosine suggests interference.

Raw expert outputs are measured before gates, so K=4 routing zeros cannot create
an artificial difference from K=8. Diagnostic gradients use `autograd.grad()` and
never alter the gradients used by training. Output and BA cosine are `NaN` at step
0 because all B matrices start at zero; this is mathematically undefined and is
excluded from the tail summary.
