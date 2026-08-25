# TODO — Training-AI-for-emotion (công việc còn lại)

> Cập nhật lần cuối: sau audit branch `ml-optimization-v2`.
> Release frozen: `d5676375c61d67d984363e539c26d22a19c9f346` — KHÔNG bao giờ amend/rebase.
> Quy tắc: không fabricate metrics; `NOT MEASURED` không được đổi thành `PASS`;
> test set chỉ dùng cho final comparison.

## A. ML / Model optimization

- [ ] **A1. E0 V1 reproduction hoàn thành trên Colab GPU**
      Trạng thái: RUNNING (epoch ~2/25 tại lần theo dõi cuối).
      Hoàn thành khi: MLflow run = FINISHED, đủ 25 epochs, có test metrics + confusion matrix.
      ⚠️ Chỉ tin kết quả từ MLflow/artifact thật, không tin log giữa chừng.
- [ ] **A2. Đánh giá HARD GATE E0** — so với baseline acc 49.83% / macro-F1 0.4210 / Fear recall 0.046
      → REPRODUCED | APPROXIMATELY REPRODUCED | NOT REPRODUCED | INCOMPLETE.
      Nếu NOT REPRODUCED → STOP, investigation trước khi chạy bất kỳ experiment nào.
- [ ] **A3. E1 — class-weighted loss** (`configs/train_v2_weighted.yaml`) — chỉ chạy sau gate PASS. Weights derive từ TRAIN only.
- [ ] **A4. E2 — layer4 fine-tune** (`configs/train_v2_finetune.yaml`) — controlled, cùng seed/split.
- [ ] **A5. E3 — combined** (`configs/train_v2_combined.yaml`) — chỉ nếu E1/E2 có evidence hữu ích.
- [ ] **A6. ERROR_ANALYSIS_V2.md** — confusion Fear/Disgust/Anger/Sadness V1 vs best candidate (dựa trên matrix thật).
- [ ] **A7. MODEL_COMPARISON_V2.md** — bảng E0/E1/E2/E3 với số đo thật.
- [ ] **A8. MODEL_PROMOTION_DECISION.md** — V1 RETAINED hoặc V2 PROMOTED.
      Tiêu chí: Macro-F1 tăng AND Fear recall tăng materially AND không regress nặng class khác AND fit 512MB CPU deployment.
      Kết quả tiêu cực là bằng chứng hợp lệ — không ép V2.
- [ ] **A9. Class imbalance là HYPOTHESIS** — chưa được chứng minh cho đến khi E1 đo được. Không ghi là root cause trong docs.
- [ ] **A10. Phân biệt bắt buộc trong docs:** temporal gating giảm nhiễu thời gian, KHÔNG sửa classification bias.

## B. Deployment (operator-blocked)

- [ ] **B1. REVOKE Cloudflare token cũ** (`cfat_X9U0…` đã lộ trong chat) — CHƯA XÁC NHẬN. Token mới chỉ nằm trong Koyeb secret.
- [ ] **B2. Neon PostgreSQL provision** → DATABASE_URL vào Koyeb secret (không dán vào chat/Git).
- [ ] **B3. Koyeb deploy** (Dockerfile, port 8080) → HARD GATE: `/health` + `/info` HTTP 200 trước Pages.
- [ ] **B4. Cloudflare Pages** với API_BASE_URL → verify zero secret trong assets, HTTPS/WSS.
- [ ] **B5. Real LLM matrix 8/8 trên backend public** — NOT MEASURED hiện tại (chỉ có bằng chứng trước đó).
- [ ] **B6. Fallback live regression + cold start + RAM production** — NOT MEASURED.
- [ ] **B7. Free-tier dashboard audit** — "Expected $0/month while within current free-tier quotas", không nói "always free".

## C. Kỹ thuật nhỏ

- [x] **C1. Commit optimization artifacts** vào branch `ml-optimization-v2` (train.py, configs×4, audits, reproduce script).
- [x] **C2. Push branch lên origin** để backup trước khi Colab tiếp tục.
- [x] **C2b. Sửa 5 bug trong train.py** (phát hiện khi smoke test sau commit):
      1. Docstring lạc làm file không compile được (unterminated triple-quote)
      2. Indentation sai ở `set_seed`, `FERDataset.__getitem__`, `run_hash`, `return final_path`
      3. `build_model` trả `None` với config "head" (return nằm trong if)
      4. `~` trong config path không expand (`Path("~/.cache/...")` là literal)
      5. YAML parse `1e-5` thành string → ép `float()` cho optimizer params
      Verification: py_compile OK + CPU smoke test PASS end-to-end
      (train→val→checkpoint→MLflow, loss=1.92, val chạy, ckpt saved, 23.5s)
- [ ] **C3. Review `training.ipynb`** — ĐÃ scan: không secret/path cục bộ; notebook 1 cell khung, cần hoàn thiện các section theo protocol Colab khi chạy thật.
- [x] **C4a. Full regression sau fix: 55 passed / 7 skipped** — production không bị ảnh hưởng.
- [ ] **C4. Sau A8 + B7: freeze final** — cập nhật README/VALIDATION_REPORT với metrics cuối, quyết định kép:
      (V1 RETAINED | V2 PROMOTED) + (PUBLICLY DEPLOYED — VERIFIED | DEPLOYMENT-BLOCKED).

## Điều kiện đóng project

Tất cả mục trên hoàn thành HOẶC được đánh dấu là kết quả khoa học hợp lệ
(COMPUTE-BLOCKED / V1 RETAINED / DEPLOYMENT BLOCKED kèm lý do được ghi nhận).
Sau đó: chuyển sang Bosch CV + demo 3–5 phút + interview prep. DỪNG phát triển feature.
