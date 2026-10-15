# LAB 05 — Sequence Models: RNN, LSTM, GRU (IMDb sentiment)
Môn: Xử lý ngôn ngữ tự nhiên và ứng dụng | Học kỳ I - 2026

## 1. Cấu trúc nộp
```
lab05/
├── README.md
├── calculations.md          # RNN forward tay, info flow, order, vanishing, LSTM gates
├── prediction.md            # 5 predictions trước train
├── rnn_cell.py              # SimpleRNNCell (numpy) + verify khớp torch.nn.RNNCell
├── sequence_classification.ipynb  # toàn bộ thí nghiệm (retrain ~1-2 phút trên GPU)
├── results.csv              # metrics thật (GPU RTX 4060, seed=42)
├── learning_curves.png      # train/val loss (kèm demo overfit GRU-10ep)
├── confusion_matrix.png     # GRU-200 test 25k: [[10243,2257],[1923,10577]]
├── error_analysis.md        # 3 đúng + 3 sai (ví dụ thật từ test)
└── reflection.md            # RNN → Attention
```

## 2. Setup & cách chạy
- Yêu cầu: `torch` (cuda khuyến khích), `datasets`, `sklearn`, `matplotlib`.
  Dataset `stanfordnlp/imdb` tự tải qua `datasets` (cache sau lần đầu).
- Thứ tự: mở `sequence_classification.ipynb` chạy từ trên xuống.
  `rnn_cell.py` chạy độc lập: `python lab05/rnn_cell.py` (verify forward + S1/S2 + torch).
- Chia split: HF-train 25k → train 20k / val 5k (stratified, seed 42); HF-test 25k giữ nguyên.
  Vocab word-level top 10000 (min_freq 5, train-only), PAD=0/UNK=1.
  Chung mọi model: emb=100, hidden=128, 1 layer, batch=64, Adam 1e-3, BCE-logits,
  clip-grad 1.0, epochs=5 (trừ 2 run 10ep để demo overfit).

## 3. Kết quả chính (test 25k)
| Model | F1 | Ghi chú |
|---|---|---|
| TF-IDF uni + LR | **0.8832** | baseline lexical, 2.2s CPU — vẫn vua tổng thể |
| RNN-200 | 0.5032 | loss kẹt ~0.69, vanishing gradient trực quan |
| LSTM-200 | 0.8016 | gates cứu được, train 10ep → 0.8253 |
| GRU-200 | **0.8350** | tốt nhất nhóm recurrent; train 10ep overfit → 0.8288 |
| GRU-50 / GRU-100 | 0.7317 / 0.8021 | giữ 21%/41% tokens — context trade-off |

Điểm nhấn: GRU xử lý negation ("not good" → 0.104) nhưng gãy ở contrast
("terrible... surprisingly good" → 0.289, baseline cũng sai 0.329).
Probe vị trí từ khóa: BEGIN dễ nhất (margin 0.82), MID khó nhất (0.10) —
đo được pattern-familiarity lẫn memory, xem error_analysis.md.

## 4. AI assistance statement
- Tool: Muse Spark (opencode agent) + torch/sklearn/matplotlib.
- Purpose: debugging, giải thích API torch, boilerplate training loop,
  visualization, hỗ trợ preprocessing.
- Generated: khung training loop/notebook/plots; Modified: chọn IMDb + siêu tham số,
  thiết kế probe, diễn giải metrics/nguyên nhân lỗi viết thủ công từ output thật.
- Verified: mọi số trong results.csv copy từ log train (seed 42, GPU);
  `rnn_cell.py` khớp tay + torch <1e-6; notebook chạy lại end-to-end trước khi nộp.
