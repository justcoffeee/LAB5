# LAB 05 — Calculations (tính tay, không dùng framework cho bước suy luận)

## Bài 1 — Một bước RNN forward

Cho x_t=[1,0], h_{t-1}=[0.5,0.2],
W_xh=[[0.5,0.2],[0.1,0.4]], W_hh=[[0.3,0.1],[0.2,0.5]], b_h=[0,0].

```
W_xh x_t = [0.5*1+0.2*0, 0.1*1+0.4*0] = [0.5, 0.1]
W_hh h_{t-1} = [0.3*0.5+0.1*0.2, 0.2*0.5+0.5*0.2] = [0.15+0.02, 0.10+0.10] = [0.17, 0.20]
a_t = [0.5,0.1] + [0.17,0.20] = [0.67, 0.30]
h_t = tanh(a_t) = [tanh(0.67), tanh(0.30)] ≈ [0.5850, 0.2913]
```

Đã verify bằng `rnn_cell.py` và khớp `torch.nn.RNNCell` (sai số <1e-6).
Ý nghĩa: hidden mới = trộn **input hiện tại** (qua W_xh) với **quá khứ**
(qua W_hh), rồi squash bằng tanh về (−1,1). Chính W_hh tái sử dụng ở mọi bước
là nguồn gốc vanishing/exploding gradient.

## Bài 2 — Information flow ("I did not like this movie")

Model đọc trái→phải, h_0=0:

- h_1=f("I",h_0): chỉ biết "I".
- h_2=f("did",h_1): biết "I did".
- h_3=f("not",h_2): **đánh dấu phủ định** — vector "not" xoay h_3 sang vùng
  "phủ định đang mở" của không gian hidden.
- h_4=f("like",h_3): input "like" (tích cực) cộng với h_3 (đang phủ định) qua
  W_hh → h_4 rơi vào vùng "thích-bị-phủ-định" ≈ tiêu cực. **"not" ảnh hưởng
  "like" qua kênh duy nhất h_3 → h_4**, dù 2 từ không kề nhau trong cùng 1 phép tính.
- h_5, h_6 tiếp tục mang dấu phủ định tới cuối câu → classifier trên h_6 đoán negative.

So với Average Word2Vec: vec = mean(w) — "not" chỉ là +1/6 vector cộng vào tổng,
không có cơ chế "đảo nghĩa từ tiếp theo". Thực nghiệm LAB04:
cos(avg(S1),avg(S2))=0.9863, gần như cùng vector → BoW mù trước negation.
RNN khắc phục bằng **state có thứ tự**, nhưng với giá: thông tin "not" phải sống sót
qua mọi bước trung gian (h_4→h_5→h_6), mỗi bước nhân thêm Jacobian — càng xa càng mờ.

## Bài 3 — Sequence order ("dog bites man" vs "man bites dog")

- Average Word2Vec: cùng multiset {dog,bites,man} → mean **đồng nhất 100%**
  (đã verify `mean(S1)==mean(S2) → True`). Không thể phân biệt bằng thứ tự —
  về mặt cơ chế, phép cộng giao hoán xóa toàn bộ thông tin vị trí.
- RNN: h("dog")→h("bites")→h("man") áp dụng dãy phép biến đổi phi tuyến **không
  giao hoán** theo 2 thứ tự khác nhau → hidden cuối khác nhau
  (verify `rnn_last(S1)==rnn_last(S2) → False`). Ai cắn ai được mã hóa trong quỹ đạo
  của hidden state, classifier đọc h_T là phân biệt được.

## Vanishing / exploding gradient (trực giác)

∂h_T/∂h_t = tích (T−t) Jacobian ∂h_k/∂h_{k−1} = diag(1−tanh²)·W_hh mỗi bước.
- Nếu ‖W_hh‖·(1−tanh²) < 1 trung bình: tích mũ → 0. Lỗi ở cuối câu (h_T) hầu như
  không truyền về đầu câu → model **không học được dependency xa**
  ("The movie ..., despite ..., was excellent": "excellent" xa "movie" hàng chục bước).
- Nếu > 1: tích bùng nổ → update giật cục, loss NaN. Thực tế dùng gradient clipping.
- Hệ quả: sequence 50→200 tokens: (1) training chậm (O(T) bước tuần tự, không song
  song như attention), (2) bộ nhớ BPTT tăng, (3) dependency >~20-30 bước gần như
  không học được với RNN thuần. Đây là động lực của gates (LSTM/GRU).

## Bài tập LSTM gates

f_t=0.9, i_t=0.2, c̃_t=0.5, c_{t−1}=0.8:

```
c_t = f_t·c_{t−1} + i_t·c̃_t = 0.9*0.8 + 0.2*0.5 = 0.72 + 0.10 = 0.82
```

- Forget gate ≈ 1 → **giữ** thông tin cũ (0.72/0.82 đến từ quá khứ). Đường cell state
  c_{t−1}→c_t gần như "đường cao tốc" cộng tính, gradient chảy về xa ít bị co —
  đó là cách LSTM chống vanishing.
- Forget gate ≈ 0 → **xóa** quá khứ (reset khi đổi chủ đề/câu mới), cell viết lại
  từ input hiện tại. Input gate điều lượng thông tin mới nạp vào.
