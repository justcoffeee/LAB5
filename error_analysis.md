# Error analysis — GRU-200 (5ep, F1=0.8350), test 25k
# Confusion: TN=10243, FP=2257, FN=1923, TP=10577.

## 3 correctly classified

### 1. TP — "just so beautiful, in every sense" (test #12500, P=0.99+)
- True=pos, Pred=pos. Review dài, sentiment rải đều ("beautiful emotionally, visually...").
- Analysis: gates giữ được tông tích cực suốt 200 tokens; từ khóa mạnh lặp lại →
  cell state tích lũy đủ. Vùng model mạnh: review có cực tính nhất quán.

### 2. TP — negation "isn't bad just different" (test #12501)
- True=pos, Pred=pos.
- Analysis: mẫu "X isn't bad" đòi hỏi composition có thứ tự (not đảo bad).
  GRU xử lý đúng — bằng chứng sequence modeling vượt BoW ở negation cục bộ
  (khớp app_negation: "not good... did not like" → P(pos)=0.104).

### 3. TN — "I tried to like this, I really did, but..." (test #0, P(pos)=0.003)
- True=neg, Pred=neg, rất chắc chắn.
- Analysis: cấu trúc contrast "tried to like ... but it is..." — model bắt được
  "but"-turn ở cuối (recency + pattern quen thuộc trong train).

## 3 incorrectly classified

### 1. FP — contrast "will not like ... but I enjoy" (test #4, P=0.754)
- True=neg, Pred=pos.
- Category: **contrast/discourse**. "you probably will not like this movie...
  but I enjoy thes[e]" — tác giả thừa nhận phim dở nhưng bản thân vẫn thích
  (guilty pleasure). Model thấy "enjoy" ở cuối + bỏ qua lượng từ "will not like".
- Có phải lỗi sequence modeling? Một phần: cần gán đúng scope cho "not" và "but".
  Baseline TF-IDF cũng sai (app_contrast 0.329) → discourse vượt quá cả 2 họ model.

### 2. FP — "was a decent film, but I have a few issues" (test #11, P=0.987!)
- True=neg, Pred=pos, cực kỳ chắc chắn mà sai.
- Category: **contrast + overconfidence**. "decent film" đầu câu neo hidden state
  về positive; "but..." ở giữa không đủ sức kéo lại qua hàng chục tokens tiếp theo
  ("I don't fault the actors..."). Minh họa information bottleneck của vector h_T
  duy nhất: chi tiết ở giữa bị nén mất.
- Calibration cũng là vấn đề: sigmoid 0.987 cho 1 review lưỡng lự → cần calibration
  khi dùng probability làm threshold.

### 3. FN — "it is an amazing film because it dares to investigate hypocrisy" (#12506)
- True=pos, Pred=neg (P=0.015).
- Category: **lexical prior áp đảo + lập luận phức tạp**. "amazing" là +pos nhưng
  cả review đầy từ âm ("indictment", "hypocrisy", "against") phục vụ lập luận
  (phim dám phê phán → hay). Model đếm tông từ vựng thay vì hiểu lập luận —
  giới hạn của mọi classifier thiếu discourse/reasoning, không riêng RNN.

## Tổng kết (map với đề)
- Negation cục bộ ("isn't bad", "not good"): GRU **làm được** (hơn kỳ vọng BoW).
- Contrast/discourse ("but", "terrible ... surprisingly good"): **thất bại thường xuyên**
  ở cả GRU lẫn TF-IDF → cần context modeling mạnh hơn (attention đọc lại "terrible").
- Truncation 200 tokens giữ 67.7% tokens: review >200 từ mất đuôi — mà kết luận
  sentiment thường ở đuôi → một phần FN đến từ truncation, không phải gates.
- Sarcasm/rare words ít gặp trong top-lỗi; lỗi chủ đạo là discourse + lexical-prior,
  không phải "LSTM không hiểu câu" chung chung.
