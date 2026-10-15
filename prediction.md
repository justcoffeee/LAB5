# Prediction (viết TRƯỚC khi train — không sửa sau khi thấy kết quả)

Task: sentiment IMDb (positive/negative, cân bằng 50/50).

## Prediction 1 — TF-IDF+LR vs RNN+Embedding, model nào dùng word order trực tiếp hơn?
- Prediction: **RNN+Embedding**.
- Reason: TF-IDF (unigram) xóa thứ tự ngay ở bước representation; LR chỉ thấy
  "có/không có từ". RNN đọc w_1→w_T, h_t=f(x_t,h_{t−1}) nên "not good" và
  "good" đi qua 2 quỹ đạo hidden khác nhau. Kỳ vọng RNN hơn rõ ở câu có
  negation/contrast ("I expected it bad, but it was wonderful").
- Confidence: cao (85%).

## Prediction 2 — RNN vs LSTM, ai xử lý dependency dài tốt hơn?
- Prediction: **LSTM**.
- Reason (cơ chế, không phải "hiện đại hơn"): RNN truyền thông tin bằng 1 vector h
  qua phép nhân ma trận + tanh lặp lại → Jacobian co dần (vanishing). LSTM có thêm
  cell state c_t = f_t·c_{t−1} + i_t·c̃_t: khi forget≈1, gradient chảy thẳng về quá khứ
  theo đường cộng tính, không bị squash liên tục. Review dài 200 từ → LSTM giữ được
  sentiment ở đầu câu tốt hơn.
- Confidence: trung bình-cao (75%). GRU kỳ vọng ≈ LSTM (ít tham số hơn, train nhanh hơn).

## Prediction 3 — Sequence length 50 → 200, training RNN thay đổi thế nào?
- Prediction: chậm hơn ~3-4x (O(T) bước tuần tự), tốn RAM hơn (lưu activations cho BPTT),
  và khó học dependency xa hơn (vanishing nặng hơn). F1 có thể **không tăng**,
  thậm chí giảm nếu padding/noise nhiều.
- Reason: compute tuyến tính theo T nhưng không song song hóa được theo thời gian;
  phần đuôi dài của review IMDb thường là lan man, thêm nhiễu.
- Confidence: cao (80%).

## Prediction 4 — Truncate mọi review về 30 tokens, thông tin classification mất không?
- Prediction: Có, mất đáng kể. F1 giảm rõ (kỳ vọng −3-8 điểm).
- Reason: mở đầu review IMDb hay là tóm tắt/câu dẫn ("I rented this..."), sentiment
  thật ("...was excellent/terrible") thường nằm giữa/cuối. Cắt 30 tokens giữ lại
  đúng phần ít sentiment nhất. % tokens retained ở max_len=30 chỉ ~15-20%.
- Confidence: cao (85%).

## Prediction 5 — LSTM có chắc chắn accuracy cao hơn RNN trên mọi dataset không?
- Prediction: Không chắc.
- Reason: gates chỉ phát huy khi dependency dài **và** đủ data để học điều khiển gates
  (LSTM ~4x tham số RNN). Dataset ngắn/nhỏ/dư tín hiệu lexical (như SMS spam LAB04)
  thì RNN nhỏ, thậm chí TF-IDF+LR, có thể ngang hoặc hơn mà rẻ hơn nhiều.
  "Mạnh hơn về cơ chế" ≠ "thắng mọi benchmark".
- Confidence: cao (80%).
