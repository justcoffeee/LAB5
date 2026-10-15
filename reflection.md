# Reflection — RNN → Attention

## So sánh với LAB04 (bảng của đề, điền số thật)

| Property | TF-IDF+LR (0.8832) | AvgW2V+LR | RNN (0.5032) | LSTM (0.8016) | GRU (0.8350) |
|---|---|---|---|---|---|
| Sparse | có | không | không | không | không |
| Dense/Embedding | không | có | có (học từ đầu) | có | có |
| Word order | không | không | có (lý thuyết) | có | có |
| Context-dependent state | không | không | có (h_T) | có | có |
| Long dependency | không | không | yếu (F1≈0.5!) | cải thiện | tốt nhất nhóm |
| Cost | 2.2s CPU | thấp | 5.7s GPU | 7.9s GPU | 6.3s GPU |

RNN/LSTM/GRU giải quyết được gì mà AvgW2V không? **Thứ tự và composition cục bộ**:
S1/S2 ("dog bites man") phân biệt được (verify số), negation "isn't bad"/"not good"
xử lý đúng (P(pos)=0.104). AvgW2V coi 2 câu là 1 vector.

## 3 bài học trung thực (đối chiếu prediction)

1. **RNN thuần sụp đổ đúng như lý thuyết**: loss kẹt 0.68-0.69 suốt 5 epochs,
   F1=0.503 ≈ đoán mò trên data cân bằng. Vanishing gradient không còn là công thức
   mà là đường cong phẳng lì. (Prediction 2 ✔: LSTM/GRU > RNN rõ rệt.)
2. **Baseline lexical vẫn là vua tổng thể** (0.883 > 0.835): đa số review IMDb
   phân biệt được bằng từ vựng ("excellent/terrible/waste"); gates chỉ "đáng tiền"
   ở thiểu số câu negation/contrast. (Prediction 5 ✔: không có model thắng mọi thứ;
   Prediction 1 chỉ đúng một nửa — RNN dùng word order nhưng F1 tổng vẫn thua.)
3. **Context-length trade-off có thật**: GRU 50→100→200 tokens: F1
   0.732→0.802→0.835 (giữ 21%→68% tokens) nhưng time tăng, và train 10 epochs thì
   GRU overfit (val loss tăng từ ep6, F1 rớt 0.835→0.829). (Prediction 3 ✔ một nửa:
   chậm hơn nhưng ở đây F1 vẫn tăng tới 200; Prediction 4 ✔: cắt 30 tokens chỉ giữ
   12.7% tokens, F1 GRU-50 chỉ 0.73.)

## RNN còn vấn đề gì → vì sao cần Attention?

1. **Nút cổ chai h_T**: cả review 200 từ nén vào 1 vector 128 chiều; FP #11
   ("decent film, but...") cho thấy giữa-câu bị rơi rớt. Muốn đọc lại "terrible"
   khi đoán thì phải có **truy cập trực tiếp mọi vị trí** — chính là Query/Key/Value.
2. **Tuần tự O(T), không song song theo thời gian**; BPTT qua 200 bước vừa chậm
   vừa mờ gradient. Attention xử lý toàn sequence song song với đường gradient O(1).
3. **Probe cho thấy model học prior mẫu câu hơn là composition**: "terrible" ở cuối
   câu lạ mẫu ("but the ending was terrible" → 0.537) yếu hơn cùng từ ở đầu câu quen
   mẫu (0.152). Static h_T + gates vẫn là *trí nhớ*, chưa phải *lập luận*.
   → LAB06: cho token "nhìn" thẳng token quan trọng (attention), rồi LAB07 Transformer.
