"""LAB 05 - Simple RNN cell tu dau (numpy) + kiem chung voi torch.nn.RNNCell.

h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h)
"""
import numpy as np


class SimpleRNNCell:
    def __init__(self, input_size, hidden_size, seed=42):
        rng = np.random.default_rng(seed)
        # Xavier init
        s1 = np.sqrt(1.0 / input_size)
        s2 = np.sqrt(1.0 / hidden_size)
        self.W_xh = rng.uniform(-s1, s1, (hidden_size, input_size))
        self.W_hh = rng.uniform(-s2, s2, (hidden_size, hidden_size))
        self.b_h = np.zeros(hidden_size)

    def forward(self, x_t, h_prev):
        """x_t: (input_size,), h_prev: (hidden_size,) -> h_t: (hidden_size,)"""
        a_t = self.W_xh @ x_t + self.W_hh @ h_prev + self.b_h
        return np.tanh(a_t)

    def forward_sequence(self, xs, h0=None):
        """xs: (T, input_size) -> (T, hidden_size). Doc sequence tu trai sang phai."""
        h = np.zeros(self.W_hh.shape[0]) if h0 is None else h0
        outs = []
        for x_t in xs:
            h = self.forward(x_t, h)
            outs.append(h)
        return np.stack(outs)


if __name__ == "__main__":
    # 1) Bai 1 trong de
    cell = SimpleRNNCell(2, 2, seed=0)
    cell.W_xh = np.array([[0.5, 0.2], [0.1, 0.4]])
    cell.W_hh = np.array([[0.3, 0.1], [0.2, 0.5]])
    cell.b_h = np.zeros(2)
    h = cell.forward(np.array([1.0, 0.0]), np.array([0.5, 0.2]))
    print("a_t =", (cell.W_xh @ np.array([1., 0.]) + cell.W_hh @ np.array([.5, .2])).tolist())
    print("h_t =", h.tolist(), "(ky vong ~[0.5850, 0.2913])")

    # 2) S1 vs S2: thu tu khac nhau -> hidden cuoi khac nhau
    rng = np.random.default_rng(1)
    toy_emb = {"dog": rng.normal(size=4), "bites": rng.normal(size=4),
               "man": rng.normal(size=4)}
    c2 = SimpleRNNCell(4, 4, seed=1)
    s1 = np.stack([toy_emb[w] for w in ["dog", "bites", "man"]])
    s2 = np.stack([toy_emb[w] for w in ["man", "bites", "dog"]])
    print("mean(S1)==mean(S2):", np.allclose(s1.mean(0), s2.mean(0)), "(avg-W2V mu)")
    print("rnn_last(S1)==rnn_last(S2):",
          np.allclose(c2.forward_sequence(s1)[-1], c2.forward_sequence(s2)[-1]),
          "(RNN phan biet duoc)")

    # 3) Kiem chung voi torch.nn.RNNCell ( nonlinearity=tanh )
    import torch
    tcell = torch.nn.RNNCell(2, 2, nonlinearity="tanh")
    with torch.no_grad():
        tcell.weight_ih.copy_(torch.tensor(cell.W_xh, dtype=torch.float32))
        tcell.weight_hh.copy_(torch.tensor(cell.W_hh, dtype=torch.float32))
        tcell.bias_ih.zero_(); tcell.bias_hh.zero_()
        th = tcell(torch.tensor([[1.0, 0.0]]), torch.tensor([[0.5, 0.2]]))
    print("torch h_t =", th.tolist(), "match:", np.allclose(th.numpy(), h, atol=1e-6))
