import torch
import torch.nn as nn
import math


class DownScaleSeqAttention(nn.Module):
    def __init__(
        self,
        dmodel_in: int,
        dmodel_out: int,
        in_seq: int,
        out_seq: int,
        num_head: int,
    ):
        super().__init__()

        assert dmodel_out % num_head == 0, "out_channel must be divisible by num_heads"
        assert in_seq % out_seq == 0, "in_seq must be divisible by out_seq"

        self.dmodel_in = dmodel_in
        self.dmodel_out = dmodel_out
        self.in_seq = in_seq
        self.out_seq = out_seq
        self.num_heads = num_head
        self.head_dim = dmodel_out // num_head

        self.num_channel = in_seq // out_seq
        self.seq_proj = nn.ModuleList(
            [nn.Linear(in_seq, out_seq) for _ in range(self.num_channel)]
        )
        self.w_q = nn.ModuleList(
            [nn.Linear(dmodel_in, dmodel_out) for _ in range(self.num_channel)]
        )
        self.w_k = nn.ModuleList(
            [nn.Linear(dmodel_in, dmodel_out) for _ in range(self.num_channel)]
        )
        self.w_v = nn.ModuleList(
            [nn.Linear(dmodel_in, dmodel_out) for _ in range(self.num_channel)]
        )

        self.channel_mixer = nn.Linear(self.num_channel, 1)

    def attention(self, x, channel_id):
        batch_size = x.size(0)
        x = self.seq_proj[channel_id](x.transpose(1, 2)).transpose(1, 2)

        q = self.w_q[channel_id](x)
        k = self.w_k[channel_id](x)
        v = self.w_v[channel_id](x)

        q = q.view(batch_size, self.out_seq, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        k = k.view(batch_size, self.out_seq, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        v = v.view(batch_size, self.out_seq, self.num_heads, self.head_dim).transpose(
            1, 2
        )

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = scores.softmax(dim=-1)

        out = attn @ v

        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(batch_size, self.out_seq, self.dmodel_out)
        )

        return out

    def forward(self, x):
        assert x.shape[1] == self.in_seq
        assert x.shape[2] == self.dmodel_in

        arr = [
            self.attention(x, i) for i in range(self.num_channel)
        ]  # list of (B, S, D)
        stacked = torch.stack(arr, dim=1)  # Shape: (B, C, S, D)

        # 👇 Reshape for mixing: move C to last and apply linear
        # B, C, S, D = stacked.shape
        mixed = self.channel_mixer(stacked.permute(0, 2, 3, 1))  # (B, S, D, 1)
        output = mixed.squeeze(-1)  # (B, S, D)

        return output


class FlexScaleSeqAttentionV1(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        in_seq: int,
        hidden_seq: int,
        out_seq: int,
        num_head: int,
    ):
        super().__init__()

        assert out_channel % num_head == 0, "out_channel must be divisible by num_heads"

        self.in_channel = in_channel
        self.hidden_seq = hidden_seq
        self.out_channel = out_channel
        self.in_seq = in_seq
        self.out_seq = out_seq
        self.num_heads = num_head
        self.head_dim = out_channel // num_head

        self.hidden_seq_proj = nn.Linear(in_seq, hidden_seq)

        self.w_q = nn.Linear(in_channel, out_channel)
        self.w_k = nn.Linear(in_channel, out_channel)
        self.w_v = nn.Linear(in_channel, out_channel)

        self.out_seq_proj = nn.Linear(hidden_seq, out_seq)

    def forward(self, x):
        # start = time.time()
        # cur = time.time()
        batch_size = x.size(0)
        assert x.shape[1] == self.in_seq
        assert x.shape[2] == self.in_channel

        x = self.hidden_seq_proj(x.transpose(1, 2)).transpose(1, 2)

        # Project inputs
        q = self.w_q(x)  # [B, in_seq, out_channel]
        k = self.w_k(x)
        v = self.w_v(x)

        # print(f"Project input: {time.time() - cur:.6f}s")
        # cur = time.time()

        # Split into heads: [B, hidden_seq, num_heads, head_dim] -> [B, num_heads, in_seq, head_dim]
        q = q.view(
            batch_size, self.hidden_seq, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = k.view(
            batch_size, self.hidden_seq, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = v.view(
            batch_size, self.hidden_seq, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # print(f"Split heads: {time.time() - cur:.6f}s")
        # cur = time.time()

        # Attention: [B, num_heads, hidden_seq, head_dim] x [B, num_heads, head_dim, hidden_seq] = [B, num_heads, hidden_seq, hidden_seq]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = scores.softmax(dim=-1)

        # print(f"Attention cal: {time.time() - start:.6f}s")
        # cur = time.time()

        # Downscale across the sequence dimension
        # [B, num_heads, hidden_seq, out_seq]
        flexscale = self.out_seq_proj(attn)  # projects hidden_seq → out_seq per head

        # Apply attention: [B, num_heads, out_seq, hidden_seq] x [B, num_heads, hidden_seq, head_dim]
        out = torch.matmul(
            flexscale.transpose(-2, -1), v
        )  # [B, num_heads, out_seq, head_dim]

        # Merge heads: [B, out_seq, out_channel]
        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(batch_size, self.out_seq, self.out_channel)
        )

        # print(f"FHA forward time: {time.time() - start:.6f}s")
        return out
