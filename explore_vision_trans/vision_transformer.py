# Idea:
# Image batch_size, channel (3), W, H -> Con2d (out_channel=head, kernel_size=1)
# batch_size, out_channel, W * H -> Position Embedding
# Query (batch_size, W*H, out_channel ) @ Key (batch_size, out_channel, W*H) each out_channel could consider is a head
# attention_score (batch_size, out_channel, W*H, W*H) @ Merge matrix (batch_size, out_channel, W*H, W*H / n)
# reduce_dim_attention_score (batch_size, out_channel, W*H, W*H / n) transpose (3, 4) @ Value (batch_size, out_channel, W*H )
# (batch_size, out_channel, W*H / n) normalize across dim = 1
# batch_size, W*H / n -> batch_size, W / n, H / n -> repeat until enough
