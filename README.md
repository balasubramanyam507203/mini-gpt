# MiniGPT — Built from Scratch

A minimal GPT implementation built from scratch using PyTorch.
No external LLM libraries used — only pure PyTorch.

## What's Inside

- Self Attention mechanism (Q, K, V projections + causal mask)
- Feed Forward Network (Linear → ReLU → Linear)
- Transformer Block (Attention + FFN + Residual + LayerNorm)
- MiniGPT (stack of Transformer Blocks)
- Greedy Generation
- Temperature Sampling
- Stop Token Generation

## Architecture
```
Token Embedding
Positional Embedding
TransformerBlock x N
    → Self Attention (Q, K, V)
    → Add & LayerNorm
    → Feed Forward Network
    → Add & LayerNorm
Linear Head → Logits
```

## How to Run
```bash
pip install torch
python mini_gpt.py
```

## Key Concepts Covered

- Autoregressive generation (token by token)
- Causal masking (no looking at future tokens)
- Residual connections (x + sublayer)
- LayerNorm for training stability
- Temperature sampling (T<1 focused, T>1 random)
- EOS stop token

## Learning Journey

Built during a 4 week learning journey:
- Week 1 → ML Foundations
- Week 2 → PyTorch Basics
- Week 3 → Transformers Core
- Week 4 → MiniGPT + Generation
