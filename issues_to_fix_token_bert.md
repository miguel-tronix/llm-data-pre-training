# BPE Tokenization Issues & Fix Plan

> Analysis of `src/llm_data_pretraining/training/tokenization.py`
> Target architecture: Encoder-only (BERT-style)

---

## Critical Bugs

### 1. Typo in `TemplateProcessing` special_tokens

**File**: `tokenization.py:146`

```python
("UNK]", self.tokenizer.token_to_id("[UNK]")),
```

**Problem**: Missing `[` prefix. `"UNK]"` is not a registered special token. This will raise a runtime error because the key doesn't match any special token in the tokenizer.

**Fix**: Change to `"[UNK]"`.

---

### 2. `uint16` overflow risk

**File**: `tokenization.py:244-246`

```python
tokens_array = np.array(all_tokens, dtype=np.uint16)
```

**Problem**: `np.uint16` max value is 65,535. The `TokenizerConfig` allows `vocab_size` up to 1,000,000. If `vocab_size > 65535`, token IDs silently overflow and wrap around, corrupting the data.

**Fix**: Either:
- Add a runtime check `if self.tokenizer.get_vocab_size() > 65535: dtype = np.int32`
- Or always use `np.int32` (costs 2x disk space but is safe)

---

### 3. `"ab"` append mode for token output

**File**: `tokenization.py:250`

```python
with open(output_path, "ab") as f:
```

**Problem**: Append mode means re-running tokenization doubles the file. If the script is run twice, the second run appends to the first run's output.

**Fix**: Change to `"wb"` (write binary, overwrite).

---

### 4. Dead `max_length` config

**File**: `tokenization.py:38-40` (defined), `tokenization.py:278` (should be used)

```python
# In TokenizerConfig:
max_length: int = Field(default=512, ge=64, le=4096)

# In tokenize_text():
encoding = self.tokenizer.encode(text)  # no truncation!
```

**Problem**: `max_length` is defined in config but never passed to `encode()`. It exists as dead configuration — no truncation ever happens.

**Fix**: Either pass `max_length` to `encode(truncation=True, max_length=self.config.max_length)` or remove the field from config if not needed.

---

## Pre-tokenizer & Out-of-Vocabulary Problem (Biggest Issue)

### 5. `Whitespace` pre-tokenizer causes `[UNK]` token loss

**File**: `tokenization.py:133`

```python
self.tokenizer.pre_tokenizer = Whitespace()
```

**Problem**:
- `Whitespace` splits only on whitespace characters. Punctuation stays attached (`"hello,"` → one token `"hello,"`).
- No byte-level fallback. Any character/word unseen during training produces `[UNK]` — data loss.
- This is **not** the standard for modern BPE tokenizers. GPT-2, BERT, and most LLMs use `ByteLevel`.

**Fix**: Replace with `ByteLevel` pre-tokenizer:

```python
from tokenizers.pre_tokenizers import ByteLevel
self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
```

`ByteLevel` maps every possible byte to a token, guaranteeing zero `[UNK]` at inference time.

---

### 6. Missing `ByteLevel` decoder

**File**: `tokenization.py` (not present)

**Problem**: When using `ByteLevel` pre-tokenizer, the decoder must also be `ByteLevel` for correct `decode()` output. Currently there is no decoder set.

**Fix**: Add in `__init__`:

```python
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
self.tokenizer.decoder = ByteLevelDecoder()
```

---

### 7. Missing text normalization

**File**: `tokenization.py` (not present)

**Problem**: No unicode normalization (NFKC/NFD/NFC) or any normalizer is configured. "Hello" and "hello" are distinct tokens; "café" (NFC) and "cafe\u0301" (NFD) are treated differently even though they render identically.

**Fix**: Add a normalizer:

```python
from tokenizers.normalizers import NFKC, Lowercase, Sequence

# For BERT-style, typically NFKC + optional Lowercase:
self.tokenizer.normalizer = Sequence([NFKC()])
```

---

## Architectural Issues

### 8. In-memory token accumulation causes OOM risk

**File**: `tokenization.py:214,244`

```python
all_tokens = []       # grows unbounded
...
all_tokens.extend(batch)
...
tokens_array = np.array(all_tokens, dtype=np.uint16)
```

**Problem**: For large corpora (billions of tokens), this list grows until it exhausts RAM. There is no streaming to disk.

**Fix**: Stream tokens to disk incrementally:

```python
with open(output_path, "wb") as f:
    batch = []
    for line in f:
        text = line.strip()
        if text:
            batch.extend(self.tokenize_text(text))
            if len(batch) >= batch_size:
                f.write(np.array(batch, dtype=dtype).tobytes())
                batch = []
    if batch:
        f.write(np.array(batch, dtype=dtype).tobytes())
```

This makes `batch_size` actually meaningful for memory management.

---

### 9. Batch accumulation is misleading

**File**: `tokenization.py:231-234`

```python
if len(batch) >= batch_size:
    all_tokens.extend(batch)  # still goes to all_tokens!
    total_tokens += len(batch)
    batch = []
```

**Problem**: The `batch` list is emptied into `all_tokens` every `batch_size` steps, but `all_tokens` still holds everything. This provides zero memory benefit — it's a false sense of batching.

**Fix**: Resolved by fix #8 (streaming to disk). Remove `all_tokens` entirely and write directly to file.

---

### 10. Mixed special token schemes

**File**: `tokenization.py:28-35`

```python
special_tokens: ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<|endoftext|>"]
```

**Problem**: This mixes BERT-style (`[CLS]`, `[SEP]`, `[MASK]`, `[PAD]`) with GPT-style (`<|endoftext|>`). For BERT-style encoder-only models, `<|endoftext|>` is non-standard and won't be used.

**Fix**: Remove `<|endoftext|>` from special tokens for BERT-style training:

```python
special_tokens: ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
```

---

### 11. Remove `TemplateProcessing` (post-processing)

**File**: `tokenization.py:138-151`

**Problem**: The `TemplateProcessing` auto-wraps every sequence with `[CLS] ... [SEP]`. When you handle CLS/SEP in the training data pipeline, this double-wraps or conflicts.

**Fix** (as requested): Remove the entire `TemplateProcessing` block. The tokenizer will output raw token IDs; the training code handles special token insertion.

---

## Minor Issues

### 12. No `truncation` in `tokenize_text`

**File**: `tokenization.py:278`

If you keep `max_length` as a concept, `tokenize_text` should truncate:

```python
encoding = self.tokenizer.encode(text, truncation=True, max_length=self.config.max_length)
```

---

### 13. `.env` vs code default alignment

| Parameter | `.env` | Code default |
|---|---|---|
| `vocab_size` | 50000 | 30000 |

These should be aligned to avoid confusion about which value is actually used.

---

## Summary of Changes

| # | Priority | File | What to Change |
|---|---|---|---|
| 1 | **Critical** | `tokenization.py:146` | Fix `"UNK]"` → `"[UNK]"` |
| 2 | **Critical** | `tokenization.py:244-246` | Guard against uint16 overflow |
| 3 | **Critical** | `tokenization.py:250` | `"ab"` → `"wb"` |
| 4 | **Critical** | `tokenization.py:133` | `Whitespace` → `ByteLevel` pre-tokenizer |
| 5 | **High** | `tokenization.py:__init__` | Add `ByteLevelDecoder` |
| 6 | **High** | `tokenization.py:__init__` | Add `NFKC` normalizer |
| 7 | **High** | `tokenization.py:214-251` | Stream to disk, remove in-memory accumulation |
| 8 | **High** | `tokenization.py:138-151` | Remove `TemplateProcessing` |
| 9 | **Medium** | `tokenization.py:28-35` | Remove `<\|endoftext\|>` from special tokens |
| 10 | **Medium** | `tokenization.py:278` | Add truncation to `encode()` or remove dead config |
| 11 | **Low** | `.env` | Align `BPE_VOCAB_SIZE` with code default |

---

*Analysis date: 2026-05-13*
