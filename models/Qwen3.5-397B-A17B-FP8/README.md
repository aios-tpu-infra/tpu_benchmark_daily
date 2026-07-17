# Qwen3.5-397B-A17B-FP8 metadata snapshot

This directory contains only the configuration, tokenizer, preprocessing
metadata, and license from `Qwen/Qwen3.5-397B-A17B-FP8`. It intentionally does
not contain model weights or a Safetensors index.

The benchmark server uses this directory together with vLLM's
`--load-format dummy`, so startup never needs to contact Hugging Face or read
the 397B checkpoint. `SOURCE.json` records the exact upstream revision used to
create this snapshot.
