# textgen-extensions

从 [oobabooga/textgen](https://github.com/oobabooga/textgen) 的 `extensions/` 目录导出，并按**纯 XPU 支线**的偏好做了两处统一改动。

## 一、设备回退规则：XPU -> CPU

所有会做设备选择的扩展统一为：

```python
device = "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
```

即：优先使用 Intel XPU，不可用时回退 CPU。**不再检测 CUDA**。

| 文件 | 改动点 |
|---|---|
| `superboogav2/chromadb.py` | 嵌入模型设备 |
| `superbooga/chromadb.py` | 嵌入模型设备 |
| `coqui_tts/script.py` | TTS 设备 |
| `whisper_stt/script.py` | Whisper 设备、显存清理、设备下拉框（`xpu` / `cpu` / `none`） |
| `silero_tts/script.py` | TTS 设备 |
| `silero_tts/test_tts.py` | TTS 设备 |
| `send_pictures/script.py` | BLIP 图片描述设备 |

## 二、嵌入模型：BGE-M3 + 三源测速回退

原先硬编码的 `sentence-transformers/all-mpnet-base-v2`（768 维）换成 **BGE-M3**（1024 维、max_seq 8192、多语言）。

新增 `embedding_source.py`，被 `superboogav2` 与 `superbooga` 共用，解析顺序是**本地优先、缺了才联网**：

1. 本地目录 `<项目根>/user_data/emb/bge-m3` 已完整 → 直接使用，**不联网、不测速**。
2. 目录不完整，但存在裸权重 `<项目根>/user_data/models/bge-m3.safetensors`
   → 同卷**硬链接**为 `model.safetensors`（不额外占 2.27 GB），只补下缺失的小文件。
3. 什么都没有 → 先测速，再从最快且可用的源完整下载 `BAAI/bge-m3` 到目录 1。

三个来源硬编码在 `SOURCES` 里，运行时按实测延迟排序，前一个失败自动顺延：

| name | 说明 | 分支 |
|---|---|---|
| `huggingface` | HuggingFace 官方 | `main` |
| `modelscope` | 魔搭 ModelScope | `master` |
| `hf-mirror` | HF-Mirror 镜像 | `main` |

需要的文件固定为这几个：

```
config.json
config_sentence_transformers.json
modules.json
sentence_bert_config.json
special_tokens_map.json
tokenizer.json
tokenizer_config.json
sentencepiece.bpe.model
1_Pooling/config.json
model.safetensors
```

### 单独跑诊断

不启动 webui 也能测速 + 装配：

```bash
python -m extensions.embedding_source
```

### 换别的模型

```bash
set TEXTGEN_EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5     # 或本地目录路径
```

设了就不走 BGE-M3 的装配逻辑。

## 注意事项

- 需要 PyTorch 构建带 XPU 支持（存在 `torch.xpu`）。旧版 PyTorch 没有 `torch.xpu`，`hasattr` 判断会安全回退到 CPU。
- 若你的环境只有 NVIDIA CUDA，本改动会让相关扩展回退到 CPU，请按需自行调整。
- **`hf-mirror.com` 会拒绝 urllib 的默认 User-Agent（403）**，所以下载统一带浏览器 UA；不带 UA 时它会在测速阶段就被判为不可用。
- 嵌入维度从 768 变成 1024。`superbooga` / `superboogav2` 用的都是内存版 `chromadb.Client`，重启即重建，不涉及磁盘索引迁移。
- `superboogav2/nltk_data/` 是上游随仓库携带的数据文件，原样保留。
- 模型权重本身**不在本仓库**，由使用方按上述规则下载到项目 `user_data/` 下。

By DSH
