# textgen-extensions

从 [oobabooga/textgen](https://github.com/oobabooga/textgen) 的 `extensions/` 目录导出，并按本机偏好统一了设备选择逻辑。

## 设备回退规则：XPU -> CPU

所有会做设备选择的扩展统一为：

```python
device = "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
```

即：优先使用 Intel XPU，不可用时回退 CPU。**不再检测 CUDA**。

| 文件 | 改动点 |
|---|---|
| `superboogav2/chromadb.py` | 嵌入模型设备；嵌入模型名提取为 `EMBEDDING_MODEL_NAME` 常量 |
| `coqui_tts/script.py` | TTS 设备 |
| `whisper_stt/script.py` | Whisper 设备、显存清理、设备下拉框（`xpu` / `cpu` / `none`） |
| `silero_tts/script.py` | TTS 设备 |
| `silero_tts/test_tts.py` | TTS 设备 |
| `send_pictures/script.py` | BLIP 图片描述设备 |

## 注意事项

- 需要 PyTorch 构建带 XPU 支持（存在 `torch.xpu`）。旧版 PyTorch 没有 `torch.xpu`，`hasattr` 判断会安全回退到 CPU。
- 若你的环境只有 NVIDIA CUDA，本改动会让相关扩展回退到 CPU，请按需自行调整。
- `superboogav2/nltk_data/` 是上游随仓库携带的数据文件，原样保留。

By DSH
