# textgen-extensions

从 [oobabooga/textgen](https://github.com/oobabooga/textgen) 的 `extensions/` 目录导出，并按**纯 XPU 支线**的偏好做了两处统一改动。

## 一、设备回退规则：XPU -> CPU

所有会做设备选择的扩展统一为：

```python
device = "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
```

即：优先使用 Intel XPU，不可用时回退 CPU。本支线的目标平台就是 Intel 蓝营。

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

## 三、Gradio 4 兼容性修复

本机装的是 `gradio 4.37.2+custom.21`，它移除了 Gradio 3 的**组件级 `.update()`**。对全部 14 个扩展做了实证审计（把每处 `gr.X(...)` 的构造参数逐一对照实际的 `inspect.signature`，字面量参数的直接真造一遍），全仓库确认只有 3 处需要改：

| 文件 | 行 | 改动 |
|---|---|---|
| `whisper_stt/script.py` | 85 | `gr.Audio.update(interactive=True)` → `gr.update(interactive=True)` |
| `whisper_stt/script.py` | 87 | `gr.Audio.update(interactive=False)` → `gr.update(interactive=False)` |
| `sd_api_pictures/script.py` | 270 | `gr.Textbox.update(label=msg)` → `gr.update(label=msg)` |

其余 52 处 `gr.*` 构造全部实测通过（没有 `gr.Box`、没有失效类名、没有非法参数），也没有 `.style()` 用法。

`whisper_stt` 那两处是**必崩**的：改设备下拉框或换模型就会 `AttributeError: type object 'Audio' has no attribute 'update'`。修完已实测跑通 `reload_whispermodel`，`xpu` / `cpu` / `none` 三条分支都返回合法更新字典。

另外修了 `sd_api_pictures/script.py` 两处会让 Python 3.12+ 报 `SyntaxWarning: invalid escape sequence` 的非原始字符串（第 91、251 行），regex 行为经用例比对完全等价。

## 四、superboogav2：API 调用崩溃修复

用 OpenAI / Anthropic 兼容端点跑模型时，superboogav2 会在生成 prompt 阶段崩掉：

```
TypeError: argument should be a str or an os.PathLike object where __fspath__ returns a str, not 'NoneType'
  ... superboogav2/chat_handler.py → _get_names → load_character_memoized → sanitize_filename(None)
```

根因**不是**本仓库改动引入的：`modules/api/completions.py` 用原始请求体当 state（`generate_params = body`），只把解析好的 `name1` / `name2` 写回去，**没有写回 `character`**，于是 `state['character']` 是「键存在、值为 None」。而 `_get_names` 用的是 `state.get('character', default)` —— `.get()` 在值为 `None` 时返回 `None` 而不是默认值。

`superboogav2/chat_handler.py::_get_names` 的改动：

1. **快路径**：`state['name1']` / `state['name2']` 都可用时直接返回。UI 与兼容端点都会把规范化的名字写进 state，没必要再读角色文件。这同时消掉了每轮推理的额外 I/O —— `load_character` 会删除并重建角色头像缓存（`generate_pfp_cache`）并重新解析一遍角色 YAML，而 `load_character_memoized` 其实**并没有 memoize**（`@functools.cache` 挂在下一个函数 `load_instruction_template_memoized` 上）。
2. **慢路径**（名字不全时才走）：`state.get('character', default)` → `state.get('character') or default`，让值为 `None` 也走回退；并给 `load_character_memoized` 加 `try/except`，角色文件缺失（`load_character` 会 `raise ValueError`）时降级用已有名字，不再打断整轮推理。

实测（真实 `chat_handler` + 真实角色文件，并计数 `load_character` 调用）：

| state | 返回 | `load_character` 调用 |
|---|---|---|
| `{'character': None, 'name1': 'You', 'name2': 'Assistant'}` | `('You', 'Assistant')` | 0 |
| `{'name1': 'You', 'name2': 'Assistant'}` | `('You', 'Assistant')` | 0 |
| `{'character': 'NoSuchChar', 'name1': 'You', 'name2': 'FB'}` | `('You', 'FB')` | 0 |
| `{'name1': 'You', 'character': 'Assistant'}` | `('You', 'Assistant')` | 1 |
| `{'character': 'NoSuchChar'}` | `('You', 'NoSuchChar')` | 1（优雅降级） |

顺带修掉一个名字来源不一致：兼容端点在未传 `character` 时用的是 `shared.default_settings['character']`（`Assistant`），而旧代码回退用的是 `shared.settings['character']`，导致喂给 RAG 的历史文本人名与实际对话对象不符。现在直接用 state 里已解析好的名字。

## 注意事项

- 需要 PyTorch 构建带 XPU 支持（存在 `torch.xpu`），本支线按 `torch 2.9.0+xpu` 验证。旧版 PyTorch 没有 `torch.xpu`，`hasattr` 判断会安全回退到 CPU。
- **`hf-mirror.com` 会拒绝 urllib 的默认 User-Agent（403）**，所以下载统一带浏览器 UA；不带 UA 时它会在测速阶段就被判为不可用。
- 嵌入维度从 768 变成 1024。`superbooga` / `superboogav2` 用的都是内存版 `chromadb.Client`，重启即重建，不涉及磁盘索引迁移。
- `superboogav2/nltk_data/` 是上游随仓库携带的数据文件，原样保留。
- 模型权重本身**不在本仓库**，由使用方按上述规则下载到项目 `user_data/` 下。

By DSH
