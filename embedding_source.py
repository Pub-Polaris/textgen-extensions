"""
嵌入模型统一来源解析（纯 XPU 支线）
===================================

superboogav2 与 superbooga 共用的嵌入模型装配器，取代原先硬编码的
``sentence-transformers/all-mpnet-base-v2``，默认改用 **BGE-M3**。

解析顺序（本地优先，缺了才联网）
--------------------------------
1. 本地目录 ``user_data/emb/bge-m3`` 已完整 -> 直接返回，**不联网、不测速**。
2. 目录不完整，但存在裸权重 ``user_data/models/bge-m3.safetensors``
   -> 拿它当 ``model.safetensors``（同卷硬链接，不占双份空间），只补下缺失的
      config / tokenizer 等小文件。
3. 什么都没有 -> 从测速最快的源完整下载 ``BAAI/bge-m3`` 到本地目录。

来源是硬编码的三个（顺序无关，运行时按实测延迟排序）：

===========  ================================  ========
name         说明                              分支
===========  ================================  ========
huggingface  HuggingFace 官方                  main
modelscope   魔搭 ModelScope                   master
hf-mirror    HF-Mirror 镜像                    main
===========  ================================  ========

前一个源下载失败会自动退到下一个；三个都不可用才报错。

注意：``hf-mirror`` 会拒绝 urllib 的默认 User-Agent（403），所以所有请求
统一带浏览器 UA —— 这一点已实测确认。

单独跑一次诊断（不启动 webui）：

    python -m extensions.embedding_source
"""

from __future__ import annotations

import os
import shutil
import time
import urllib.request
from pathlib import Path

try:
    from modules.logging_colors import logger
except Exception:  # 独立运行 / 脱离 webui 时兜底
    import logging

    logger = logging.getLogger("embedding_source")
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

# --------------------------------------------------------------------------
# 硬编码来源
# --------------------------------------------------------------------------

EMBEDDING_REPO = "BAAI/bge-m3"

#: hf-mirror 需要非默认 UA，否则 403
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) textgen-embedding-fetch/1.0"

SOURCES = (
    {
        "name": "huggingface",
        "label": "HuggingFace 官方",
        "template": "https://huggingface.co/{repo}/resolve/{rev}/{file}",
        "rev": "main",
    },
    {
        "name": "modelscope",
        "label": "魔搭 ModelScope",
        "template": "https://modelscope.cn/models/{repo}/resolve/{rev}/{file}",
        "rev": "master",
    },
    {
        "name": "hf-mirror",
        "label": "HF-Mirror 镜像",
        "template": "https://hf-mirror.com/{repo}/resolve/{rev}/{file}",
        "rev": "main",
    },
)

#: 测速探针文件（够小、所有源都有）
PROBE_FILE = "config.json"
PROBE_TIMEOUT = 5.0
DOWNLOAD_TIMEOUT = 120.0
CHUNK = 1024 * 1024

# --------------------------------------------------------------------------
# 项目内路径
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMB_MODELS_DIR = PROJECT_ROOT / "user_data" / "emb"
LOCAL_MODEL_DIR = EMB_MODELS_DIR / "bge-m3"

#: 裸权重候选（用户已有的那个放最前）
LOCAL_WEIGHT_CANDIDATES = (
    PROJECT_ROOT / "user_data" / "models" / "bge-m3.safetensors",
    LOCAL_MODEL_DIR / "bge-m3.safetensors",
    LOCAL_MODEL_DIR / "model.safetensors",
)

#: 组装成 sentence-transformers 目录所需的小文件
COMPANION_FILES = (
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "sentencepiece.bpe.model",
    "1_Pooling/config.json",
)

WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")

#: 权重至少得有这么大才算数（BGE-M3 约 2.27 GB）
MIN_WEIGHT_BYTES = 100 * 1024 * 1024

# --------------------------------------------------------------------------
# 内部状态
# --------------------------------------------------------------------------

_probed: list[dict] | None = None
_resolved: str | None = None


def _url(source: dict, file: str) -> str:
    return source["template"].format(repo=EMBEDDING_REPO, rev=source["rev"], file=file)


def _open(url: str, timeout: float):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=timeout)


def _fmt_error(exc: BaseException) -> str:
    reason = getattr(exc, "reason", None)
    return f"{type(exc).__name__}: {reason or exc}"


# --------------------------------------------------------------------------
# 测速
# --------------------------------------------------------------------------


def probe_sources(force: bool = False) -> list[dict]:
    """依次探测三个来源的可达性与延迟，返回按延迟升序排好的来源列表。"""
    global _probed
    if _probed is not None and not force:
        return _probed

    timed: list[tuple[float, dict]] = []
    logger.info("[embedding] 来源测速中（探测 %s）…", PROBE_FILE)
    for source in SOURCES:
        url = _url(source, PROBE_FILE)
        started = time.perf_counter()
        try:
            with _open(url, PROBE_TIMEOUT) as response:
                response.read(1)
            elapsed_ms = (time.perf_counter() - started) * 1000
            timed.append((elapsed_ms, source))
            logger.info("[embedding]   %-18s %6d ms  OK", source["name"], round(elapsed_ms))
        except Exception as exc:
            logger.info("[embedding]   %-18s    --- 不可用（%s）", source["name"], _fmt_error(exc))

    if not timed:
        raise RuntimeError(
            "[embedding] 三个来源全部不可用（huggingface / modelscope / hf-mirror），"
            "请检查网络或代理。"
        )

    timed.sort(key=lambda item: item[0])
    _probed = [source for _, source in timed]
    logger.info("[embedding] 最快来源：%s", _probed[0]["label"])
    return _probed


def sources_report(force: bool = True) -> str:
    """人类可读的测速报告（日志 / 诊断用）。"""
    lines = []
    for index, source in enumerate(probe_sources(force=force), start=1):
        lines.append(f"{index}. {source['name']} ({source['label']}) rev={source['rev']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 本地目录完整性
# --------------------------------------------------------------------------


def _find_local_weight() -> Path | None:
    for candidate in LOCAL_WEIGHT_CANDIDATES:
        try:
            if candidate.is_file() and candidate.stat().st_size >= MIN_WEIGHT_BYTES:
                return candidate
        except OSError:
            continue
    return None


def _missing_files() -> list[str]:
    """本地目录里还缺哪些文件（含权重）。"""
    missing = [name for name in COMPANION_FILES if not (LOCAL_MODEL_DIR / name).is_file()]
    if not any((LOCAL_MODEL_DIR / name).is_file() for name in WEIGHT_FILES):
        missing.append("model.safetensors")
    return missing


def is_local_model_complete() -> bool:
    if not LOCAL_MODEL_DIR.is_dir():
        return False
    if _missing_files():
        return False
    return any(
        (LOCAL_MODEL_DIR / name).is_file() and (LOCAL_MODEL_DIR / name).stat().st_size >= MIN_WEIGHT_BYTES
        for name in WEIGHT_FILES
    )


# --------------------------------------------------------------------------
# 下载
# --------------------------------------------------------------------------


def _download_one(source: dict, relative_path: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    with _open(_url(source, relative_path), DOWNLOAD_TIMEOUT) as response, open(partial, "wb") as handle:
        shutil.copyfileobj(response, handle, length=CHUNK)
    if partial.stat().st_size == 0:
        partial.unlink(missing_ok=True)
        raise RuntimeError("下载内容为空")
    os.replace(partial, destination)


def fetch_file(relative_path: str) -> dict:
    """按测速顺序逐个来源尝试下载单个文件，返回成功的来源。"""
    last_error: BaseException | None = None
    for source in probe_sources():
        try:
            _download_one(source, relative_path, LOCAL_MODEL_DIR / relative_path)
            logger.info("[embedding]   下载 %s <- %s", relative_path, source["name"])
            return source
        except Exception as exc:
            last_error = exc
            logger.info("[embedding]   %s 下载 %s 失败：%s", source["name"], relative_path, _fmt_error(exc))
    raise RuntimeError(f"[embedding] 所有来源都无法下载 {relative_path}（最后错误：{last_error}）")


def _link_or_copy(source: Path, destination: Path) -> None:
    """同卷优先硬链接，省一份 2.27 GB；不行再复制。"""
    destination.unlink(missing_ok=True)
    try:
        os.link(source, destination)
        logger.info("[embedding]   硬链接 %s -> %s", source, destination)
        return
    except OSError:
        pass
    shutil.copy2(source, destination)
    logger.info("[embedding]   复制 %s -> %s", source, destination)


# --------------------------------------------------------------------------
# 对外入口
# --------------------------------------------------------------------------


def ensure_embedding_model(force_download: bool = False) -> str:
    """保证本地 BGE-M3 可用，返回可直接交给 sentence-transformers 的目录路径。"""
    global _resolved
    if _resolved is not None and not force_download:
        return _resolved

    if is_local_model_complete() and not force_download:
        logger.info("[embedding] 本地 BGE-M3 已完整，跳过联网：%s", LOCAL_MODEL_DIR)
        _resolved = str(LOCAL_MODEL_DIR)
        return _resolved

    LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    weight = _find_local_weight()
    if weight is not None:
        target = LOCAL_MODEL_DIR / "model.safetensors"
        if weight.resolve() != target.resolve():
            _link_or_copy(weight, target)
        logger.info("[embedding] 复用本地权重：%s", weight)

    missing = _missing_files()
    if missing:
        best = probe_sources()[0]
        logger.info(
            "[embedding] 需补齐 %d 个文件，使用最快来源 %s：%s",
            len(missing),
            best["label"],
            ", ".join(missing),
        )
        for relative_path in missing:
            fetch_file(relative_path)

    if not is_local_model_complete():
        raise RuntimeError(f"[embedding] 装配失败，{LOCAL_MODEL_DIR} 仍不完整")

    logger.info("[embedding] BGE-M3 就绪：%s", LOCAL_MODEL_DIR)
    _resolved = str(LOCAL_MODEL_DIR)
    return _resolved


if __name__ == "__main__":
    print("模型仓库:", EMBEDDING_REPO)
    print("装配目录:", LOCAL_MODEL_DIR)
    print("本地是否已完整:", is_local_model_complete())
    print("--- 来源测速 ---")
    print(sources_report())
    print("--- 解析结果 ---")
    print(ensure_embedding_model())
