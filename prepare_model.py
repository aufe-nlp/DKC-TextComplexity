#!/usr/bin/env python3
"""Prepare the exact original hfl/chinese-roberta-wwm-ext weights on AutoDL."""
import argparse
import hashlib
import os
from pathlib import Path
import sys
import urllib.request

EXPECTED_SHA256 = "1ded5a5a1c7841dee6e47942f7b5bf2bcf6f73ff19197580f852f7f638f86b35"
EXPECTED_SIZE = 411578458
ROOT = Path(__file__).resolve().parent
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="下载并核验原始RoBERTa权重；已有完整safetensors时只检查加载")
    parser.add_argument("--endpoint", default="https://huggingface.co", help="默认官方地址；如需镜像请显式提供地址")
    args = parser.parse_args()
    import torch
    from transformers import AutoModel, AutoTokenizer
    root = ROOT / "chinese-roberta-wwm-ext"
    safe, binary = root / "model.safetensors", root / "pytorch_model.bin"
    if not safe.exists():
        version = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
        if version < (2, 6):
            raise RuntimeError("只有.bin权重时需要PyTorch 2.6以上。请选用相应AutoDL镜像后再准备模型。")
        if not binary.is_file() or binary.stat().st_size != EXPECTED_SIZE or sha(binary) != EXPECTED_SHA256:
            url = args.endpoint.rstrip("/") + "/hfl/chinese-roberta-wwm-ext/resolve/main/pytorch_model.bin"
            temporary = root / "pytorch_model.bin.download"
            print(f"下载原始权重（约412MB）：{url}", flush=True)
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as f:
                total, checkpoint = 0, 0
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    f.write(block)
                    total += len(block)
                    if total - checkpoint >= 32 * 1024 * 1024:
                        print(f"  已下载 {total / 1024**2:.0f} MiB", flush=True)
                        checkpoint = total
            if temporary.stat().st_size != EXPECTED_SIZE or sha(temporary) != EXPECTED_SHA256:
                raise RuntimeError("下载文件与原始Git LFS指针的大小/哈希不符，已保留.download文件，未替换权重")
            os.replace(temporary, binary)
        print("原始二进制权重大小和SHA-256通过，转换为safetensors。", flush=True)
        model = AutoModel.from_pretrained(root, use_safetensors=False, local_files_only=True)
        model.save_pretrained(root, safe_serialization=True)
    model = AutoModel.from_pretrained(root, local_files_only=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    if model.config.hidden_size != 768 or model.config.num_hidden_layers != 12:
        raise RuntimeError("本地资源不是预期的12层、768维RoBERTa中文底座")
    with torch.inference_mode():
        outputs = model(**tokenizer("这是一条加载检查评论。", return_tensors="pt"), output_hidden_states=True)
    if not torch.isfinite(outputs.last_hidden_state).all():
        raise RuntimeError("模型加载后的输出存在非有限值")
    print("完整权重已就绪，离线加载和真实前向检查通过。")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"模型准备失败：{error}", file=sys.stderr)
        sys.exit(1)
