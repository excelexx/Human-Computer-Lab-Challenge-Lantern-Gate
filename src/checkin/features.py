"""Resumable feature extraction; cache identity binds models and preprocessing."""
import hashlib
import inspect
import json
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import cv2
from .data import VideoProcessor, l2_normalize, read_manifest
from .encoders import VisionEncoder, TextEncoder
from .settings import runtime_home

def feature_metadata(vision,text,processor):
    code="\n".join(inspect.getsource(obj) for obj in (VideoProcessor,l2_normalize,VisionEncoder,TextEncoder))
    return {"vision":vision.metadata(),"text":text.metadata(),"processor":processor.version,
            "frames":processor.frames,"min_valid":processor.min_valid,
            "implementation_sha256":hashlib.sha256(code.encode()).hexdigest(),
            "normalization":"mean over valid frames then L2; text masked mean then L2"}

def feature_identity(metadata):
    return hashlib.sha256(json.dumps(metadata,sort_keys=True).encode()).hexdigest()

def _atomic_npz(path, **arrays):
    temp = Path(str(path) + ".tmp")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temp, path)

def extract(home=None, splits=("train","dev","test"), limit=None, device="cuda", chunk_size=64, preprocessing_workers=4):
    home = runtime_home(home)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    processor = VideoProcessor(home / "models/vision/yunet.onnx")
    vision = VisionEncoder(home / "models/vision/enet_b2_7.pt", device=device)
    text = TextEncoder(home / "models/text", device=device, batch_size=8)
    metadata=feature_metadata(vision,text,processor)
    identity=feature_identity(metadata)
    chunk_root = home / "cache/chunks" / identity[:16]
    chunk_root.mkdir(parents=True, exist_ok=True)
    worker_state=threading.local()
    def prepare_row(item):
        row,contact=item
        if not hasattr(worker_state,"processor"):
            worker_state.processor=VideoProcessor(home / "models/vision/yunet.onnx")
        try:
            return worker_state.processor.process(row["video_path"],contact_path=contact)
        except Exception as error:
            return [],{"available":False,"reason":"preprocess_error","error":str(error),
                       "valid_frame_fraction":0.0,"mean_detection_score":0.0}
    for split in splits:
        rows = read_manifest(home, split)
        full_count = len(rows)
        if limit and limit < full_count:
            rng = np.random.default_rng(42)
            chosen = []
            for label in range(7):
                indices = [i for i,r in enumerate(rows) if r["label"] == label]
                chosen.extend(rng.choice(indices, min(len(indices), max(1,limit//14)), replace=False).tolist())
            remaining = sorted(set(range(len(rows))) - set(chosen))
            chosen.extend(rng.choice(remaining, max(0,limit-len(chosen)), replace=False).tolist())
            rows = [rows[i] for i in sorted(chosen)[:limit]]
        output_chunks, quality_rows = [], []
        start_time = time.perf_counter()
        for start in range(0, len(rows), chunk_size):
            batch = rows[start:start+chunk_size]
            inputs=[{key:r.get(key) for key in ("id","text","label","media_sha256","video_path")} for r in batch]
            id_hash=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()[:12]
            cache_path = chunk_root / f"{split}-{start:06d}-{id_hash}.npz"
            quality_path = cache_path.with_suffix(".json")
            if not (cache_path.exists() and quality_path.exists()):
                features = np.zeros((len(batch),1408),dtype=np.float32)
                quality = np.zeros((len(batch),3),dtype=np.float32)
                details = []
                items=[(row,home / "reports/contact-sheets" / f"{row['id'].replace(':','-')}.jpg" if start==0 else None) for row in batch]
                # Each worker owns its detector. Ordered map preserves annotation
                # alignment, and the 64-row chunk bounds retained image memory.
                with ThreadPoolExecutor(max_workers=max(1,preprocessing_workers)) as executor:
                    prepared=list(executor.map(prepare_row,items))
                for i,(row,(crops,q)) in enumerate(zip(batch,prepared)):
                    if crops:
                        features[i] = l2_normalize(vision.encode_faces(crops).mean(axis=0))
                        quality[i] = [q["valid_frame_fraction"],q["mean_detection_score"],1.0]
                    details.append({"id":row["id"], "label":row["label"], **q})
                lengths=[len(ids) for ids in text.tokenizer([r["text"] for r in batch],truncation=False,padding=False)["input_ids"]]
                for detail,length in zip(details,lengths):
                    detail["text_token_count"]=length
                    detail["text_truncated"]=length>text.max_length
                text_features = l2_normalize(text.encode([r["text"] for r in batch]))
                _atomic_npz(cache_path, ids=np.array([r["id"] for r in batch]),
                    labels=np.array([r["label"] for r in batch],dtype=np.int64),
                    vision=features,text=text_features,quality=quality,feature_identity=np.array(identity))
                quality_path.write_text(json.dumps(details,indent=2),encoding="utf-8")
            with np.load(cache_path,allow_pickle=False) as cached:
                output_chunks.append({k:cached[k] for k in ("ids","labels","vision","text","quality")})
            quality_rows.extend(json.loads(quality_path.read_text(encoding="utf-8")))
            elapsed=time.perf_counter()-start_time
            print(f"{split}: {min(start+chunk_size,len(rows))}/{len(rows)}; eligible {sum(q['available'] for q in quality_rows)}; elapsed {elapsed:.1f}s",flush=True)
        arrays={k:np.concatenate([chunk[k] for chunk in output_chunks],axis=0) for k in output_chunks[0]}
        _atomic_npz(home / f"cache/{split}.npz",**arrays,feature_identity=np.array(identity))
        report={"split":split,"rows":len(rows),"full_split_rows":full_count,"complete_split":len(rows)==full_count,
            "feature_identity":identity,"feature_metadata":metadata,"elapsed_seconds":time.perf_counter()-start_time,
            "quality_reasons":dict(Counter(q["reason"] for q in quality_rows)),
            "text_truncated_count":sum(q.get("text_truncated",False) for q in quality_rows),
            "eligible_by_label":dict(Counter(str(q["label"]) for q in quality_rows if q["available"]))}
        (home/f"cache/{split}.metadata.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
        (home/f"reports/quality-{split}.jsonl").write_text("".join(json.dumps(q)+"\n" for q in quality_rows),encoding="utf-8")
        print(json.dumps(report,indent=2),flush=True)
    return {"feature_identity":identity,"splits":list(splits)}
