"""Reproducible local preparation, training, evaluation, replay and measurement."""
from __future__ import annotations
import argparse
import json
import platform
import subprocess
import threading
import time
import uuid
from pathlib import Path
import numpy as np
from .settings import runtime_home

def doctor(home):
    import torch,psutil
    result={"platform":platform.platform(),"python":platform.python_version(),"torch":torch.__version__,
            "cuda_available":torch.cuda.is_available(),"ram_total_gib":psutil.virtual_memory().total/2**30,
            "home":str(home),"cpu":platform.processor(),"physical_cpu_cores":psutil.cpu_count(logical=False),
            "logical_cpu_cores":psutil.cpu_count(logical=True),"torch_cuda_runtime":torch.version.cuda}
    try:
        result["nvidia_driver"]=subprocess.check_output(["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader"],text=True,timeout=5).strip()
    except (OSError,subprocess.SubprocessError):
        result["nvidia_driver"]=None
    if torch.cuda.is_available():
        result.update(gpu=torch.cuda.get_device_name(),vram_mib=torch.cuda.get_device_properties(0).total_memory/2**20)
    (home/"reports/environment.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result

def replay(home, identity):
    from .data import read_manifest
    from .pipeline import CheckInPipeline
    rows={row["id"]:row for row in read_manifest(home)}
    if identity not in rows:
        raise ValueError(f"Unknown MELD identity: {identity}")
    row=rows[identity]
    pipeline=CheckInPipeline(home)
    events=[]
    for event in pipeline.stream(row["text"],row["video_path"],[],"meld-replay",identity):
        events.append(json.loads(json.dumps(event)))
        if event["type"]=="text_delta":
            print(event["text"],end="",flush=True)
        elif event["type"]=="state":
            print(json.dumps(event["state"]["emotion"]),flush=True)
    result={"input":row,"events":events,"note":"Ground-truth emotion is retained for evaluation only; not supplied to inference."}
    path=home/"reports"/f"replay-{identity.replace(':','-')}.json"
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    if not any(event["type"]=="done" for event in events):
        raise RuntimeError(f"Replay did not complete. Inspect {path}")
    print("\nSaved",path,flush=True)
    return result

def benchmark(home,turns=30,fallback_turns=10,warmup=3):
    import psutil
    from .pipeline import CheckInPipeline
    from .data import read_manifest
    rows={row["id"]:row for row in read_manifest(home,"test")}
    with np.load(home/"cache/test.npz",allow_pickle=False) as cache:
        eligible=[str(i) for i,q in zip(cache["ids"],cache["quality"]) if q[2]>0]
    if len(eligible)<max(turns,warmup):
        raise ValueError("Not enough eligible real test examples for the benchmark")
    pipeline=CheckInPipeline(home)
    samples=[]
    peaks={"gpu_used_mib":0,"python_rss_mib":0,"generator_rss_mib":0,"system_ram_used_mib":0}
    stop=threading.Event()
    try:
        pid=json.loads((home/"runtime/generator.json").read_text(encoding="utf-8-sig"))["pid"]
        generator_process=psutil.Process(pid)
    except (OSError,KeyError,ValueError,psutil.Error):
        generator_process=None
    def monitor():
        process=psutil.Process()
        while not stop.is_set():
            peaks["python_rss_mib"]=max(peaks["python_rss_mib"],process.memory_info().rss/2**20)
            memory=psutil.virtual_memory()
            peaks["system_ram_used_mib"]=max(peaks["system_ram_used_mib"],(memory.total-memory.available)/2**20)
            if generator_process:
                try: peaks["generator_rss_mib"]=max(peaks["generator_rss_mib"],generator_process.memory_info().rss/2**20)
                except psutil.Error: pass
            try:
                memory=float(subprocess.check_output(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True,timeout=3).splitlines()[0])
                peaks["gpu_used_mib"]=max(peaks["gpu_used_mib"],memory)
            except (OSError,ValueError,subprocess.SubprocessError): pass
            stop.wait(.5)
    thread=threading.Thread(target=monitor,daemon=True)
    thread.start()
    try:
        cases=[("warmup",eligible[i]) for i in range(warmup)]+[("fusion",eligible[i]) for i in range(turns)]+[("text_fallback",eligible[i%len(eligible)]) for i in range(fallback_turns)]
        for number,(path,identity) in enumerate(cases):
            row=rows[identity]
            final=None
            before=time.perf_counter()
            for event in pipeline.stream(row["text"],None if path=="text_fallback" else row["video_path"],[],"benchmark",str(number)):
                if event["type"]=="done": final=event["state"]
                if event["type"]=="error": raise RuntimeError(event["error"])
            if final is None: raise RuntimeError("Missing completion event")
            if path!="warmup":
                if final["emotion"]["source"]!=path: raise RuntimeError("Benchmark path mismatched requested case")
                samples.append({"id":identity,"path":path,"wall_ms":(time.perf_counter()-before)*1000,
                    "timing":final["timing"],"response":final["response"]["text"],"emotion":final["emotion"],
                    "text_characters":len(row["text"]),"vision":final["vision"]})
            print(f"Benchmark {number+1}/{len(cases)}: {path}",flush=True)
    finally:
        stop.set(); thread.join(timeout=4)
    summaries={}
    for path in ("fusion","text_fallback"):
        subset=[s for s in samples if s["path"]==path]
        summaries[path]={"n":len(subset)}
        for metric in ("classification_ms","first_token_ms","completion_ms"):
            values=[s["timing"][metric] for s in subset if s["timing"][metric] is not None]
            summaries[path][metric]={"p50":float(np.percentile(values,50)),"p95":float(np.percentile(values,95)),"max":float(max(values))} if values else None
    result={"warmup":warmup,"samples":samples,"summary":summaries,"observed_peaks":peaks,
        "conditions":{"frames":8,"text_max_wordpieces":128,"generation_max_tokens":96,"concurrency":1,
                      "sampling":"First eligible official test utterances in manifest order; fallback reuses their text without video. This is a MELD replay benchmark, not a webcam or browser latency study.",
                      "gpu_sampling_seconds":.5,"clock":"backend monotonic; capture excluded",
                      "memory_scope":"GPU and system RAM peaks include desktop/background processes; Python and generator RSS are process-specific."}}
    (home/"reports/benchmark.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    return {"summary":summaries,"observed_peaks":peaks}

def audit_data(home,split,count):
    from .data import read_manifest,VideoProcessor
    processor=VideoProcessor(home/"models/vision/yunet.onnx")
    rows=read_manifest(home,split)
    rng=np.random.default_rng(42)
    chosen=rng.choice(len(rows),min(count,len(rows)),replace=False)
    report=[]
    for i in chosen:
        row=rows[int(i)]
        contact=home/"reports/audit"/f"{row['id'].replace(':','-')}.jpg"
        _,quality=processor.process(row["video_path"],contact)
        report.append({**row,"quality":quality,"contact_sheet":str(contact),"manual_speaker_review":"not_reviewed"})
    out=home/f"reports/audit-{split}.json"
    out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    return {"generated_contact_sheets":len(report),"manual_review_completed":False,"report":str(out)}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["doctor","download","prepare","extract","audit-data","audit-parameters","replay","benchmark"])
    parser.add_argument("--home")
    parser.add_argument("--split",choices=["train","dev","test","all"],default="all")
    parser.add_argument("--limit",type=int)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--models-only",action="store_true")
    parser.add_argument("--count",type=int,default=150)
    parser.add_argument("--id",default="test:0:0")
    parser.add_argument("--turns",type=int,default=30)
    parser.add_argument("--fallback-turns",type=int,default=10)
    parser.add_argument("--warmup",type=int,default=3)
    args=parser.parse_args()
    home=runtime_home(args.home)
    if args.command=="doctor": result=doctor(home)
    elif args.command=="download":
        from .downloads import download_all
        result=download_all(home,include_meld=not args.models_only)
    elif args.command=="prepare":
        from .data import prepare
        result={"rows":len(prepare(home))}
    elif args.command=="extract":
        from .features import extract
        result=extract(home,splits=("train","dev","test") if args.split=="all" else (args.split,),limit=args.limit,device=args.device)
    elif args.command=="audit-data": result=audit_data(home,"train" if args.split=="all" else args.split,args.count)
    elif args.command=="audit-parameters":
        from .audit import audit_parameters
        result=audit_parameters(home)
    elif args.command=="replay": result=replay(home,args.id)
    else: result=benchmark(home,args.turns,args.fallback_turns,args.warmup)
    print(json.dumps(result,indent=2,default=str),flush=True)

if __name__=="__main__": main()
