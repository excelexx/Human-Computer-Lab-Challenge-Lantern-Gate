# Lantern Gate

## Windows setup

**You need:** Python **3.12**, Git, FFmpeg (`ffmpeg` and `ffprobe` on PATH), and an NVIDIA GPU with an up-to-date driver. Tested on an **RTX 3080 10 GB with 32 GB RAM**.

### 1. Get the project

Open **PowerShell** and paste:

```powershell
git clone https://github.com/excelexx/Human-Computer-Lab-Challenge-Lantern-Gate.git
cd Human-Computer-Lab-Challenge-Lantern-Gate
```

### 2. Install everything once

Run these commands in that same window. The model download is several GB, so give it time.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m checkin.cli download --models-only
.\.venv\Scripts\python.exe .\scripts\install-heads.py
```

**No training or MELD dataset download is needed to play.** The trained emotion classifiers are included.

### 3. Start the game

```powershell
.\scripts\start.ps1
```

Open **[localhost:7860](http://127.0.0.1:7860/)** if the browser does not open automatically.

**Next time:** open PowerShell in the project folder and run only `.\scripts\start.ps1`.

**To stop:** run `.\scripts\stop.ps1`.

## Play

1. Use **WASD or arrow keys** to walk toward Mara.
2. Click **Start camera** and allow webcam access. The camera is optional.
3. Click an example reply, or type your own and press **Enter**. Mara's reply uses your words and available emotion evidence.

Press **Esc** to leave the conversation. The first reply may take longer while the models load.

## Need help?

Run this to check your setup:

```powershell
.\.venv\Scripts\python.exe -m checkin.cli doctor
```

See the [technical guide](TECHNICAL_GUIDE.md) for troubleshooting, training, model details and custom setup paths.

## About

Lantern Gate is a pixel-art game demonstrating a local, emotion-aware NPC. It combines text and vision using MELD-trained classifiers and a local Qwen language model. All inference runs on your computer, with **4.465 billion total learned parameters**, below the 6-billion limit.

Emotion tags are estimates. Vision did **not** outperform text alone in the recorded evaluation.

- [Results and limitations](REPORT.md)
- [Model and asset credits](THIRD_PARTY.md)
