# External components and provenance

This document identifies the major reused components and material generated work. The machine-readable source of model revisions, artifact SHA256 values, paths, and parameter accounting is [manifests/models.json](manifests/models.json). Downloading a publicly accessible artifact does not by itself establish unrestricted rights to its training data or downstream use.

## Dataset

**MELD: Multimodal EmotionLines Dataset**, by Soujanya Poria, Devamanyu Hazarika, Navonil Majumder, Gautam Naik, Erik Cambria, and Rada Mihalcea, is the shared task dataset. The project uses utterance text, silent video processing, and the seven emotion labels. Audio is not an input to the models. Official train/dev/test identities are preserved.

- [Official repository and acquisition instructions](https://github.com/declare-lab/MELD)
- [Official ACL 2019 paper](https://aclanthology.org/P19-1050/)
- [Dataset publisher's Hugging Face archive](https://huggingface.co/datasets/declare-lab/MELD)

The archive URL uses the publisher's current branch, but its content is pinned by SHA256. All three annotation CSVs are pinned to official repository revision `e8cedf27b5d2877e198332c957127e16eb214afe`, with individual SHA256 checks and expected split counts. Preparation records their provenance and exact cross-split video overlaps. Three test videos have identical bytes to training videos in this official archive; primary evaluation preserves the official split, with a separate diagnostic excluding those three examples.

MELD contains excerpts from the television series *Friends*. This project supplies acquisition code and provenance rather than bundling the media for redistribution. Repository code permissions should not be treated as blanket copyright permission for the television material. Follow the dataset publisher's terms and keep any submission-specific media rights separate.

## Learned components

### YuNet face detector

The project loads `face_detection_yunet_2023mar.onnx` from OpenCV Zoo at revision `f12e12798e8314f7c074a6656816c048dcc95b7a`. The model directory carries an MIT license. YuNet finds faces; the project adds deterministic geometric quality/selection rules rather than a second learned tracker.

- [Pinned model source](https://github.com/opencv/opencv_zoo/tree/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet)
- [Model license](https://github.com/opencv/opencv_zoo/blob/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/LICENSE)

### EmotiEffLib / HSEmotion visual model

The visual checkpoint is the author's AffectNet-trained EfficientNet-B2 model `enet_b2_7.pt`, from EmotiEffLib revision `520a051c64cd191521e5934655314e769a319684`. The repository license is Apache-2.0. The original classifier remains attached and is included in the parameter inventory. This project adds its own MELD-supervised visual head; the original seven-class AffectNet output is not silently relabeled as MELD training.

- [Pinned repository](https://github.com/sb-ai-lab/EmotiEffLib/tree/520a051c64cd191521e5934655314e769a319684)
- [Pinned repository license](https://github.com/sb-ai-lab/EmotiEffLib/blob/520a051c64cd191521e5934655314e769a319684/LICENSE)
- [Author's distributed weights](https://github.com/sb-ai-lab/EmotiEffLib/blob/520a051c64cd191521e5934655314e769a319684/models/affectnet_emotions/enet_b2_7.pt)

The code license and the underlying training-data terms are different questions. AffectNet has its own access/license process; its academic agreement restricts dataset use to noncommercial research/education and addresses commercial access separately. This project does not download or redistribute AffectNet itself, and does not claim that the repository's Apache license resolves every restriction relevant to derived weights or commercial deployment. Treat commercial permission as unresolved here.

- [AffectNet owner's dataset page](https://mohammadmahoor.com/pages/databases/affectnet/)
- [AffectNet academic agreement, May 2024](https://mohammadmahoor.com/wp-content/uploads/2024/06/AffectNet-Agreement-v2.1-10May2024.pdf)

The author distributes this checkpoint as a serialized complete PyTorch module. The loader checks its exact pinned hash before using the legacy module loader. It does not accept arbitrary uploaded model files.

### Microsoft DeBERTa-v3-large

The text encoder uses `microsoft/deberta-v3-large` at revision `64a8c8eab3e352a784c658aef62be1662607476f`, whose model card identifies an MIT license. Tokenizer/configuration/weight artifacts are pinned and hash-checked. The clean pretrained encoder is used with a new MELD head; a third-party MELD fine-tune is not substituted.

- [Official model card](https://huggingface.co/microsoft/deberta-v3-large)
- [Pinned artifact revision](https://huggingface.co/microsoft/deberta-v3-large/tree/64a8c8eab3e352a784c658aef62be1662607476f)
- [Microsoft DeBERTa repository](https://github.com/microsoft/DeBERTa)

### Qwen3-4B-Instruct-2507 and quantization

The response model is `Qwen/Qwen3-4B-Instruct-2507`, original revision `cdbee75f17c01a7cc42f958dc650907174af0554`, under Apache-2.0. The local artifact is the **Q5_K_M GGUF conversion published by bartowski**, revision `ae44f08e1392f39c0e474af10c3ff8355c8b6688`. It is a third-party quantization of Qwen's weights, not an original unquantized checkpoint or a model trained by this project.

- [Official model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- [Original pinned configuration](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/cdbee75f17c01a7cc42f958dc650907174af0554/config.json)
- [Pinned quantizer model card](https://huggingface.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF/blob/ae44f08e1392f39c0e474af10c3ff8355c8b6688/README.md)

The quantized file is hash-verified. Its lower storage precision does not reduce the reported 4,022,468,096 learned parameters. Response behavior and quantization effects must be evaluated with this actual artifact; upstream model benchmarks are not this project's measured response quality.

### Project-trained heads

The visual, text, and fusion heads are defined in `src/checkin/models.py` and trained by `src/checkin/train.py`. They are new project components, with MELD as their supervised data source. Training reports and checkpoint metadata establish which heads actually exist in a runtime directory. No trained head is claimed merely because its architecture is defined.

## Runtimes and libraries

The native inference server is [llama.cpp](https://github.com/ggml-org/llama.cpp) release `b11146`, using the Windows CUDA 12.4 binaries. llama.cpp uses the [MIT license](https://github.com/ggml-org/llama.cpp/blob/b11146/LICENSE). Accompanying CUDA runtime libraries remain subject to NVIDIA's terms; the llama.cpp code license does not replace them.

Major Python dependencies include [PyTorch](https://pytorch.org/), [torchvision](https://github.com/pytorch/vision), [timm](https://github.com/huggingface/pytorch-image-models), [Transformers](https://github.com/huggingface/transformers), [SentencePiece](https://github.com/google/sentencepiece), [OpenCV](https://opencv.org/), [NumPy](https://numpy.org/), [scikit-learn](https://scikit-learn.org/), [Gradio](https://github.com/gradio-app/gradio), [Pydantic](https://docs.pydantic.dev/), [HTTPX](https://www.python-httpx.org/), [psutil](https://github.com/giampaolo/psutil), and [Matplotlib](https://matplotlib.org/). Their dependencies and licenses are supplied with the installed distributions; this list is attribution, not a replacement for those notices. `pyproject.toml` and the resolved dependency lock define versions.

[FFmpeg](https://ffmpeg.org/) and ffprobe are external tools used by Gradio for video inspection/transcoding. Applicable LGPL/GPL and codec terms depend on the selected build; the user-installed binary is not embedded in this source repository. The interface uses system fonts with no downloaded font assets.

## AI-generated work and ownership of results

Codex generated or substantially assisted the project-specific implementation, integration code, training/evaluation utilities, response prompt, tests, interface styling, and documentation. This is material AI assistance, not evidence that any particular model result is correct.

The handoff owner should retain responsibility for the resulting code and claims: inspect the source, run the tests, review visual sampling, verify measured reports, and distinguish completed runs from planned experiments. The repository records checkable artifacts instead of asserting that generated code has inherently been validated. Test fixtures deliberately use artificial events and are labeled as such; they never replace the live inference path or measured MELD predictions.

When redistributing allowed components, preserve their applicable license and attribution notices. This provenance document is not a grant of additional rights to third-party weights, datasets, software, or media.
