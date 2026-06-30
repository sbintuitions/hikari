---
license: mit
library_name: transformers
pipeline_tag: automatic-speech-recognition
language:
  - en
  - ja
  - ru
  - de
tags:
  - speech-translation
  - simultaneous-translation
  - streaming-asr
  - whisper
  - s2tt
model_id: sbintuitions/hikari-medium
---

# Hikari-medium

[![GitHub](https://img.shields.io/badge/GitHub-sbintuitions%2Fhikari-181717?logo=github)](https://github.com/sbintuitions/hikari)
[![arXiv](https://img.shields.io/badge/arXiv-2603.11578-b31b1b.svg)](https://arxiv.org/abs/2603.11578)

**Hikari-medium** is a streaming speech-to-text translation and transcription model. It performs simultaneous, low-latency translation directly from audio, without waiting for an utterance to finish.

## Highlights

- 🎧 **Simultaneous S2TT** — emits target-language text while the speaker is still talking.
- 🌐 **Language pairs** — EN→JA, EN→RU, EN→DE, JA→EN, plus streaming ASR in English.
- ⚡ **Low latency** — fully causal Whisper encoder + depth decoder; CUDA-graph captured for fast autoregressive decoding.
- 🔄 **Single model, multiple tasks** — task and target language are selectable at runtime.
- 🖥️ **Browser-based demo** — WebRTC microphone input through a Gradio client.

## Architecture

Hikari is a Whisper-style encoder-decoder with two modifications:

- The encoder is made **fully causal**, so it can be unrolled over streaming audio chunks.
- A **depth decoder** is added on top of the Whisper decoder for causal alignment between audio frames and text tokens.

Training and the causal-alignment objective are described in the paper.

```
audio chunks ──▶ Causal Whisper Encoder ──▶ Whisper Decoder ──▶ Depth Decoder ──▶ streaming text
```

## Supported tasks

| Task          | Description                                                  |
| ------------- | ------------------------------------------------------------ |
| `transcribe`  | Simultaneous speech-to-text (English)                        |
| `translate`   | Simultaneous speech-to-text translation (EN→JA, EN→DE, EN→RU, JA→EN) |

## Usage

The model is intended to be served with the `hikari-server` / `hikari-client` tools from the [Hikari repository](https://github.com/sbintuitions/hikari).

```bash
# install
uv venv .venv --python=3.10 && source .venv/bin/activate
uv pip install torch==2.8.0 torchcodec==0.7.0 torchaudio==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu126
uv pip install "hikari @ git+https://github.com/sbintuitions/hikari"

# start the server (GPU machine) — the checkpoint is fetched from this repo
hikari-server --port 4440 --checkpoint sbintuitions/hikari-medium --device cuda:0

# start the client (local machine), then open http://localhost:5666
hikari-client --server-port 4440 --app-port 5666 --chunk-ms 80
```

See the [README on GitHub](https://github.com/sbintuitions/hikari) for SSH tunneling, development install, and the full configuration surface.

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.8.0 with CUDA (server)
- CUDA GPU (tested on A100, H100); the client runs on CPU

## Acknowledgements

Built on top of OpenAI's [Whisper](https://huggingface.co/openai/whisper-medium); the depth decoder reuses components from Kyutai's [Moshi](https://github.com/kyutai-labs/moshi).

## License

Released under the [MIT License](https://github.com/sbintuitions/hikari/blob/main/LICENSE).

## Citation

```bibtex
@misc{koshkin2026streamingtranslationtranscriptionspeechtotext,
      title={Streaming Translation and Transcription Through Speech-to-Text Causal Alignment},
      author={Roman Koshkin and Jeon Haesung and Lianbo Liu and Hao Shi and Mengjie Zhao and Yusuke Fujita and Yui Sudo},
      year={2026},
      eprint={2603.11578},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.11578}
}
```
