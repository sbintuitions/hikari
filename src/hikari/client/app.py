"""
Gradio WebRTC client for the Hikari S2S server.

Usage:
    hikari-client --server-port 4440 --app-port 5666 --chunk-ms 80
"""

import argparse
import asyncio
import debugpy
import base64
import json
import queue
import threading
import time
from typing import Literal

import fastrtc
import gradio as gr
import librosa
import numpy as np
import soundfile
import websockets
from termcolor import cprint

from hikari.client.vad import ConstantStreamHandler


class CommunicationManager:
    """Manages the WebSocket connection to the server."""

    def __init__(self, server_port=None):
        self.uri = f"ws://localhost:{server_port}"
        self.incoming_queue = queue.Queue()
        self.outgoing_queue = queue.Queue()
        self.websocket = None
        self.stop_event = threading.Event()

    def _parse_data(self, data) -> tuple[np.ndarray | None, str | None, float | None, dict]:
        try:
            message = json.loads(data)
            text = message.get("text", "")
            audio_b64 = message.get("audio", "")
            timestamp = message.get("timestamp")
            incoming_telemetry = message.get("telemetry", {})

            if not audio_b64:
                return None, text, timestamp, incoming_telemetry

            audio_bytes = base64.b64decode(audio_b64)
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            return audio_array, text, timestamp, incoming_telemetry
        except (json.JSONDecodeError, TypeError, ValueError):
            return None, None, None, None

    @staticmethod
    def int16tofloat32(audio: np.ndarray) -> np.ndarray:
        return audio.astype(np.float32) / 32768.0

    @staticmethod
    def float32toint16(audio: np.ndarray) -> np.ndarray:
        return (audio * 32768.0).astype(np.int16)

    def _format_data(self, audio, text):
        if audio is not None:
            audio_bytes = audio.tobytes()
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        else:
            audio_b64 = None
        return json.dumps({"text": text, "audio": audio_b64, "timestamp": time.time()})

    async def _receive_loop(self):
        try:
            async for message in self.websocket:
                audio, text, original_timestamp, incoming_telemetry = self._parse_data(message)
                if audio is not None and original_timestamp is not None:
                    rtt_ms = (time.time() - original_timestamp) * 1000
                    telemetry = {
                        "rtt_ms": rtt_ms,
                        "server_incoming_queue": incoming_telemetry.get("queue_in", 0),
                        "server_outgoing_queue": incoming_telemetry.get("queue_out", 0),
                        "WP": incoming_telemetry.get("WP", "N/A"),
                        "speech_prob": incoming_telemetry.get("speech_prob", "N/A"),
                        "tokens_in_context": incoming_telemetry.get("tokens_in_context", "N/A"),
                    }
                    self.incoming_queue.put((audio, text, telemetry))
        except websockets.exceptions.ConnectionClosed:
            print("CLIENT: Server connection closed.")
        finally:
            self.stop_event.set()

    async def _send_loop(self):
        while not self.stop_event.is_set():
            try:
                audio, text = self.outgoing_queue.get_nowait()
                data_to_send = self._format_data(audio, text)
                await self.websocket.send(data_to_send)
            except queue.Empty:
                await asyncio.sleep(0.002)
            except websockets.exceptions.ConnectionClosed:
                break

    async def _connect_and_manage(self):
        try:
            async with websockets.connect(self.uri) as websocket:
                print("CLIENT: Connected to server.")
                self.websocket = websocket
                self.stop_event.clear()

                receive_task = asyncio.create_task(self._receive_loop())
                send_task = asyncio.create_task(self._send_loop())

                done, pending = await asyncio.wait(
                    [receive_task, send_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
        except (OSError, websockets.exceptions.ConnectionClosedError) as e:
            print(f"CLIENT: Connection failed: {e}")
        finally:
            self.stop_event.set()
            print("CLIENT: Disconnected.")

    def _run_client(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._connect_and_manage())

    def send_control_message(self, settings: dict):
        message = json.dumps(
            {
                "control": {
                    "task": settings.get("task", "translate"),
                    "tgt_lang": settings.get("tgt_lang", "English"),
                    "baseline_wait_penalty": float(settings.get("baseline_wait_penalty", 1.0)),
                    "wait_penalty_boost": float(settings.get("wait_penalty_boost", 0.6)),
                    "wait_penalty_decay": float(settings.get("wait_penalty_decay", 0.3)),
                    "decoder_context": settings.get("decoder_context", "337"),
                    "repetition_penalty": float(settings.get("repetition_penalty", 40.0)),
                },
                "timestamp": time.time(),
            }
        )

        self.outgoing_queue.put((None, message))

    def start(self):
        self.thread = threading.Thread(target=self._run_client, daemon=True)
        self.thread.start()
        print("CLIENT: Communication manager started.")


def load_and_resample(path, target_sr=16000, mono=True):
    """Load an audio file and resample to the target sample rate."""
    audio, sr = soundfile.read(path, dtype="float32")
    if mono and audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr, res_type="soxr_hq")
    return audio, target_sr


def make_audio(sr=16000, duration=0.06, path=None):
    if path is None:
        raise ValueError("An audio file path must be provided.")

    audio, sr = load_and_resample(path, target_sr=16000)

    num_samples = int(sr * duration)
    for i in range(0, len(audio), num_samples):
        chunk = audio[i : i + num_samples].astype(np.float32)
        if len(chunk) < num_samples:
            chunk = np.pad(chunk, (0, num_samples - len(chunk)), mode="constant")
        yield chunk


def run_app(CHUNK_MS, server_port, app_port):
    print("Starting WebRTC client")

    concurrency_limit = 1
    manager = CommunicationManager(server_port=server_port)
    manager.start()
    time.sleep(1)

    shared_state = {"OUTPUT_TEXT": ""}

    def stream_file_to_response(file):
        audio, sr = soundfile.read(file.name, dtype="float32")
        shared_state["OUTPUT_TEXT"] = ""
        shared_state["audio_generator"] = make_audio(path=file.name, duration=CHUNK_MS / 1000)
        return

    def additional_outputs(OUTPUT_TEXT, status_text, collected_audio, telemetry):
        return fastrtc.AdditionalOutputs(
            OUTPUT_TEXT,
            status_text,
            collected_audio,
            f"{telemetry['rtt_ms']:.2f}",
            telemetry["client_incoming_queue"],
            telemetry["client_outgoing_queue"],
            telemetry["server_incoming_queue"],
            telemetry["server_outgoing_queue"],
            f"{telemetry['WP']:.2f}",
            f"{telemetry['speech_prob']:.2f}",
            telemetry.get("tokens_in_context", "N/A"),
        )

    def response(
        input_audio: tuple[int, np.ndarray],
        webrtc_id: str,
        task: str | None,
        tgt_lang: str | None,
        baseline_wait_penalty: str | None,
        wait_penalty_boost: str | None,
        wait_penalty_decay: str | None,
        repetition_penalty: str | None,
    ):
        if "audio_generator" not in shared_state:
            sr, audio_data = input_audio
        else:
            audio_data = (next(shared_state["audio_generator"]) * 32768.0).astype(np.int16).reshape(1, -1)

        manager.outgoing_queue.put((audio_data, "dummy text"))
        cprint(
            f"CLIENT: Sent chunk. Incoming queue size: {manager.incoming_queue.qsize()} | Outgoing queue size: {manager.outgoing_queue.qsize()}",
            color="blue",
        )

        while not manager.incoming_queue.empty():
            try:
                processed_audio, processed_text, telemetry = manager.incoming_queue.get_nowait()
                telemetry["client_incoming_queue"] = manager.incoming_queue.qsize()
                telemetry["client_outgoing_queue"] = manager.outgoing_queue.qsize()

                if processed_text is not None:
                    shared_state["OUTPUT_TEXT"] += processed_text
                cprint(
                    f"CLIENT: Received processed data! Text: '{processed_text}', Audio shape: {processed_audio.shape} | RTT: {telemetry['rtt_ms']:.2f} ms\n",
                    color="blue",
                )

                yield (
                    (16000, processed_audio),
                    additional_outputs(shared_state["OUTPUT_TEXT"], "Connected", None, telemetry),
                )

            except queue.Empty:
                yield (
                    None,
                    additional_outputs("", "Error", None, telemetry),
                )

    title = "Hikari S2T (WebRTC)"

    with gr.Blocks(title=title) as demo:
        title_markdown = gr.Markdown(f"# {title}")
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            chat = fastrtc.WebRTC(
                                label="WebRTC Chat",
                                modality="audio",
                                mode="send-receive",
                                full_screen=False,
                            )
                            file_input = gr.File(
                                label="Upload audio file", file_types=[".wav", ".mp3"], type="filepath"
                            )
                output_text = gr.Textbox(label="Output", lines=3, interactive=False)

                with gr.Accordion("Advanced", open=True):
                    collected_audio = gr.Audio(
                        label="Full Audio",
                        type="numpy",
                        format="wav",
                        interactive=False,
                        autoplay=True,
                    )

                with gr.Accordion("Settings Help"):
                    gr.Markdown(
                        "- **Task**: Choose between `transcribe` (speech-to-text) or `translate` (speech-to-text translation).\n"
                        "- **Target Language**: Select the language you want the output in.\n"
                        "- **Baseline wait penalty**: Adjusts latency vs. accuracy trade-off (higher values may increase delay but improve translation quality).\n"
                        "- To apply new settings, end the current conversation and start a new one."
                    )

            with gr.Column():
                with gr.Row():
                    status_text = gr.Textbox(label="Status", lines=1, interactive=False)
                    rtt_ms = gr.Textbox(label="RTT (ms)", lines=1, interactive=False)
                    WP = gr.Textbox(label="WP", lines=1, interactive=False)
                    speech_prob = gr.Textbox(label="Speech Prob", lines=1, interactive=False)
                    client_incoming_queue = gr.Textbox(label="Client in_q", lines=1, interactive=False)
                    client_outgoing_queue = gr.Textbox(label="Client out_q", lines=1, interactive=False)
                    server_incoming_queue = gr.Textbox(label="Server in_q", lines=1, interactive=False)
                    server_outgoing_queue = gr.Textbox(label="Server out_q", lines=1, interactive=False)
                    tokens_in_context = gr.Textbox(label="tokens_in_ctx", lines=1, interactive=False)

                preset_task_dropdown = gr.Dropdown(
                    label="Task",
                    choices=["transcribe", "translate"],
                    value="translate",
                )
                preset_tgt_lang_dropdown = gr.Dropdown(
                    label="Target Language",
                    choices=["English", "German", "Japanese", "Russian"],
                    value="Japanese",
                )
                preset_baseline_wait_penalty_dropdown = gr.Dropdown(
                    label="Baseline wait penalty",
                    choices=[0.0, 0.3, 0.6, 1.0, 1.3, 1.7, 2.0],
                    value=0.0,
                )
                preset_wpb_dropdown = gr.Dropdown(
                    label="Wait penalty boost",
                    choices=[0.0, 0.3, 0.6, 1.0, 1.1, 1.2, 1.3, 1.4, 1.6, 1.8],
                    value=0.6,
                )
                preset_wpd_dropdown = gr.Dropdown(
                    label="Wait penalty decay",
                    choices=[0.0, 0.2, 0.3, 0.4, 0.6, 0.9],
                    value=0.3,
                )
                preset_repetition_penalty_dropdown = gr.Dropdown(
                    label="Repetition penalty",
                    choices=[0, 20, 40],
                    value=40,
                )
                decoder_context_dropdown = gr.Dropdown(
                    label="Decoder context length",
                    choices=[100, 200, 300, 337, "max"],
                    value=337,
                )

                apply_button = gr.Button("Reset / Apply Settings")

                def apply_settings(
                    task,
                    tgt_lang,
                    baseline_wait_penalty,
                    wait_penalty_boost,
                    decoder_context,
                    wait_penalty_decay,
                    repetition_penalty,
                ):
                    shared_state["OUTPUT_TEXT"] = ""
                    _ = shared_state.pop("audio_generator", None)
                    manager.send_control_message(
                        dict(
                            task=task,
                            tgt_lang=tgt_lang,
                            baseline_wait_penalty=baseline_wait_penalty,
                            wait_penalty_boost=wait_penalty_boost,
                            wait_penalty_decay=wait_penalty_decay,
                            decoder_context=decoder_context,
                            repetition_penalty=repetition_penalty,
                        )
                    )
                    return ["Applied", ""]

                apply_button.click(
                    fn=apply_settings,
                    inputs=[
                        preset_task_dropdown,
                        preset_tgt_lang_dropdown,
                        preset_baseline_wait_penalty_dropdown,
                        preset_wpb_dropdown,
                        decoder_context_dropdown,
                        preset_wpd_dropdown,
                        preset_repetition_penalty_dropdown,
                    ],
                    outputs=[status_text, output_text],
                )

        file_input.upload(
            fn=stream_file_to_response,
            inputs=[file_input],
        )
        chat.stream(
            ConstantStreamHandler(
                response, chunk_ms=CHUNK_MS
            ),
            inputs=[
                chat,
                preset_task_dropdown,
                preset_tgt_lang_dropdown,
                preset_baseline_wait_penalty_dropdown,
                preset_wpb_dropdown,
                preset_wpd_dropdown,
                preset_repetition_penalty_dropdown,
            ],
            concurrency_limit=concurrency_limit,
            outputs=[chat],
        )
        chat.on_additional_outputs(
            lambda *args: args,
            outputs=[
                output_text,
                status_text,
                collected_audio,
                rtt_ms,
                client_incoming_queue,
                client_outgoing_queue,
                server_incoming_queue,
                server_outgoing_queue,
                WP,
                speech_prob,
                tokens_in_context,
            ],
            concurrency_limit=concurrency_limit,
            show_progress="hidden",
        )

    demo.launch(server_name="localhost", server_port=app_port, show_api=False)


def main():
    parser = argparse.ArgumentParser(description="Hikari S2T WebRTC client")
    parser.add_argument("--chunk-ms", type=int, default=80, help="Chunk size in milliseconds")
    parser.add_argument("--server-port", type=int, required=True, help="WebSocket server port")
    parser.add_argument("--app-port", type=int, default=5666, help="Frontend app port")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.debug:
        debugpy.listen(5678)
        print("Waiting for debugger to attach...")
        debugpy.wait_for_client() # blocks until attached
    cprint(f"CHUNK_MS: {args.chunk_ms}", color="red")
    run_app(CHUNK_MS=args.chunk_ms, server_port=args.server_port, app_port=args.app_port)


if __name__ == "__main__":
    main()
