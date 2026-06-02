# Copyright (c) 2026, SB Intuitions Corp
#
# Unless otherwise stated, this project is licensed under the MIT License.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""
WebSocket server for streaming speech-to-text translation.

Usage:
    hikari-server --port 4440 --checkpoint /path/to/checkpoint --device cuda:0
"""

import argparse
import asyncio
import base64
import json
import logging
import queue
import threading
import time

import debugpy
import librosa
import numpy as np
import websockets
from termcolor import cprint

from hikari.utils import Timer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


class Server:
    def __init__(self, host="0.0.0.0", port=None, checkpoint=None, device="cuda:0", debug=False):
        self.host = host
        self.port = port
        self.debug = debug
        self.incoming_queue = queue.Queue()
        self.outgoing_queue = queue.Queue()
        self.websocket = None
        self.stop_event = threading.Event()
        from hikari.server.model_wrapper import ModelWrapper

        self.model_wrapper = ModelWrapper(
            checkpoint=checkpoint, device=device, debug=debug
        )
        self.timer = Timer(logger=logger)

    def clear_all_queues(self):
        while not self.incoming_queue.empty():
            try:
                self.incoming_queue.get_nowait()
            except queue.Empty:
                break
        while not self.outgoing_queue.empty():
            try:
                self.outgoing_queue.get_nowait()
            except queue.Empty:
                break

    def _parse_data(self, data):
        try:
            message = json.loads(data)
            text = message.get("text", None)
            audio_b64 = message.get("audio", None)
            timestamp = message.get("timestamp")

            try:
                parsed_text = json.loads(text)
                if "control" in parsed_text:
                    control = parsed_text["control"]
                else:
                    control = None
            except json.JSONDecodeError:
                control = None

            if not audio_b64:
                return None, text, timestamp, control

            audio_bytes = base64.b64decode(audio_b64)
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            return audio_array, text, timestamp, control
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            print(f"Error parsing data: {e}")
            return None, None, None, None

    def _format_data(self, audio, text, timestamp, telemetry=None):
        audio_bytes = audio.tobytes()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        if telemetry:
            return json.dumps({"text": text, "audio": audio_b64, "timestamp": timestamp, "telemetry": telemetry})
        return json.dumps({"text": text, "audio": audio_b64, "timestamp": timestamp, "telemetry": None})

    @staticmethod
    def int16tofloat32(audio: np.ndarray) -> np.ndarray:
        return audio.astype(np.float32) / 32768.0

    @staticmethod
    def float32toint16(audio: np.ndarray) -> np.ndarray:
        return (audio * 32768.0).astype(np.int16)

    def _model(self, chunk):
        cprint(
            f"SERVER: Processing. Incoming queue size: {self.incoming_queue.qsize()} | Outgoing queue size: {self.outgoing_queue.qsize()}",
            color="blue",
        )
        audio, text = chunk
        print(f"SERVER: Processing chunk... Text: '{text}', Audio shape: {audio.shape} {audio.dtype}")

        word, dec_pred = self.model_wrapper.get_one_token(self.int16tofloat32(audio))
        if dec_pred is not None:
            dec_pred = librosa.resample(
                dec_pred.astype(np.float32),
                orig_sr=24000,
                target_sr=16000,
                res_type="soxr_hq",
            )
            dec_pred = np.clip(dec_pred, -1.0, 1.0)
            audio = self.float32toint16(0.75 * dec_pred + 0.25 * self.int16tofloat32(audio))

        processed_text = word
        processed_audio = audio

        return processed_text, processed_audio

    def _processing_loop(self):
        print("SERVER: Processing thread started.")
        while not self.stop_event.is_set():
            try:
                audio, text, timestamp = self.incoming_queue.get(timeout=0.1)
                processed_text, processed_audio = self._model((audio, text))
                self.outgoing_queue.put((processed_audio, processed_text, timestamp))
            except queue.Empty:
                continue
        print("SERVER: Processing thread stopped.")

    async def _receive_loop(self):
        try:
            async for message in self.websocket:
                audio, text, timestamp, control = self._parse_data(message)
                if control:
                    cprint(f"SERVER: Received control message: {control}", color="green")
                    self.model_wrapper.apply_settings(control, debug=self.debug)
                    self.clear_all_queues()
                else:
                    if audio is not None:
                        self.incoming_queue.put((audio, text, timestamp))

        except websockets.exceptions.ConnectionClosed:
            print("SERVER: Client connection closed.")
        finally:
            self.stop_event.set()

    async def _send_loop(self):
        while not self.stop_event.is_set():
            try:
                processed_audio, processed_text, timestamp = self.outgoing_queue.get_nowait()
                telemetry = {
                    "queue_in": self.incoming_queue.qsize(),
                    "queue_out": self.outgoing_queue.qsize(),
                    "WP": self.model_wrapper.wait_penalty,
                    "speech_prob": self.model_wrapper.speech_prob,
                    "tokens_in_context": len(self.model_wrapper.decoder_input_ids_lst),
                }
                data_to_send = self._format_data(processed_audio, processed_text, timestamp, telemetry)
                await self.websocket.send(data_to_send)
            except queue.Empty:
                await asyncio.sleep(0.001)
            except websockets.exceptions.ConnectionClosed:
                break

    async def _connection_handler(self, websocket):
        print(f"SERVER: Client connected from {websocket.remote_address}")
        self.websocket = websocket
        self.stop_event.clear()

        processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        processing_thread.start()

        receive_task = asyncio.create_task(self._receive_loop())
        send_task = asyncio.create_task(self._send_loop())

        done, pending = await asyncio.wait(
            [receive_task, send_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()

        processing_thread.join()
        print("SERVER: Connection handler finished.")

    def start(self):
        print(f"SERVER: Starting WebSocket server on ws://{self.host}:{self.port}")
        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.server_thread.start()

    def _run_server(self):
        async def main():
            async with websockets.serve(
                self._connection_handler,
                self.host,
                self.port,
                ping_interval=200,
                ping_timeout=400,
            ):
                await asyncio.Future()

        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print("SERVER: Shutting down.")


def main():
    parser = argparse.ArgumentParser(description="Hikari S2T WebSocket server")
    parser.add_argument("--port", default=4440, type=int, help="WebSocket server port")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to model checkpoint")
    parser.add_argument("--device", default="cuda:0", type=str, help="CUDA device")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.debug:
        debugpy.listen(5679)
        print("Waiting for debugger to attach...")
        debugpy.wait_for_client()  # blocks until attached

    server = Server(
        port=args.port,
        checkpoint=args.checkpoint,
        device=args.device,
        # override_mode=args.override_mode,
        debug=args.debug,
    )

    server.start()
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
