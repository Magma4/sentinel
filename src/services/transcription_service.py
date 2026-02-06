
import logging
import os
import sys

try:
    import mlx_whisper
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

logger = logging.getLogger("sentinel.transcription")

class TranscriptionService:
    def __init__(self, model_size="large-v3", device="cpu", compute_type="int8"):
        """
        Initialize the Transcription Service.
        """
        self.repo_map = {
            "large-v3": "mlx-community/whisper-large-v3-mlx",
            "medium": "mlx-community/whisper-medium-mlx",
            "small": "mlx-community/whisper-small-mlx",
            "base": "mlx-community/whisper-base-mlx"
        }
        self.model_path = self.repo_map.get(model_size, "mlx-community/whisper-large-v3-mlx")
        if HAS_MLX:
            logger.info(f"TranscriptionService initialized (MLX). Target Model: {self.model_path}")
        else:
            logger.warning("MLX Whisper not found. Transcription will not be available (Linux/Non-Mac Environment).")

    def transcribe(self, audio_file_path: str, initial_prompt: str = None) -> str:
        """
        Transcribe audio using Apple MLX (Hardware Accelerated).
        """
        if not HAS_MLX:
            msg = "Transcription unavailable: mlx-whisper is only supported on macOS Apple Silicon."
            logger.error(msg)
            return f"[Error: {msg}]"

        try:
            logger.info(f"Starting MLX transcription for {audio_file_path}...")
            result = mlx_whisper.transcribe(
                audio_file_path,
                path_or_hf_repo=self.model_path,
                initial_prompt=initial_prompt,
                verbose=False
            )
            text = result.get("text", "").strip()
            logger.info("MLX Transcription complete.")
            return text

        except Exception as e:
            logger.error(f"MLX Transcription error: {e}")
            raise e
