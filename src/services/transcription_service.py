
import logging
import os
import sys

# --- Backend Selection ---
HAS_MLX = False
HAS_FASTER_WHISPER = False

try:
    import mlx_whisper
    HAS_MLX = True
except ImportError:
    pass

try:
    from faster_whisper import WhisperModel
    HAS_FASTER_WHISPER = True
except ImportError:
    pass

logger = logging.getLogger("sentinel.transcription")

class TranscriptionService:
    def __init__(self, model_size="large-v3", device="cpu", compute_type="int8"):
        """
        Initialize the Transcription Service.
        Auto-detects the best backend: MLX (Mac) or Faster-Whisper (Linux/Windows).
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.backend = None

        if HAS_MLX:
            self.backend = "mlx"
            # Map generic sizes to specific MLX-optimized HF repositories
            self.repo_map = {
                "large-v3": "mlx-community/whisper-large-v3-mlx",
                "medium": "mlx-community/whisper-medium-mlx",
                "small": "mlx-community/whisper-small-mlx",
                "base": "mlx-community/whisper-base-mlx"
            }
            self.model_path = self.repo_map.get(model_size, "mlx-community/whisper-large-v3-mlx")
            logger.info(f"TranscriptionService initialized (Backend: MLX). Model: {self.model_path}")

        elif HAS_FASTER_WHISPER:
            self.backend = "faster_whisper"
            # Faster-whisper handles model downloading automatically given the size string
            # We use 'int8' for CPU performance on Linux/Cloud
            if device == "cpu":
                self.compute_type = "int8"
            logger.info(f"TranscriptionService initialized (Backend: Faster-Whisper). Model: {model_size}, Compute: {self.compute_type}")

        else:
            logger.warning("No valid transcription backend found (mlx-whisper or faster-whisper missing). Feature disabled.")

    def transcribe(self, audio_file_path: str, initial_prompt: str = None) -> str:
        """
        Transcribe audio using the available backend.
        """
        if not self.backend:
            msg = "Transcription unavailable: No compatible backend installed."
            logger.error(msg)
            return f"[Error: {msg}]"

        try:
            logger.info(f"Starting transcription for {audio_file_path} using {self.backend}...")

            # --- MLX Backend ---
            if self.backend == "mlx":
                result = mlx_whisper.transcribe(
                    audio_file_path,
                    path_or_hf_repo=self.model_path,
                    initial_prompt=initial_prompt,
                    verbose=False
                )
                text = result.get("text", "").strip()
                logger.info("MLX Transcription complete.")
                return text

            # --- Faster-Whisper Backend ---
            elif self.backend == "faster_whisper":
                # Initialize model on-demand to save memory when not in use
                model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)

                segments, info = model.transcribe(
                    audio_file_path,
                    beam_size=5,
                    initial_prompt=initial_prompt
                )

                # 'segments' is a generator, so we must iterate to get text
                text_segments = [segment.text for segment in segments]
                full_text = " ".join(text_segments).strip()

                logger.info("Faster-Whisper Transcription complete.")
                return full_text

        except Exception as e:
            logger.error(f"Transcription error ({self.backend}): {e}")
            raise e
