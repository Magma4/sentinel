
import logging
import mlx_whisper
import os

logger = logging.getLogger("sentinel.transcription")

class TranscriptionService:
    def __init__(self, model_size="large-v3", device="cpu", compute_type="int8"):
        """
        Initialize the Transcription Service.
        For MLX, we don't strictly preload the model object in __init__ like with faster-whisper,
        as the library handles caching.

        Args:
            model_size: 'large-v3', 'medium', etc. Mapped to mlx-community repos.
        """
        # Map generic sizes to specific MLX-optimized HF repositories
        # 4-bit quantized models are the default recommendation for MLX
        self.repo_map = {
            "large-v3": "mlx-community/whisper-large-v3-mlx",
            "medium": "mlx-community/whisper-medium-mlx",
            "small": "mlx-community/whisper-small-mlx",
            "base": "mlx-community/whisper-base-mlx"
        }

        self.model_path = self.repo_map.get(model_size, "mlx-community/whisper-large-v3-mlx")
        logger.info(f"TranscriptionService initialized. Target Model: {self.model_path}")

    def transcribe(self, audio_file_path: str, initial_prompt: str = None) -> str:
        """
        Transcribe audio using Apple MLX (Hardware Accelerated).
        """
        try:
            logger.info(f"Starting MLX transcription for {audio_file_path}...")

            # mlx_whisper.transcribe returns a dict: {'text': '...', 'segments': [...]}
            # decode_options can include initial_prompt (Note: key might vary, usually 'initial_prompt' works directly or in options)
            # Checking mlx-whisper signature, it accepts **decode_options.

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
