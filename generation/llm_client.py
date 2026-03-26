"""
Hugging Face LLM client using the official InferenceClient.
This replaces the manual HTTP router calls.
"""

from huggingface_hub import InferenceClient
from config.settings import HF_API_TOKEN, HF_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE
from monitoring.logger import get_logger

logger = get_logger(__name__)

client = InferenceClient(
    model=HF_MODEL,
    token=HF_API_TOKEN
)


def generate(prompt: str, max_tokens: int = 150, temperature: float = 0.2) -> str:
    """
    Generate text from Hugging Face hosted model using chat_completion for conversational task.
    """

    try:

        logger.info(f"Sending prompt to Hugging Face model: {HF_MODEL}")

        # DEBUG PRINTS
        print("Sending prompt to model:", HF_MODEL)
        print("Prompt length:", len(prompt))
        print("Prompt preview:", prompt[:500])  # optional but very useful

        messages = [
            {
                "role": "system",
                "content": "You are a helpful document Q&A assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        response = client.chat_completion(
            model=HF_MODEL,
            messages=messages,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE
        )

        return response.choices[0].message.content.strip()

    # Token / authentication issues
    except PermissionError as exc:

        logger.error(f"Hugging Face authentication error: {exc}")

        raise RuntimeError(
            "Invalid Hugging Face API token. Please check HF_API_TOKEN."
        )

    # Model loading or unavailable
    except RuntimeError as exc:

        logger.error(f"Hugging Face runtime error: {exc}")

        raise RuntimeError(
            f"Model '{HF_MODEL}' may be unavailable or still loading."
        )

    # Network issues
    except ConnectionError as exc:

        logger.error(f"Hugging Face connection error: {exc}")

        raise RuntimeError(
            "Cannot connect to Hugging Face servers. Check internet connection."
        )

    # Catch-all fallback
    except Exception as exc:

        logger.error(f"Hugging Face generation failed: {str(exc)}")

        raise RuntimeError(f"LLM generation failed: {str(exc)}")


def is_hf_available() -> bool:
    """
    Check if Hugging Face API is reachable.
    """

    try:
        # Use chat_completion instead of text_generation
        messages = [{"role": "user", "content": "Hello"}]
        client.chat_completion(
            model=HF_MODEL,
            messages=messages,
            max_tokens=5
        )
        return True

    except Exception as exc:

        logger.error(f"HuggingFace availability check failed: {exc}")

        return False