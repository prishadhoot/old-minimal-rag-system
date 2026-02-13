"""
LLM Module

Responsibilities:
- Accept context + question
- Call OpenRouter (Gemma model via OpenRouter provider)
- Enforce NOT_FOUND protocol
- Return string answer

Constraints:
- No retrieval
- No formatting of Chunk objects
- Retries on 429 (rate limit) with fallback models
- If API fails after retries → raise RuntimeError

Uses OpenRouter as the provider - Gemma model only. Not OpenAI.
"""

import time
import httpx

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
RETRY_DELAY_SEC = 2
MAX_RETRIES = 3


class LLMClient:
    """
    LLM client for OpenRouter provider (Gemma model).
    
    Uses OpenRouter API directly - not OpenAI.
    
    Constraints:
    - No retrieval
    - No formatting of Chunk objects
    - No retries
    - Prompt template internal to this class
    """
    
    PROMPT_TEMPLATE = """You are a factual assistant.
Answer using ONLY the provided context.
If the answer is not found in the context, respond with exactly: NOT_FOUND

Context:
{context}

Question: {question}

Answer:"""
    
    def __init__(self, api_key: str, model: str = "google/gemma-3-27b-it:free", fallback_models: list = None):
        """
        Initialize LLM client.
        
        Args:
            api_key: OpenRouter API key
            model: OpenRouter model ID (e.g. google/gemma-3-27b-it:free)
            fallback_models: Optional list of model IDs to try when primary returns 429
            
        Raises:
            RuntimeError: If API key is missing or invalid
        """
        if not api_key:
            raise RuntimeError("OpenRouter API key is required")
        
        self.model = model
        self.fallback_models = fallback_models or []
        self.api_key = api_key.strip()
    
    def _call_api(self, model: str, prompt: str) -> str:
        """Make a single API call. Raises on error."""
        with httpx.Client() as client:
            response = client.post(
                OPENROUTER_API_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return content.strip() if content else ""
    
    def generate(self, context: str, question: str) -> str:
        """
        Generate answer using OpenRouter (Gemma).
        
        Args:
            context: Context string (formatted externally)
            question: User question
            
        Returns:
            String response from LLM
            
        Raises:
            RuntimeError: If API call fails after retries
        """
        prompt = self.PROMPT_TEMPLATE.format(
            context=context,
            question=question
        )
        
        models_to_try = [self.model] + [m for m in self.fallback_models if m != self.model]
        last_error = None
        
        for attempt, model in enumerate(models_to_try):
            try:
                return self._call_api(model, prompt)
            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code == 429:
                    if attempt < len(models_to_try) - 1:
                        time.sleep(RETRY_DELAY_SEC * (attempt + 1))
                        continue
                    raise RuntimeError(
                        "OpenRouter rate limit (429). Free tier is limited. "
                        "Retry in a few minutes, or add your own API key at "
                        "https://openrouter.ai/settings/integrations for higher limits."
                    )
                raise RuntimeError(f"OpenRouter API call failed: {e.response.text}")
            except Exception as e:
                last_error = e
                if attempt < len(models_to_try) - 1:
                    time.sleep(RETRY_DELAY_SEC * (attempt + 1))
                    continue
                raise RuntimeError(f"OpenRouter API call failed: {e}")
        
        raise RuntimeError(f"OpenRouter API call failed: {last_error}")
