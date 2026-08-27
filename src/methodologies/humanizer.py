"""v1.0 multi-method dispatcher (reference).

The optional detection and API dependencies are imported only when their features
are used. The recommended v1.5 Standard Pipeline therefore remains lightweight.
"""

import importlib
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass

import click
import toml

logger = logging.getLogger("humanize_text")

# Guardrails for the HTTP API.
MAX_INPUT_CHARS = 20_000
REQUEST_TIMEOUT_SECONDS = 180


@dataclass
class HumanizeResult:
    text: str
    method_used: str
    processing_time: float


class Humanizer:
    METHODS = {
        "translation_chain": ("src.methodologies.translation_chain", "TranslationChainProcessor"),
        "llm_rewrite": ("src.methodologies.llm_rewriter", "LLMRewriteProcessor"),
        "detection_guided": ("src.methodologies.detection_pipeline", "DetectionGuidedProcessor"),
        "mixed_engine": ("src.methodologies.mixed_engine", "MixedEngineProcessor"),
    }

    def __init__(self, config_path: str = "config/config.toml"):
        with open(config_path, "r", encoding="utf-8") as config_file:
            self.config = toml.load(config_file)

    @classmethod
    def _processor_class(cls, method: str):
        module_name, class_name = cls.METHODS[method]
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    def process(self, text: str, method: str = None, **kwargs) -> HumanizeResult:
        method = method or self.config["general"]["default_method"]
        if method not in self.METHODS:
            raise ValueError(f"Unknown method: {method}. Available: {list(self.METHODS.keys())}")

        processor = self._processor_class(method)(self.config)
        start = time.time()
        result_text = processor.process(text, **kwargs)
        return HumanizeResult(
            text=result_text,
            method_used=method,
            processing_time=time.time() - start,
        )


def create_api_app():
    """Create the optional FastAPI application.

    Install ``fastapi``, ``pydantic`` and ``uvicorn`` before using ``--serve``.
    They are intentionally not required by the standard CLI pipeline.
    """
    try:
        import httpx
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import HTMLResponse, JSONResponse
        from pydantic import BaseModel, field_validator
    except ImportError as exc:
        raise RuntimeError(
            "API dependencies are missing. Install fastapi, pydantic and uvicorn to use --serve."
        ) from exc

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    api = FastAPI(title="Humanize Text API")

    # --- Cached, hot-reloadable config + Humanizer -----------------------------
    # Reloading the TOML on every request is wasteful; reload only when the file
    # actually changes on disk (so edits to config.toml are still picked up).
    _cache: dict = {"path": None, "mtime": None, "humanizer": None}

    def get_humanizer() -> "Humanizer":
        config_path = os.environ.get("CONFIG_PATH", "config/config.toml")
        try:
            mtime = os.path.getmtime(config_path)
        except OSError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Config file not found or unreadable: {config_path}",
            ) from exc
        if _cache["humanizer"] is None or _cache["path"] != config_path or _cache["mtime"] != mtime:
            try:
                _cache["humanizer"] = Humanizer(config_path=config_path)
            except Exception as exc:  # malformed TOML, etc.
                logger.exception("Failed to load config")
                raise HTTPException(status_code=500, detail=f"Invalid config: {exc}") from exc
            _cache["path"] = config_path
            _cache["mtime"] = mtime
        return _cache["humanizer"]

    _executor = ThreadPoolExecutor(max_workers=4)

    @api.get("/", response_class=HTMLResponse)
    def index():
        template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
        try:
            with open(template_path, "r", encoding="utf-8") as page:
                return page.read()
        except OSError:
            return HTMLResponse("<h1>Humanize Text</h1><p>UI template missing.</p>", status_code=200)

    class HumanizeRequest(BaseModel):
        text: str
        method: str = "llm_rewrite"
        language: str = "en"
        tier: str = "standard"

        @field_validator("text")
        @classmethod
        def _text_ok(cls, v: str) -> str:
            v = (v or "").strip()
            if not v:
                raise ValueError("Text must not be empty.")
            if len(v) > MAX_INPUT_CHARS:
                raise ValueError(f"Text too long ({len(v)} chars); max is {MAX_INPUT_CHARS}.")
            return v

    @api.post("/humanize")
    def api_humanize(request: HumanizeRequest):
        if request.method not in Humanizer.METHODS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown method '{request.method}'. Available: {list(Humanizer.METHODS)}",
            )

        humanizer = get_humanizer()

        def _run():
            return humanizer.process(request.text, method=request.method, tier=request.tier)

        try:
            result = _executor.submit(_run).result(timeout=REQUEST_TIMEOUT_SECONDS)
        except FutureTimeout:
            logger.warning("Request timed out after %ss (method=%s)", REQUEST_TIMEOUT_SECONDS, request.method)
            raise HTTPException(
                status_code=504,
                detail=f"Processing timed out after {REQUEST_TIMEOUT_SECONDS}s. Try shorter text.",
            )
        except (ImportError, ModuleNotFoundError) as exc:
            logger.warning("Method '%s' unavailable: %s", request.method, exc)
            raise HTTPException(
                status_code=501,
                detail=f"Method '{request.method}' needs extra dependencies that aren't installed.",
            ) from exc
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            hint = "check the API key" if code in (401, 403) else "rate limited, retry shortly" if code == 429 else "upstream error"
            logger.warning("LLM provider returned %s: %s", code, hint)
            raise HTTPException(status_code=502, detail=f"LLM provider returned {code} ({hint}).") from exc
        except (httpx.TimeoutException, httpx.RequestError) as exc:
            logger.warning("Upstream network error: %s", exc)
            raise HTTPException(status_code=502, detail="Could not reach the upstream service (network error).") from exc
        except ValueError as exc:
            # config/provider/api-key validation, unknown-language, etc.
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Unexpected error during humanize")
            raise HTTPException(status_code=500, detail=f"Processing failed: {exc}") from exc

        if not result.text or not result.text.strip():
            raise HTTPException(status_code=502, detail="Upstream returned an empty result. Please retry.")

        return {
            "result": result.text,
            "method": result.method_used,
            "processing_time_ms": int(result.processing_time * 1000),
        }

    class DetectRequest(BaseModel):
        text: str

        @field_validator("text")
        @classmethod
        def _text_ok(cls, v: str) -> str:
            v = (v or "").strip()
            if not v:
                raise ValueError("Text must not be empty.")
            if len(v) > MAX_INPUT_CHARS:
                raise ValueError(f"Text too long ({len(v)} chars); max is {MAX_INPUT_CHARS}.")
            return v

    @api.post("/detect")
    def api_detect(request: DetectRequest):
        from .detectors.heuristic import analyze

        try:
            result = analyze(request.text)
        except Exception as exc:
            logger.exception("Detector failed")
            raise HTTPException(status_code=500, detail=f"Detection failed: {exc}") from exc

        result["note"] = (
            "Ensemble stylometric heuristic (dependency-free). Not an ML classifier — "
            "use as a signal, not proof of authorship."
        )
        return result

    @api.get("/methods")
    def api_methods():
        return {"methods": list(Humanizer.METHODS.keys())}

    @api.get("/health")
    def api_health():
        config_path = os.environ.get("CONFIG_PATH", "config/config.toml")
        return {
            "status": "ok",
            "config_present": os.path.isfile(config_path),
            "methods": list(Humanizer.METHODS.keys()),
        }

    return api


@click.command()
@click.option("--input", "input_text", required=True, help="Input text or path to text file")
@click.option("--method", default=None, help="Humanization method")
@click.option("--output", default=None, help="Output file path")
@click.option("--config", default="config/config.toml", help="Config file path")
@click.option("--tier", default="standard", help="Processing tier")
@click.option("--language", default="en", help="Target language code")
@click.option("--serve", is_flag=True, help="Start API server instead of CLI processing")
def main(input_text, method, output, config, tier, language, serve):
    if serve:
        try:
            import uvicorn
        except ImportError as exc:
            raise click.ClickException(
                "API dependencies are missing. Install fastapi, pydantic and uvicorn."
            ) from exc
        uvicorn.run(create_api_app(), host="0.0.0.0", port=8000)
        return

    import os

    if os.path.isfile(input_text):
        with open(input_text, "r", encoding="utf-8") as input_file:
            input_text = input_file.read()

    result = Humanizer(config_path=config).process(input_text, method=method, tier=tier)
    if output:
        with open(output, "w", encoding="utf-8") as output_file:
            output_file.write(result.text)
        click.echo(f"Written to {output} ({result.processing_time:.1f}s)")
    else:
        click.echo(result.text)


if __name__ == "__main__":
    main()
