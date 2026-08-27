# Humanize Text

A self-hosted web app that rewrites AI-generated text into a more natural,
human-sounding voice — and estimates how "AI-like" a passage reads.

Built with FastAPI, packaged with Docker, and served through a single-page
web UI. No external frontend build step; open the URL and use it.

---

## Features

- **Humanize** — rewrite text with three selectable methods:
  - `llm_rewrite` — LLM rewrite (DeepSeek by default) that varies sentence
    rhythm and vocabulary. Best quality.
  - `translation_chain` — round-trip translation (EN → ZH → JA → EN) using
    free engines. No API key required.
  - `mixed_engine` — multi-engine translation blend.
- **Detect AI** — a dependency-free ensemble stylometric detector that scores
  how likely a passage is AI-generated, with a confidence level and a
  per-signal breakdown.
- **Web UI** — dark, animated interface with Humanize / Detect modes, live
  word count, copy-to-clipboard, and keyboard shortcut (⌘/Ctrl + Enter).
- **Hardened API** — input validation, request timeouts, and graceful error
  mapping (no raw stack traces).
- **Dockerized** — one command to build and run, with a container healthcheck.

---

## Quick start

```bash
# 1. Configure (copy the example and add your keys)
cp config/config.example.toml config/config.toml
#   set api_keys.deepseek_api_key = "sk-..."   (only needed for llm_rewrite)

# 2. Build and run
docker compose up -d --build

# 3. Open the app
#   http://localhost:8000
```

> `config/config.toml` is git-ignored — your API keys never get committed.

To pick up config changes, run `docker compose restart` (the config folder is
volume-mounted, so no rebuild is needed).

---

## API

Base URL: `http://localhost:8000`

| Method | Path        | Description                                  |
|--------|-------------|----------------------------------------------|
| GET    | `/`         | Web UI                                        |
| POST   | `/humanize` | Rewrite text (see body below)                 |
| POST   | `/detect`   | Score AI-likelihood of text                   |
| GET    | `/methods`  | List available humanize methods               |
| GET    | `/health`   | Health check + config/method status           |
| GET    | `/docs`     | Interactive Swagger UI                         |

### Humanize

```bash
curl -X POST http://localhost:8000/humanize \
  -H "Content-Type: application/json" \
  -d '{"text":"YOUR TEXT HERE","method":"llm_rewrite"}'
```

```json
{
  "result": "…rewritten text…",
  "method": "llm_rewrite",
  "processing_time_ms": 4200
}
```

### Detect

```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"text":"YOUR TEXT HERE"}'
```

```json
{
  "ai_likelihood": 0.74,
  "percent": 74,
  "label": "ai",
  "verdict": "Likely AI-generated",
  "confidence": 0.43,
  "signals": { "words": 52, "sentences": 5, "...": "..." },
  "top_signals": [ { "signal": "formal connector density", "ai_score": 0.9 } ]
}
```

Limits: text must be non-empty and at most **20,000 characters**.

---

## Configuration

Edit `config/config.toml`:

```toml
[api_keys]
deepseek_api_key = ""      # required for llm_rewrite (or use openrouter)
openrouter_api_key = ""

[llm]
provider = "deepseek"      # "deepseek" | "openrouter" | "atlascloud"
model = ""                 # empty = provider default
temperature = 1.3
```

Provider and keys can also be set via environment variables
(`LLM_PROVIDER`, `LLM_API_KEY`, `DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY`).

---

## Project layout

```
src/methodologies/
  humanizer.py            # FastAPI app: /humanize, /detect, UI, health
  templates/index.html    # single-page web UI
  translation_chain.py    # translation-based humanizers
  llm_rewriter.py         # LLM rewrite method
  detectors/
    heuristic.py          # ensemble AI-text detector (used by /detect)
config/
  config.example.toml     # copy to config.toml and fill in
docker/Dockerfile
docker-compose.yml
```

---

## Notes

- The AI detector is a **stylometric heuristic**, not a machine-learning
  classifier. Use it as a signal, not as proof of authorship.
- The `translation_chain` method needs outbound internet access to the
  translation engines but no API key.
- `detection_guided` (an experimental ML method) requires `torch` and
  `transformers`, which are not installed by default.

---

## License

MIT — see [LICENSE](LICENSE).
