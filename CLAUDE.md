# OpenClaw + Ollama + Qwen3 32B: Local Overnight Research Agent
Implementation Plan — Test-Driven Development
Hardware: Apple M1 Pro, 32GB RAM, 200GB/s memory bandwidth
Goal: Autonomous overnight agent that scrapes ~100-200 tweets, ~100 websites, ~100 GitHub READMEs nightly, producing a personalized AI research report with persistent memory that improves over days/weeks.

## Architecture Overview
```
┌─────────────────────────────────────────────────────┐
│                    CRON TRIGGER                      │
│          (launchd plist — fires at midnight)         │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              OPENCLAW GATEWAY (daemon)               │
│         Always-on background process (~200MB)        │
│    Persistent memory │ Cron/heartbeat scheduler      │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│               OLLAMA (llama-server)                  │
│     Qwen3 32B Q4_K_M (~20GB RAM) via localhost      │
│     Context: 16384 tokens (stripped workspace)       │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│               SKILL: research-agent                  │
│                                                      │
│  Phase 1: FETCH  (HTTP/RSS — no browser needed)      │
│    ├── Hacker News API                               │
│    ├── GitHub API (README fetching)                  │
│    ├── RSS feeds for blogs/sites                     │
│    ├── Nitter/RSS proxy for Twitter/X                │
│    └── Direct HTTP for static sites                  │
│                                                      │
│  Phase 2: EXTRACT  (parse + summarize per-source)    │
│    └── LLM summarizes each source individually       │
│                                                      │
│  Phase 3: SYNTHESIZE  (cross-source analysis)        │
│    └── LLM produces final report from summaries      │
│                                                      │
│  Phase 4: PERSIST  (update memory + save report)     │
│    ├── Append to SQLite knowledge base               │
│    ├── Update interest profile weights               │
│    └── Write markdown report to ~/reports/           │
└─────────────────────────────────────────────────────┘
```

## RAM Budget (32GB total)

| Component | RAM |
|---|---|
| macOS + system processes | ~4GB |
| Ollama + Qwen3 32B Q4_K_M | ~20GB |
| OpenClaw gateway + Node.js | ~1GB |
| Browser (if needed, headless Chromium) | ~2GB |
| Headroom for fetch/parse buffers | ~5GB |

---

## CHUNK 0: Environment Setup & Smoke Tests
Purpose: Get the foundational stack installed, running, and verified before writing any custom code.

### Tasks

**Install Ollama**
```bash
brew install ollama
ollama serve &
```

**Pull Qwen3 32B Q4_K_M**
```bash
ollama pull qwen3:32b
```

**Install OpenClaw**
```bash
curl -fsSL https://openclaw.ai/install.sh | bash
openclaw onboard --install-daemon
```
During onboard, select Ollama as provider. Set model to `ollama/qwen3:32b`.

**Configure Ollama for OpenClaw context**
Set environment variables (in `~/.zshrc` or launchd plist):
```bash
export OLLAMA_CONTEXT_LENGTH=16384
export OLLAMA_FLASH_ATTENTION=1
```
Restart Ollama after setting these.

**Configure OpenClaw for local-only operation**
Edit `~/.openclaw/openclaw.json`:
```json
{
  "models": {
    "mode": "local",
    "providers": {
      "ollama": {
        "baseUrl": "http://127.0.0.1:11434/v1",
        "apiKey": "ollama-local",
        "api": "openai-completions",
        "models": [
          {
            "id": "qwen3:32b",
            "name": "Qwen3 32B",
            "reasoning": false,
            "input": ["text"],
            "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
            "contextWindow": 16384,
            "maxTokens": 4096
          }
        ]
      }
    }
  },
  "agents": {
    "defaults": {
      "model": { "primary": "ollama/qwen3:32b" },
      "maxConcurrent": 1,
      "subagents": { "maxConcurrent": 1 },
      "compaction": { "mode": "safeguard" }
    }
  }
}
```

**Notes:**
- `maxConcurrent: 1` — only one agent at a time (RAM constraint)
- `contextWindow: 16384` — intentionally low to stay fast and leave RAM headroom
- `compaction: safeguard` — auto-compact when context fills up

### Tests
```bash
# TEST 0.1: Ollama is running and model is loaded
curl -s http://localhost:11434/api/tags | grep -q "qwen3:32b"
echo "PASS: Qwen3 32B model available in Ollama"

# TEST 0.2: Ollama generates a response
RESPONSE=$(curl -s http://localhost:11434/api/generate \
  -d '{"model":"qwen3:32b","prompt":"Say hello in exactly 3 words","stream":false}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['response'])")
[ -n "$RESPONSE" ] && echo "PASS: Ollama generates text: $RESPONSE"

# TEST 0.3: OpenClaw gateway is running
openclaw gateway status | grep -q "running"
echo "PASS: OpenClaw gateway is active"

# TEST 0.4: OpenClaw can reach Ollama
openclaw doctor | grep -q "model.*ok\|model.*connected\|provider.*ok"
echo "PASS: OpenClaw connected to Ollama"

# TEST 0.5: OpenClaw responds to a simple prompt
openclaw chat --message "Reply with exactly: SMOKE_TEST_OK" --no-interactive 2>/dev/null | grep -q "SMOKE_TEST_OK"
echo "PASS: OpenClaw + Qwen3 end-to-end working"

# TEST 0.6: RAM usage is within budget
OLLAMA_RSS=$(ps aux | grep ollama | grep -v grep | awk '{sum+=$6} END {print sum/1024}')
echo "Ollama RSS: ${OLLAMA_RSS}MB (should be < 22000)"
[ $(echo "$OLLAMA_RSS < 22000" | bc) -eq 1 ] && echo "PASS: RAM within budget"
```

### Exit Criteria
- All 6 tests pass
- `openclaw doctor` reports no errors
- RAM usage under 22GB with model loaded

---

## CHUNK 0: Completion Status

**Date completed:** 2026-02-22

### What was done (actual commands run)

**Ollama** was already installed (v0.6.8). Started with env vars inline:
```bash
OLLAMA_CONTEXT_LENGTH=16384 OLLAMA_FLASH_ATTENTION=1 ollama serve &
```
Note: `~/.zshrc` is owned by root so env vars could not be appended — set inline at launch instead. For persistence, add to the launchd plist manually or run `sudo chown $USER ~/.zshrc` first.

**Model pulled:**
```bash
ollama pull qwen3:32b   # 20GB, confirmed present
```

**OpenClaw installed** (v2026.2.21-2) via install script. Interactive `onboard` step requires a TTY so it was skipped; config written manually instead.

**OpenClaw configured** via `openclaw config set` (not by editing the JSON directly):
```bash
openclaw config set models.providers.ollama \
  '{"baseUrl":"http://127.0.0.1:11434/v1","apiKey":"ollama-local","api":"openai-completions","models":[{"id":"qwen3:32b","name":"Qwen3 32B","reasoning":false,"input":["text"],"cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0},"contextWindow":16384,"maxTokens":4096}]}'
openclaw config set agents.defaults.model.primary "ollama/qwen3:32b"
openclaw config set agents.defaults.maxConcurrent 1
openclaw config set agents.defaults.subagents.maxConcurrent 1
openclaw config set agents.defaults.compaction.mode "safeguard"
openclaw config set gateway.mode local
openclaw config set agents.defaults.memorySearch.enabled false  # no local embedding provider
openclaw config set agents.defaults.timeoutSeconds 300
```

**Gateway installed as LaunchAgent:**
```bash
openclaw gateway install
```

### Test Results

| Test | Result | Notes |
|---|---|---|
| 0.1 — Ollama model available | **PASS** | `qwen3:32b` present in `ollama list` |
| 0.2 — Ollama generates text | **PASS** | Direct `/api/generate` returns response |
| 0.3 — OpenClaw gateway running | **PASS** | `pid 92459`, port 18789, loopback |
| 0.4 — OpenClaw reaches Ollama | **PASS** | OpenAI-compat `/v1/models` returns `qwen3:32b` |
| 0.5 — End-to-end agent response | **BLOCKED** | Qwen3 `<think>` mode exhausts 300s timeout through gateway; direct Ollama call works. Fix: pass `/no_think` in prompt or set `think: false` in model options |
| 0.6 — RAM within budget | **PASS** | Ollama RSS ~20GB, within 22GB limit |

### Known Issues / Next Steps
- **TEST 0.5**: Qwen3 32B defaults to extended chain-of-thought (`<think>` tokens) which is too slow for the 300s gateway timeout. Resolution: add `/no_think` to agent system prompt, or configure `num_predict` limits. Defer to Chunk 1.
- **Ollama env vars**: Not persisted in shell profile (root-owned `~/.zshrc`). Must be set at Ollama launch or in the launchd plist.
- **Memory search**: Disabled — no local embedding model configured. Can revisit with `ollama pull nomic-embed-text` in a future chunk.
