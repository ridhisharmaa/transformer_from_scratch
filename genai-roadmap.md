# Gen AI Engineer Roadmap — v2 (resume-driven)
_Target: Blinkit / Meesho / Zomato / Swiggy / Paytm + big tech_
_2027 grad · updated 2026-08-17_

## What changed from v1
1. **DSA is off the critical path.** LeetCode 1769, top 18%, 615+ solved, 51 contests — you are already
   past the bar for these companies. v1 budgeted 175 hrs for DSA; that was wrong.
   Now: **45 min/day maintenance** (weekly contest + weak topics). ~130 hrs freed.
2. **Timeline compressed: ~8 weeks, not 4 months.**
3. **Structure is now build-first.** You learn RAG by building the RAG project, not by studying RAG
   then building it. Study topics are attached to the week that needs them.
4. **Every week is tied to a resume bullet becoming true.** No study for its own sake.

## Reality check (unchanged)
- Google / Microsoft / Netflix rarely hire freshers into "GenAI Engineer" titles — they hire SWE and
  route you later. Your LeetCode already handles that gate.
- **Blinkit, Meesho, Zomato, Swiggy, Paytm, Sarvam, Fractal and AI startups DO hire freshers directly
  into GenAI/LLM roles.** This is the primary target. Portfolio decides it.

---

## The 8-week plan

### Weeks 1–2 — Finish the Transformer  (~30 hrs)
**Build:** plan.txt steps 8–13 — multi-head backward, encoder/decoder block backward, full backward
chain, Adam, training loop with teacher forcing, train + greedy decode.

**Study alongside (the "modern LLM delta"):**
| Topic | Why |
|---|---|
| Decoder-only vs encoder-decoder vs encoder-only | "Why is GPT decoder-only?" — near-guaranteed |
| Tokenization: BPE / WordPiece / SentencePiece | your tokenizer.py is naive |
| RoPE vs sinusoidal vs learned | you know the baseline |
| RMSNorm vs LayerNorm; pre-norm vs post-norm | you wrote LayerNorm backward |
| SwiGLU vs ReLU FFN | direct delta from feed_forward.py |
| KV cache — why decode is O(n) not O(n^2) | asked in every LLM systems interview |
| GQA / MQA vs MHA | memory-bandwidth reasoning |
| Sampling: temperature, top-k, top-p | easy marks |

**Resume unlocked:** Transformer bullet 3 (training loss numbers).

---

### Week 3 — Port to PyTorch: decoder-only mini-GPT  (~15 hrs)
**Build:** rewrite as a GPT-style decoder-only model in PyTorch with RoPE + RMSNorm + SwiGLU +
**KV caching**. Benchmark decode latency cached vs uncached and record the number.

Why this week exists: **PyTorch currently appears nowhere in your experience** — only as "without
using PyTorch." That is a hole for a GenAI role. This closes it and gives you the systems-engineer
signal without needing a 4th project.

**Resume unlocked:** Transformer bullet 4 (KV-cache speedup %) + PyTorch in Skills becomes real.

---

### Weeks 4–7 — Agentic RAG Assistant  (~100 hrs) ← THE BIG ONE
Domain: pick something q-commerce shaped (product catalog + policy docs + an orders table) so it
reads as directly relevant to Blinkit/Meesho/Zomato.

**Week 4 — Retrieval core.** Ingestion, chunking strategies, embeddings, Qdrant, baseline dense
search. Then BM25, hybrid fusion, cross-encoder reranking. Measure recall@5 at each step.
_Study: bi- vs cross-encoder, HNSW vs IVF, chunking failure modes, lost-in-the-middle._

**Week 5 — Agent layer.** LangGraph tool-calling agent: semantic catalog search, text-to-SQL order
lookup, policy retrieval. Multi-step planning, error handling, citation-grounded answers.
_Study: ReAct, function calling, agent failure modes (error compounding, context bloat, cost blowup), MCP._

**Week 6 — Evaluation harness.** This is the differentiator — build it properly. Golden Q&A set,
RAGAS faithfulness + context precision, LLM-as-judge, wired into GitHub Actions so regressions fail
the build. Add citation enforcement + low-confidence abstention; measure hallucination rate before/after.
_Study: why eval is the hard part, LLM-as-judge bias, regression testing._

**Week 7 — Serving & polish.** FastAPI + SSE streaming, semantic caching, Docker, Langfuse tracing.
Record p95 TTFT and cost per 1K queries. Write the README: architecture diagram, tradeoffs you
**rejected**, measured numbers.
_Study: vLLM/PagedAttention, continuous batching, prefill vs decode, TTFT vs throughput._

**Resume unlocked:** all of Project 2, plus most of the Gen AI Skills row.

---

### Week 8 — Gaps building doesn't teach + interview prep  (~30 hrs)
- **LoRA / PEFT** — know the math: dW = BA, why rank r << d. Do one small fine-tune so the skill is real.
- **Quantization** — int8/int4, GPTQ/AWQ/GGUF. VRAM math: "how much to serve 7B in fp16 vs int4?"
- **Alignment** — pretraining -> SFT -> RLHF (reward model + PPO) -> DPO. Intuition only.
- **Scaling laws**, Chinchilla-optimal, long-context.
- **Decision framework:** prompt vs RAG vs fine-tune. Classic question.
- **ML system design out loud:** design a RAG chatbot, design an LLM serving platform, design ChatGPT.
- Finalize resume — replace every [FILL] marker with real numbers.
- Mock interviews + STAR behavioral stories.

---

## Running throughout
- **DSA: 45 min/day.** Weekly contest + weak topics. Maintenance only, do not over-invest.
- **Apply from Week 5 onward.** Do not wait for the portfolio to be perfect — interviews teach you
  what is missing faster than studying does.
- **Chase a GenAI internship in parallel.** No internship is the one real hole in the resume; even a
  2-month one at a small AI startup changes how it reads.

---

## Time budget

| Block | Hours |
|---|---|
| Weeks 1–2 finish transformer + architecture study | 30 |
| Week 3 PyTorch mini-GPT | 15 |
| Weeks 4–7 agentic RAG project | 100 |
| Week 8 gaps + interview prep | 30 |
| DSA maintenance (8 wks x 45min) | 40 |
| **Total** | **~215 hrs** |

| Pace | Calendar |
|---|---|
| 4 hrs/day | ~8 weeks |
| 3 hrs/day | ~10 weeks |
| 6 hrs/day | ~6 weeks |

---

## The three projects (final)
1. **Transformer from scratch** + PyTorch mini-GPT — proves you understand the machine, not the API.
   Rarest thing on a fresher resume. Headline: hand-derived backprop, gradient-checked to ~1e-10.
2. **Agentic RAG with eval harness** — proves you can ship. Eval numbers are what make it credible.
3. **NASA C-MAPSS** — proves classical ML / time-series. Balances a GenAI-heavy resume.

CheckMate is dropped (copied project — indefensible under 20 minutes of interrogation).

Rule for all three: README with architecture diagram, tradeoffs you **rejected**, and measured numbers.
