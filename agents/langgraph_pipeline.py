"""SDLC multi-agent pipeline built as a LangGraph state graph (Groq LLM)."""
from __future__ import annotations

import operator
import os
import time
from pathlib import Path
from typing import Annotated, Any, TypedDict

from dotenv import load_dotenv
from groq import Groq
from langgraph.graph import END, START, StateGraph

from feedback.store import feedback_prompt_for_agent
from observability.tracker import ObservabilityTracker
from rag.knowledge import build_context
from utils.docx_writer import write_release_notes

ROOT = Path(__file__).resolve().parents[1]
# Do not override env vars already set by Streamlit Cloud / HF secrets.
load_dotenv(ROOT / ".env", override=False)

OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_groq_api_key() -> str:
    """Prefer process env, then Streamlit secrets (Community Cloud)."""
    key = (os.getenv("GROQ_API_KEY") or "").strip()
    if key:
        return key
    try:
        import streamlit as st

        value = st.secrets.get("GROQ_API_KEY")  # type: ignore[attr-defined]
        if value:
            key = str(value).strip()
            os.environ["GROQ_API_KEY"] = key
            return key
    except Exception:
        pass
    raise RuntimeError(
        "GROQ_API_KEY missing. Set it in Streamlit Cloud Secrets, or in local .env / .streamlit/secrets.toml"
    )

AGENTS = [
    {
        "key": "ba",
        "name": "Business Analyst",
        "role": "Business Analyst",
        "system": (
            "You are an expert healthcare Business Analyst. Clarify business, "
            "functional, and non-functional requirements. Respect HIPAA/PHI. "
            "Produce user stories, acceptance criteria, and a mermaid flowchart "
            "of the main process flow."
        ),
    },
    {
        "key": "architect",
        "name": "Solution Architect",
        "role": "Solution Architect",
        "system": (
            "You are a healthcare Solution Architect. Design pragmatic architecture: "
            "components, data model, APIs, security/HIPAA, deployment."
        ),
    },
    {
        "key": "developer",
        "name": "Full-Stack Developer",
        "role": "Full-Stack Developer",
        "system": (
            "You are an expert Python developer. Produce Python backend design and "
            "Streamlit frontend structure with clear code snippets."
        ),
    },
    {
        "key": "tester",
        "name": "QA Engineer",
        "role": "QA Engineer",
        "system": (
            "You are a QA engineer. Create pytest unit tests and automation test "
            "cases covering happy path and edge cases for healthcare workflows."
        ),
    },
    {
        "key": "devops",
        "name": "DevOps Engineer",
        "role": "DevOps Engineer",
        "system": (
            "You are a healthcare DevOps Engineer. Produce a CI/CD pipeline "
            "(GitHub Actions YAML), Dockerfile, environment and secrets strategy, "
            "infrastructure/deployment plan, monitoring and alerting, backup and "
            "rollback procedures, with HIPAA-aware controls for PHI."
        ),
    },
    {
        "key": "docs",
        "name": "Technical Writer",
        "role": "Technical Writer",
        "system": (
            "You are a technical writer skilled at Word-ready documentation. "
            "Produce SDLC process summary and release notes."
        ),
    },
]


class SDLCState(TypedDict):
    """Shared state passed between graph nodes."""

    requirement: str
    wiki_topic: str
    context: str
    # Each agent node returns only its own entry; the reducer merges them.
    artifacts: Annotated[dict[str, str], operator.or_]
    timings: Annotated[dict[str, float], operator.or_]


def _groq_client() -> Groq:
    return Groq(api_key=_resolve_groq_api_key())


def _empty_response_error(resp: Any) -> RuntimeError:
    return RuntimeError(
        f"Model returned an empty response (finish_reason={resp.choices[0].finish_reason})"
    )


def _chat(client: Groq, system: str, user: str) -> str:
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    last_exc: Exception | None = None
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            if "rate_limit" in msg or "429" in msg:
                time.sleep(5 * (attempt + 1))
                continue
            raise
        text = (resp.choices[0].message.content or "").strip()
        if text:
            return text
        # Retry rather than ship a blank section in the release notes.
        last_exc = _empty_response_error(resp)
    raise last_exc  # type: ignore[misc]


def _truncate(text: str, max_chars: int) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20] + "\n...[truncated]..."


def _prior_outputs(artifacts: dict[str, str], budget: int = 3600) -> str:
    """Split the budget across all prior agents so later ones are not cut off."""
    if not artifacts:
        return "(none yet)"
    per_agent = budget // len(artifacts)
    return "\n\n".join(f"### {name}\n{_truncate(out, per_agent)}" for name, out in artifacts.items())


def _retrieve_context(state: SDLCState) -> dict[str, Any]:
    """Graph node: domain PDFs (BM25) + Wikipedia."""
    return {"context": build_context(state["requirement"], wiki_topic=state["wiki_topic"])}


def _make_agent_node(client: Groq, spec: dict[str, str]):
    """Build one graph node per SDLC role."""

    def node(state: SDLCState) -> dict[str, Any]:
        fb = feedback_prompt_for_agent(spec["role"])
        user = (
            f"Healthcare software requirement:\n{_truncate(state['requirement'], 1200)}\n\n"
            f"Domain + Wikipedia context:\n{_truncate(state['context'], 2500)}\n\n"
            f"{_truncate(fb, 600)}\n\n"
            f"Prior agent outputs:\n{_prior_outputs(state['artifacts'])}\n\n"
            f"Produce your deliverable as {spec['role']}."
        )
        started = time.time()
        try:
            output = _chat(client, spec["system"], user)
        except Exception as exc:
            raise RuntimeError(f"{spec['name']} failed: {exc}") from exc
        return {
            "artifacts": {spec["name"]: output},
            "timings": {spec["name"]: time.time() - started},
        }

    return node


def build_graph(client: Groq):
    """START -> retrieve_context -> BA -> Architect -> Developer -> QA -> DevOps -> Writer -> END."""
    graph = StateGraph(SDLCState)
    graph.add_node("retrieve_context", _retrieve_context)
    previous = "retrieve_context"
    graph.add_edge(START, previous)
    for spec in AGENTS:
        graph.add_node(spec["key"], _make_agent_node(client, spec))
        graph.add_edge(previous, spec["key"])
        previous = spec["key"]
    graph.add_edge(previous, END)
    return graph.compile()


def run_sdlc_pipeline(requirement: str, wiki_topic: str = "electronic health record") -> dict[str, Any]:
    tracker = ObservabilityTracker(requirement)
    engine = "langgraph"

    try:
        app = build_graph(_groq_client())
        state = app.invoke(
            {
                "requirement": requirement,
                "wiki_topic": wiki_topic,
                "context": "",
                "artifacts": {},
                "timings": {},
            }
        )
        artifacts: dict[str, str] = state["artifacts"]
        timings: dict[str, float] = state["timings"]

        for spec in AGENTS:
            output = artifacts.get(spec["name"], "")
            tracker.record_agent(
                agent=spec["name"],
                role=spec["role"],
                input_text=requirement,
                output=output,
                notes=f"engine={engine}",
                latency_sec=timings.get(spec["name"]),
            )

        docx_path = OUTPUT_DIR / f"release_notes_{tracker.run.run_id}.docx"
        write_release_notes(
            docx_path,
            title=f"SDLC Release Notes — {tracker.run.run_id}",
            requirement=requirement,
            sections=artifacts,
        )
        tracker.finish("completed")
        return {
            "ok": True,
            "run_id": tracker.run.run_id,
            "engine": engine,
            "context_preview": state["context"][:1200],
            "artifacts": artifacts,
            "docx_path": str(docx_path),
            "metrics": tracker.summary(),
            "final": artifacts.get("Technical Writer", ""),
        }
    except Exception as exc:
        tracker.finish("failed")
        return {
            "ok": False,
            "run_id": tracker.run.run_id,
            "error": str(exc),
            "metrics": tracker.summary(),
        }
