"""Observability & Langfuse Tracing Service (Feature 16).

Provides end-to-end distributed tracing for /chat requests:
- One root Trace per /chat invocation with user, session, and conversation metadata.
- Fine-grained Spans for:
  1. `retrieval` (recording vector/keyword candidate counts, retrieved chunk IDs, latency).
  2. `permission_filtering` (recording allowed vs filtered chunk counts, user permission tokens).
  3. `reranking` (recording candidate count, cross-encoder scores per chunk, latency).
  4. `generation` (recording Groq prompt & completion token usage, citations, latency).
- Trace Events recording Guardrail triggers and Model Fallback events.
- Compatible with Langfuse cloud export and in-memory trace registry.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import logging
import time
from typing import Any, Dict, Generator, List, Optional
import uuid

from app.core.config import settings
from app.models.user import User

logger = logging.getLogger(__name__)


@dataclass
class TraceEvent:
    """Represents a discrete event or guardrail/fallback trigger on a trace."""

    name: str
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "timestamp": self.timestamp,
            "data": self.data,
        }


@dataclass
class Span:
    """Represents an execution span within a chat trace."""

    name: str
    start_time: float = field(default_factory=time.perf_counter)
    end_time: Optional[float] = None
    duration_ms: float = 0.0
    input: Any = None
    output: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: str = "ok"  # "ok" or "error"
    error_message: Optional[str] = None

    def finish(
        self,
        output: Any = None,
        status: str = "ok",
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mark span completion and record duration."""
        self.end_time = time.perf_counter()
        self.duration_ms = round((self.end_time - self.start_time) * 1000, 2)
        if output is not None:
            self.output = output
        self.status = status
        self.error_message = error_message
        if metadata:
            self.metadata.update(metadata)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "input": self.input,
            "output": self.output,
            "metadata": self.metadata,
            "error_message": self.error_message,
        }


@dataclass
class Trace:
    """Root trace for a single /chat query execution."""

    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "chat_request"
    user_id: str = ""
    user_email: str = ""
    user_role: str = ""
    user_department: str = ""
    conversation_id: str = ""
    input_query: str = ""
    output_answer: Optional[str] = None
    start_time: float = field(default_factory=time.perf_counter)
    end_time: Optional[float] = None
    duration_ms: float = 0.0
    spans: List[Span] = field(default_factory=list)
    events: List[TraceEvent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_span(self, span: Span) -> None:
        self.spans.append(span)

    def add_event(self, name: str, data: Optional[Dict[str, Any]] = None) -> None:
        event = TraceEvent(name=name, data=data or {})
        self.events.append(event)
        logger.info("Trace [%s] Event: %s -> %s", self.trace_id[:8], name, data)

    def finish(self, output_answer: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.end_time = time.perf_counter()
        self.duration_ms = round((self.end_time - self.start_time) * 1000, 2)
        if output_answer is not None:
            self.output_answer = output_answer
        if metadata:
            self.metadata.update(metadata)

    def get_span(self, name: str) -> Optional[Span]:
        """Find the first span matching name."""
        for s in self.spans:
            if s.name == name:
                return s
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "user_id": self.user_id,
            "user_email": self.user_email,
            "user_role": self.user_role,
            "user_department": self.user_department,
            "conversation_id": self.conversation_id,
            "input_query": self.input_query,
            "output_answer": self.output_answer,
            "duration_ms": self.duration_ms,
            "spans": [s.to_dict() for s in self.spans],
            "events": [e.to_dict() for e in self.events],
            "metadata": self.metadata,
        }


# Context variable holding the active trace for the current async task
_CURRENT_TRACE: ContextVar[Optional[Trace]] = ContextVar("current_trace", default=None)

# In-memory registry for querying recent traces (useful for tests and admin observation)
_RECENT_TRACES: List[Trace] = []
MAX_SAVED_TRACES = 100


def get_current_trace() -> Optional[Trace]:
    """Get the active Trace for the current context."""
    return _CURRENT_TRACE.get()


def start_trace(
    user: User,
    query: str,
    conversation_id: Optional[uuid.UUID] = None,
    name: str = "chat_request",
    metadata: Optional[Dict[str, Any]] = None,
) -> Trace:
    """Initialize and set a new active Trace in the current context."""
    trace = Trace(
        name=name,
        user_id=str(user.id),
        user_email=user.email,
        user_role=user.role,
        user_department=user.department,
        conversation_id=str(conversation_id or uuid.uuid4()),
        input_query=query,
        metadata=metadata or {},
    )
    _CURRENT_TRACE.set(trace)
    return trace


def end_trace(output_answer: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Optional[Trace]:
    """Finalize the active trace and store in the trace registry."""
    trace = _CURRENT_TRACE.get()
    if trace:
        trace.finish(output_answer=output_answer, metadata=metadata)
        _RECENT_TRACES.append(trace)
        if len(_RECENT_TRACES) > MAX_SAVED_TRACES:
            _RECENT_TRACES.pop(0)

        # Export to Langfuse if configured
        _export_to_langfuse(trace)

        logger.info(
            "Trace [%s] completed in %.2fms with %d spans and %d events.",
            trace.trace_id[:8],
            trace.duration_ms,
            len(trace.spans),
            len(trace.events),
        )
    return trace


@contextmanager
def record_span(
    name: str,
    input_data: Any = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Generator[Span, None, None]:
    """Context manager to record a named execution span on the active trace.

    Usage:
        with record_span("retrieval", input_data=query) as span:
            results = do_retrieval()
            span.finish(output=len(results), metadata={"chunk_ids": ...})
    """
    span = Span(name=name, input=input_data, metadata=metadata or {})
    trace = _CURRENT_TRACE.get()
    if trace:
        trace.add_span(span)

    try:
        yield span
    except Exception as exc:
        span.finish(status="error", error_message=str(exc))
        raise
    else:
        if span.end_time is None:
            span.finish()


def record_trace_event(name: str, data: Optional[Dict[str, Any]] = None) -> None:
    """Record an event (e.g. guardrail trigger or fallback) on the active trace."""
    trace = _CURRENT_TRACE.get()
    if trace:
        trace.add_event(name=name, data=data or {})


def get_recent_traces(limit: int = 10) -> List[Trace]:
    """Retrieve recent traces from the registry."""
    return _RECENT_TRACES[-limit:]


def clear_trace_registry() -> None:
    """Clear in-memory trace registry (useful for testing)."""
    _RECENT_TRACES.clear()


def _export_to_langfuse(trace: Trace) -> None:
    """Export completed trace to Langfuse if credentials are configured."""
    if not settings.LANGFUSE_PUBLIC_KEY or not settings.LANGFUSE_SECRET_KEY:
        return

    try:
        from langfuse import Langfuse

        lf = Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
        )

        lf_trace = lf.trace(
            id=trace.trace_id,
            name=trace.name,
            user_id=trace.user_id,
            session_id=trace.conversation_id,
            input=trace.input_query,
            output=trace.output_answer,
            metadata={
                **trace.metadata,
                "user_email": trace.user_email,
                "user_role": trace.user_role,
                "user_department": trace.user_department,
            },
        )

        for span in trace.spans:
            lf_trace.span(
                name=span.name,
                input=span.input,
                output=span.output,
                metadata=span.metadata,
                level="ERROR" if span.status == "error" else "DEFAULT",
                status_message=span.error_message,
            )

        for event in trace.events:
            lf_trace.event(
                name=event.name,
                metadata=event.data,
            )

        lf.flush()
    except Exception as exc:
        logger.debug("Langfuse export notice: %s", exc)
