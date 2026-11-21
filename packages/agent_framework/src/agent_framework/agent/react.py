"""ReActAgent - tool-calling loop for tools_agent.

Implements the Reasoning + Acting loop:
  1. Call LLM with messages + tool schemas
  2. If no tool_calls → return content as final answer
  3. Execute each tool_call, append results
  4. Optional reflexion prompt
  5. Repeat until max_turns
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agent_framework.agent.base import (
    BaseAgent,
    MessageHistory,
    estimate_tokens,
    truncate_tool_result,
)
from agent_framework.exceptions import MaxTurnsExceeded
from agent_framework.llm.types import Message
from agent_framework.tracing.context import span

logger = logging.getLogger(__name__)


class ReActAgent(BaseAgent):
    """Tool-calling agent with ReAct + optional Reflexion.

    Each ``run()`` call is an independent session: the message history
    is created fresh from the system prompt, runs the loop, and is
    discarded when done.  This isolates per-operator context.

    Parameters
    ----------
    system_prompt:
        System message injected at the start of every session.
    reflexion_enabled:
        Whether to inject a reflexion check after each tool round.
    max_tool_result_chars:
        Truncation limit for tool outputs written into messages.
    context_compress_threshold:
        Estimated token count that triggers in-session compression.
    """

    def __init__(
        self,
        *args: Any,
        system_prompt: str = "",
        reflexion_enabled: bool = True,
        max_tool_result_chars: int = 8000,
        context_compress_threshold: int = 30000,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.system_prompt = system_prompt
        self.reflexion_enabled = reflexion_enabled
        self.max_tool_result_chars = max_tool_result_chars
        self.context_compress_threshold = context_compress_threshold

    async def run(self, input_data: Any) -> Any:
        """Execute a single ReAct session.

        Parameters
        ----------
        input_data:
            Either a string (user message) or a dict with structured input.
            Dicts are JSON-serialized into the user message.

        Returns
        -------
        The final text content from the LLM (when it stops calling tools).

        Raises
        ------
        MaxTurnsExceeded
            If the loop hits ``max_turns`` without a final answer.
        """
        history = MessageHistory(self.system_prompt)

        # Build user message from input_data.
        if isinstance(input_data, str):
            user_content = input_data
        else:
            user_content = json.dumps(input_data, ensure_ascii=False, default=str)
        history.append(Message(role="user", content=user_content))

        for turn in range(1, self.max_turns + 1):
            ctx = self._hook_context(turn=turn)
            await self.hooks.on_turn_start(ctx)

            # ── maybe compress context ───────────────────────────
            await self._maybe_compress(history, turn)

            # ── LLM call ─────────────────────────────────────────
            with span(
                "llm_call",
                span_type="llm_call",
                agent_name=self.name,
                turn=turn,
            ):
                await self.hooks.on_llm_start(ctx)
                response = await self.llm_client.chat_completions(
                    history.messages,
                    tools=self.tools.schemas if self.tools.schemas else None,
                )
                await self.hooks.on_llm_end(ctx, response)

            # ── no tool_calls → final answer ─────────────────────
            if not response.has_tool_calls:
                await self.hooks.on_turn_end(ctx)
                # If a result_schema is configured, extract structured output
                # instead of returning raw text.
                if self.result_schema is not None:
                    history.append(response.message)
                    structured = await self.llm_client.get_structured_output(
                        history.messages, self.result_schema,
                    )
                    if structured is not None:
                        return structured
                    logger.warning(
                        "[%s] Structured output extraction failed, "
                        "falling back to raw content",
                        self.name,
                    )
                return response.content

            # ── execute tool calls ───────────────────────────────
            history.append(response.message)

            for tc in response.tool_calls:
                tool_name = tc.function.name
                await self.hooks.on_tool_start(ctx, tool_name)

                result_text = await self._execute_tool(tc.function.name, tc.function.arguments)
                result_text = truncate_tool_result(result_text, self.max_tool_result_chars)

                history.append(
                    Message(role="tool", tool_call_id=tc.id, content=result_text)
                )
                await self.hooks.on_tool_end(ctx, tool_name, result_text)

            # ── reflexion check ──────────────────────────────────
            if self.reflexion_enabled:
                history.append(
                    Message(
                        role="user",
                        content=(
                            "Review your last tool results. "
                            "Are they correct and complete? "
                            "If yes, provide the final answer. "
                            "If not, explain what's wrong and try again."
                        ),
                    )
                )

            await self.hooks.on_turn_end(ctx)

        raise MaxTurnsExceeded(
            f"Agent '{self.name}' exceeded {self.max_turns} turns",
            context={"agent": self.name, "max_turns": self.max_turns},
        )

    # ── tool execution ───────────────────────────────────────────

    async def _execute_tool(self, name: str, arguments_json: str) -> str:
        """Call a registered tool and return its result as a string.

        Errors are caught and returned as descriptive text so the LLM
        can self-correct rather than crash the loop.
        """
        with span("tool_call", span_type="tool", agent_name=self.name) as s:
            s.input_data = {"tool": name, "arguments": arguments_json}

            if name not in self.tools:
                available = ", ".join(self.tools.names)
                error_msg = (
                    f"Tool '{name}' not found. Available tools: {available}"
                )
                s.fail(error_msg)
                return error_msg

            try:
                args = json.loads(arguments_json)
            except json.JSONDecodeError as exc:
                error_msg = f"Invalid JSON arguments for tool '{name}': {exc}"
                s.fail(error_msg)
                return error_msg

            try:
                func = self.tools.get(name)
                assert func is not None
                import asyncio
                if asyncio.iscoroutinefunction(func):
                    result = await func(**args)
                else:
                    result = func(**args)

                result_str = (
                    json.dumps(result, ensure_ascii=False, default=str)
                    if not isinstance(result, str)
                    else result
                )
                s.output_data = {"result_preview": result_str[:500]}
                return result_str
            except Exception as exc:
                error_msg = f"Tool '{name}' raised {type(exc).__name__}: {exc}"
                s.fail(error_msg)
                logger.warning("Tool execution error: %s", error_msg)
                return error_msg

    # ── context compression ──────────────────────────────────────

    async def _maybe_compress(self, history: MessageHistory, turn: int) -> None:
        """Compress history if estimated tokens exceed threshold.

        Strategy: keep system prompt + LLM-generated summary of middle
        history + last 2 rounds of raw messages.
        """
        est_tokens = estimate_tokens(
            "".join(m.content or "" for m in history.messages)
        )
        if est_tokens < self.context_compress_threshold:
            return

        logger.info(
            "[%s] Context compression triggered at turn %d (est. %d tokens)",
            self.name,
            turn,
            est_tokens,
        )

        msgs = history.messages
        if len(msgs) < 6:
            return  # not enough history to compress

        # Keep system prompt (index 0) and last 4 messages (2 rounds).
        system_msg = msgs[0] if msgs[0].role == "system" else None
        middle = msgs[1:-4]
        recent = msgs[-4:]

        # Build a summary of the middle section.
        summary_parts: list[str] = []
        for m in middle:
            if m.role == "assistant" and m.tool_calls:
                for tc in m.tool_calls:
                    summary_parts.append(f"Called {tc.function.name}({tc.function.arguments[:100]})")
            elif m.role == "tool":
                summary_parts.append(f"Tool result: {(m.content or '')[:150]}")
            elif m.content:
                summary_parts.append(f"[{m.role}]: {m.content[:150]}")

        summary_text = "History summary: " + "; ".join(summary_parts)

        # Rebuild history.
        history.reset(keep_system=False)
        if system_msg:
            history.append(system_msg)
        history.append(Message(role="system", content=summary_text))
        history.extend(recent)

        logger.info(
            "[%s] Compressed %d messages to %d",
            self.name,
            len(msgs),
            len(history),
        )
