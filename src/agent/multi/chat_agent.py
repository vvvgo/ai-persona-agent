"""ChatAgent — modern agent with native LLM function calling.

Uses LiteLLM's bind_tools for structured tool calling when the model supports it.
Falls back to plain text generation + manual speak() call when bind_tools fails.

Features:
- Native function calling via langchain bind_tools (when supported)
- Fallback: plain text response -> speak tool (for models without tool support)
- Proper user/assistant message alternation
- Streaming with real-time console output
- Timing instrumentation for every step
- Interrupt on new chat events
"""

import os
import time
import traceback
from typing import Any, Dict, List, Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from agent.base_agent import BaseAgent
from agent.llm_clients.lc_clients import get_llm_chain
from agent.prompt_generation.message_builder import events_to_messages
from agent.prompt_generation.prompt_constructor import (
    SmartEventWaiter,
    get_game_status,
    get_mood_context,
    get_users_info,
)
from agent.tools import tools_config
from agent.tools.status_tools import get_tool_status
from character.character_manager import CharacterManager
from config_schema.general import get_name
from data_flow.ctx_handler import CtxHandler
from data_flow.ctx_host import CtxHost
from data_schema.activity_tracker import IDLE_SELF_ACTIVATE, IDLE_THRESHOLD
from data_schema.chat_structures import (
    CtxEventBase,
    default_event_check,
    interrupt_event_check,
    nick_mention_check,
)
from data_schema.ctx_structures import CtxSwarmType
from data_schema.tool_structures import SpeakEmotions, ToolBankType
from utils.debug import print_messages

SELF_NAME = get_name()
MAX_TOOL_ROUNDS = 3
# Hard limit on LLM generation per round. Reasoning models (e.g. GLM) spend part of it on thinking
MAX_RESPONSE_TOKENS = int(os.getenv("CHAT_AGENT_MAX_RESPONSE_TOKENS", "500"))
ENABLE_DEBUG_PRINT_TOOL_LIST = True
# Minimum characters for a valid response — shorter = interrupted garbage, discard it
MIN_RESPONSE_LENGTH = 12
# After an interrupt, ignore further interrupts for this many seconds
INTERRUPT_COOLDOWN = 10.0
# Max interrupts per respond() call
MAX_INTERRUPTS_PER_CALL = 2
# On busy servers with a slow reasoning model, interrupts restart every answer and it never finishes
DISABLE_INTERRUPTS = os.getenv("CHAT_AGENT_DISABLE_INTERRUPTS", "false").lower() == "true"
# If the model answered only with the chat tool, use that message as the voice line too
VOICE_CHAT_FALLBACK = os.getenv("CHAT_AGENT_VOICE_CHAT_FALLBACK", "false").lower() == "true"


def _timer(label: str):
    """Simple context-manager-like timer that prints elapsed time."""
    class _T:
        def __init__(self):
            self.start = time.time()
        def __enter__(self):
            self.start = time.time()
            return self
        def __exit__(self, *a):
            elapsed = time.time() - self.start
            print(f"[CHAT AGENT ⏱️ {elapsed:.2f}s] {label}")
    return _T()


# Tools excluded from LLM tool calling — handled directly from text output
DIRECT_TEXT_TOOLS = {"speak", "speak_list"}


def _get_available_tools(tool_bank: ToolBankType, exclude: set = DIRECT_TEXT_TOOLS) -> List:
    tools = []
    for name, record in tool_bank.items():
        if record.get("disabled", False):
            continue
        if name in exclude:
            continue
        lc_tool = record.get("tool")
        if lc_tool is not None:
            tools.append(lc_tool)
    return tools


def _execute_tool_call(tool_bank: ToolBankType, tool_name: str, tool_args: dict) -> Any:
    if tool_name not in tool_bank:
        return f"Error: tool '{tool_name}' not found"
    record = tool_bank[tool_name]
    if record.get("disabled", False):
        return f"Error: tool '{tool_name}' is disabled"
    func = record["function"]
    try:
        return func(**tool_args)
    except TypeError:
        try:
            return func(*tool_args.values())
        except Exception as e:
            return f"Error executing {tool_name}: {e}"
    except Exception as e:
        return f"Error executing {tool_name}: {e}"


class ChatAgent(BaseAgent):
    """Modern chat agent with native function calling + plain-text fallback."""

    def __init__(
        self,
        ctx_swarm: CtxSwarmType,
        ctx_handler: Optional[CtxHandler] = None,
        ctx_host: Optional[CtxHost] = None,
        tool_bank: Optional[ToolBankType] = None,
    ):
        super().__init__(ctx_swarm, ctx_handler)
        if ctx_handler is None:
            self.ctx_handler = CtxHandler(ctx_swarm)
        else:
            self.ctx_handler = ctx_handler

        self.llm = get_llm_chain()
        self.self_name = get_name()

        if tool_bank is not None:
            self.tool_bank = tool_bank
        else:
            self.tool_bank = tools_config.init_tools_module(
                ctx_swarm, self.ctx_handler, new_llm=self.llm
            )

        if ctx_host is None:
            self.ctx_host = CtxHost(ctx_handler, tool_bank=self.tool_bank)
        else:
            self.ctx_host = ctx_host

        # Try to bind tools; fall back to plain text if model doesn't support it
        self._tools_supported = False
        self._rebind_tools()

        # Cache system prompt — must be identical every call for KV cache hits
        self._cached_system_prompt = self._build_system_prompt()
        self._cached_personality = self.ctx_swarm["states"].get("personality", "streamer_abstract")

        # APPEND-ONLY prefix — the core of KV cache strategy.
        # Starts with [SystemMessage], events and LLM responses get appended.
        # Prompt sent to LLM = _cached_prefix + [HumanMessage(tail)]
        # Objects in this list are NEVER replaced or reordered.
        self._cached_prefix: List[BaseMessage] = [
            SystemMessage(content=self._cached_system_prompt),
        ]
        self._last_event_ts: int = 0  # timestamp of last processed event
        self._prefix_max_tokens: int = 15000  # trim when exceeded
        self._prefix_max_messages: int = 40  # hard limit on message count

        # Interrupt rate limiting
        self._last_interrupt_time = 0.0
        self._interrupts_this_call = 0

        # KV cache keepalive — ping LM Studio every few seconds to prevent eviction
        self._keepalive_interval = 3.0  # seconds between pings
        self._keepalive_running = False
        self._keepalive_thread = None
        self._keepalive_enabled = self._should_enable_keepalive()
        self._last_tail_content = "."  # last tail sent to LLM — reused in ping

        self.initialized = True

    def _rebind_tools(self):
        """Bind tools to LLM. If bind_tools fails, fall back to plain text mode."""
        available_tools = _get_available_tools(self.tool_bank)
        self._tool_names = [t.name for t in available_tools]

        if available_tools:
            try:
                self.llm_with_tools = self.llm.bind_tools(available_tools)
                self._tools_supported = True
                print(f"[CHAT AGENT] Native tool calling ON, {len(available_tools)} tools: {self._tool_names}")
                if ENABLE_DEBUG_PRINT_TOOL_LIST:
                    for t in available_tools:
                        desc = (t.description or "no description")
                        args = ""
                        if hasattr(t, "args_schema") and t.args_schema:
                            try:
                                schema = t.args_schema.schema()
                                props = schema.get("properties", {})
                                args = ", ".join(f"{k}: {v.get('type', '?')}" for k, v in props.items())
                            except Exception:
                                args = "?"
                        status = get_tool_status(t.name)
                        status_str = f" | status: {status}" if status else ""
                        print(f"  [{t.name}]({args}) {desc}{status_str}")
            except Exception as e:
                print(f"[CHAT AGENT] bind_tools failed ({e}), falling back to plain text mode")
                self.llm_with_tools = self.llm
                self._tools_supported = False
        else:
            self.llm_with_tools = self.llm
            self._tools_supported = False
            print("[CHAT AGENT] No tools available, plain text mode")

    def _build_system_prompt(self) -> str:
        """Build system prompt. Keep STATIC for prompt caching —
        dynamic info (tool statuses, mood etc.) goes in situation_context."""
        character = CharacterManager.get_main_character()
        personality = self.ctx_swarm["states"].get("personality", "streamer_abstract")
        prompt = character.read_character_prompt(personality)

        emotions = list(SpeakEmotions.copy())
        emo_examples = " ".join(f"*{e}*" for e in list(emotions)[:4])
        prompt += (
            f"\n\n## Output rules"
            f"\nYou are a STREAMER with voice and game chat."
            f"\n"
            f"\nVOICE (your text output): viewers hear this on stream. Players DON'T."
            f"\n- Always end with SINGLE *emotion* in single asterisks: {emo_examples} etc."
            f"\n- Allowed emotions: {', '.join(emotions)}. No others."
            f"\n- Never use double asterisks **like this**. Only single: *happy*"
            f"\n- No roleplay *actions*. Only emotion tag at the end."
            f"\n"
            f"\nCHAT (chat tool): players read this in-game. Viewers see on screen."
            f"\n- When someone writes to you in chat — ALWAYS call chat tool to reply."
            f"\n- Keep under 80 chars."
            f"\n"
            f"\nACTIONS: if a player asks you to DO something (follow, come, attack, go) —"
            f"\n  ACTUALLY DO IT with agent_action tool. Don't just talk about it."
            f"\n  Never write commands as text. Always use tool calls."
        )

        return prompt

    def _build_situation_context(self, mentioned_users: dict) -> str:
        """Build dynamic situation context. Goes in user message (after cached system+tools)."""
        users_info = get_users_info(ctx_swarm=self.ctx_swarm, chat_users=mentioned_users)
        game_status = get_game_status()
        mood = get_mood_context(self.ctx_swarm)
        activity = self.ctx_handler.activity_tracker.to_prompt_string()

        # TEMP: minimal tail for cache testing
        parts = [
            f"# Current Situation:",
            f"## Players:\n{users_info}",
        ]
        # parts = [
        #     f"# Current Situation:",
        #     f"## Game Status:\n{game_status}",
        #     f"## Your Mood:\n{mood}",
        #     f"## Activity:\n{activity}",
        #     f"## Nearby Users & Players:\n{users_info}",
        # ]

        # Tool statuses — TEMPORARILY DISABLED for cache testing
        # TODO: re-enable after confirming cache works without them
        # tool_status_parts = []
        # for tool_name in self._tool_names:
        #     status = get_tool_status(tool_name)
        #     if status:
        #         tool_status_parts.append(f"[{tool_name}] {status}")
        # speak_status = get_tool_status("speak")
        # if speak_status:
        #     tool_status_parts.append(f"[voice] {speak_status}")
        # if tool_status_parts:
        #     parts.append("## Tool info:\n" + "\n".join(tool_status_parts))

        return "\n\n".join(parts)

    def _build_reminder(self, trigger_event: CtxEventBase = None) -> str:
        base = "React to the latest events.\nText = voice *emotion*. Use game tools if needed."
        # If triggered by a player chat message — explicitly remind about chat tool
        if trigger_event and trigger_event.user and trigger_event.type == "chat":
            if not trigger_event.additional_fields.get("self", False):
                base += (
                    f"\n⚠️ {trigger_event.user} wrote in GAME CHAT. "
                    f"They CANNOT hear your voice — call chat tool to reply so they see your answer!"
                )
        return base

    def _can_interrupt(self) -> bool:
        """Check if we're allowed to interrupt right now."""
        if DISABLE_INTERRUPTS:
            return False
        if self._interrupts_this_call >= MAX_INTERRUPTS_PER_CALL:
            return False
        if time.time() - self._last_interrupt_time < INTERRUPT_COOLDOWN:
            return False
        return True

    def _record_interrupt(self):
        """Record that an interrupt happened."""
        self._last_interrupt_time = time.time()
        self._interrupts_this_call += 1

    def _estimate_tokens(self, messages: List[BaseMessage]) -> int:
        """Rough token estimate: ~3 chars per token for multilingual."""
        return sum(len(m.content) for m in messages) // 3

    @staticmethod
    def _should_enable_keepalive() -> bool:
        """Determine if KV cache keepalive should be enabled.

        OVERRIDE_PROMPT_CACHING_PING env var:
          "auto" (default) — enable for lm_studio, disable for others
          "true" — always enable
          "false" — always disable
        """
        import os
        override = os.getenv("OVERRIDE_PROMPT_CACHING_PING", "auto").lower().strip()
        if override == "true":
            return True
        if override == "false":
            return False
        # auto: enable only for local LM Studio
        model_name = os.getenv("CORE_LLM_MODEL_NAME", "").lower()
        enabled = "lm_studio" in model_name or "lm-studio" in model_name or "lmstudio" in model_name
        print(f"[CHAT AGENT] KV cache keepalive: {'ON' if enabled else 'OFF'} (model={model_name}, override={override})")
        return enabled

    def _start_keepalive(self):
        """Start background thread that pings LM Studio to keep KV cache alive."""
        if not self._keepalive_enabled:
            return
        if self._keepalive_running:
            return
        self._keepalive_running = True
        from threading import Thread
        self._keepalive_thread = Thread(target=self._keepalive_loop, daemon=True)
        self._keepalive_thread.start()

    def _stop_keepalive(self):
        """Stop keepalive pinging (called before LLM request)."""
        self._keepalive_running = False

    def _keepalive_loop(self):
        """Ping LM Studio with cached prefix + last tail to prevent KV cache eviction.
        Uses the SAME tail as last real request — maximizes cache hit on next call."""
        llm = self.llm_with_tools if self._tools_supported else self.llm
        while self._keepalive_running:
            time.sleep(self._keepalive_interval)
            if not self._keepalive_running:
                break
            if len(self._cached_prefix) <= 1:
                continue
            try:
                ping_msgs = list(self._cached_prefix) + [HumanMessage(content=self._last_tail_content)]
                # Suppress litellm info messages during ping
                import logging
                logging.getLogger("LiteLLM").setLevel(logging.ERROR)
                for _ in llm.stream(ping_msgs, max_tokens=1):
                    pass
                logging.getLogger("LiteLLM").setLevel(logging.WARNING)
            except Exception:
                pass

    def _invalidate_cache(self):
        """Full cache reset — rebuilds prefix from scratch."""
        self._cached_prefix = [SystemMessage(content=self._cached_system_prompt)]
        self._last_event_ts = 0

    def _append_to_prefix(self, msgs: List[BaseMessage]):
        """Append messages to cached prefix. Merges consecutive HumanMessages
        in-place (Qwen requires user/assistant alternation)."""
        for msg in msgs:
            if (
                isinstance(msg, HumanMessage)
                and self._cached_prefix
                and isinstance(self._cached_prefix[-1], HumanMessage)
            ):
                # Merge into existing HumanMessage — same object, mutated content
                self._cached_prefix[-1].content += "\n\n" + msg.content
            else:
                self._cached_prefix.append(msg)

    def _trim_prefix_if_needed(self):
        """Trim oldest messages from prefix if too long.
        Keeps SystemMessage[0] intact, trims from index 1."""
        token_est = self._estimate_tokens(self._cached_prefix)
        msg_count = len(self._cached_prefix)
        if token_est <= self._prefix_max_tokens and msg_count <= self._prefix_max_messages:
            return

        # Keep system message + last 2/3 of the rest
        rest = self._cached_prefix[1:]
        keep = min(len(rest) * 2 // 3, self._prefix_max_messages - 1)
        removed = len(rest) - keep
        self._cached_prefix = [self._cached_prefix[0]] + rest[-keep:]

        # After trim, ensure first message after system is HumanMessage (Qwen requirement)
        if len(self._cached_prefix) > 1 and not isinstance(self._cached_prefix[1], HumanMessage):
            self._cached_prefix.insert(1, HumanMessage(content="(conversation start)"))

        new_est = self._estimate_tokens(self._cached_prefix)
        print(f"[CHAT AGENT] Prefix trimmed: removed {removed} msgs, ~{new_est} tokens left")

    def respond(self, trigger_event: CtxEventBase) -> str:
        if not self.initialized:
            return ""
        self._interrupts_this_call = 0

        # Rebuild system prompt if personality changed → full cache invalidation
        current_personality = self.ctx_swarm["states"].get("personality", "streamer_abstract")
        if current_personality != self._cached_personality:
            print(f"[CHAT AGENT] Personality changed: {self._cached_personality} -> {current_personality}")
            self._cached_system_prompt = self._build_system_prompt()
            self._cached_personality = current_personality
            self._invalidate_cache()

        t_total = time.time()

        # --- 1. Append new events to prefix ---
        with _timer("build prompt"):
            all_events = self.ctx_handler.get_ctx_chat(validate=True, dict_format=True, limit=50)
            new_events = [
                e for e in all_events
                if int(e.get("processing_timestamp", 0)) > self._last_event_ts
            ]

            if new_events:
                new_msgs, mentioned_users = events_to_messages(
                    new_events, trigger_event.processing_timestamp
                )
                # Ensure first thing after SystemMessage is a HumanMessage
                if len(self._cached_prefix) == 1 and new_msgs and not isinstance(new_msgs[0], HumanMessage):
                    self._cached_prefix.append(HumanMessage(content="(conversation start)"))

                self._append_to_prefix(new_msgs)
                self._last_event_ts = max(
                    int(e.get("processing_timestamp", 0)) for e in new_events
                )
                cache_status = f"append {len(new_msgs)} msgs from {len(new_events)} events"
            else:
                mentioned_users = {}
                from agent.prompt_generation.message_builder import _track_user
                for ev in all_events:
                    _track_user(ev, mentioned_users)
                cache_status = "no new events"

            self._trim_prefix_if_needed()

            # --- 2. Build tail (the ONLY dynamic message) ---
            reminder = self._build_reminder(trigger_event)
            situation_context = self._build_situation_context(mentioned_users)
            tail = situation_context
            if reminder:
                tail += "\n\n" + reminder

            # --- 3. Assemble: prefix (frozen) + tail (fresh) ---
            # prefix is passed by reference — _respond_with_tools will append to it
            tail_msg = HumanMessage(content=tail)
            lc_messages = self._cached_prefix + [tail_msg]
            # Save tail for keepalive ping — next ping uses same tail for max cache hit
            self._last_tail_content = tail

        # --- Debug: prefix hash + litellm API-level hash ---
        import hashlib, json as _json

        prefix_hash = hashlib.md5(
            "|||".join(
                f"{type(m).__name__}:{m.content}:{getattr(m, 'tool_calls', '')}"
                for m in self._cached_prefix
            ).encode()
        ).hexdigest()[:8]
        if not hasattr(self, '_last_prefix_hash'):
            self._last_prefix_hash = None
        cache_hit = "HIT" if prefix_hash == self._last_prefix_hash else "MISS"
        self._last_prefix_hash = prefix_hash

        # Litellm interceptor — hash ACTUAL API request to find serialization diffs
        if not hasattr(self, '_litellm_patched'):
            self._litellm_patched = True
            self._prev_api_hash = None
            import litellm
            _orig_completion = litellm.completion
            def _cache_debug_completion(*args, **kwargs):
                msgs = kwargs.get('messages', [])
                tools = kwargs.get('tools', [])
                api_payload = _json.dumps(msgs, sort_keys=True, ensure_ascii=False) + _json.dumps(tools, sort_keys=True, ensure_ascii=False)
                h = hashlib.md5(api_payload.encode()).hexdigest()[:12]
                status = "API_HIT" if h == self._prev_api_hash else "API_MISS"
                print(f"  [LITELLM] hash={h} ({status}) n_msgs={len(msgs)}")
                if h != self._prev_api_hash and self._prev_api_hash and hasattr(self, '_prev_api_msgs'):
                    for j, (pm, cm) in enumerate(zip(self._prev_api_msgs, msgs)):
                        ps = _json.dumps(pm, sort_keys=True, ensure_ascii=False)
                        cs = _json.dumps(cm, sort_keys=True, ensure_ascii=False)
                        if ps != cs:
                            print(f"  [LITELLM DIFF] msg[{j}] role={cm.get('role','')} changed!")
                            print(f"    prev: {ps[:150]}")
                            print(f"    curr: {cs[:150]}")
                            break
                self._prev_api_hash = h
                self._prev_api_msgs = msgs
                return _orig_completion(*args, **kwargs)
            litellm.completion = _cache_debug_completion

        msg_types = [
            f"{type(m).__name__}({len(m.content)})"
            + (f"+tc:{len(m.tool_calls)}" if hasattr(m, 'tool_calls') and m.tool_calls else "")
            for m in lc_messages
        ]
        print(f"\n[CHAT AGENT] Prefix: {len(self._cached_prefix)} msgs ({cache_status})")
        print(f"[CHAT AGENT] Prefix hash: {prefix_hash} ({cache_hit})")
        print(f"[CHAT AGENT] Message types: {msg_types}")
        print("\n[CHAT AGENT] === PROMPT ===")
        print_messages(lc_messages)

        # --- 4. LLM call ---
        self._stop_keepalive()  # pause pinging during actual LLM request
        smart_event_waiter = SmartEventWaiter(ctx_handler=self.ctx_handler)
        self.ctx_swarm["fx_queue"].put("thinking")

        if self._tools_supported:
            final_text = self._respond_with_tools(lc_messages, smart_event_waiter)
        else:
            final_text = self._respond_plain_text(lc_messages, smart_event_waiter)

        smart_event_waiter.shutdown()
        self._start_keepalive()  # resume pinging to keep cache warm

        elapsed_total = time.time() - t_total
        print(f"\n[CHAT AGENT ⏱️ {elapsed_total:.2f}s total] Final response ({len(final_text)} chars): {final_text}")
        return final_text

    def _respond_with_tools(self, lc_messages: List[BaseMessage], event_waiter: SmartEventWaiter) -> str:
        """Response path: LLM text = speech, tool calls = game actions.

        IMPORTANT: appends AI/Tool messages to self._cached_prefix so they
        become part of the frozen prefix for the next call. lc_messages is
        _cached_prefix + [tail] — we rebuild it each round by re-appending tail.
        """
        final_text = ""
        tool_round = 0
        # The tail is always the last message in lc_messages
        tail_msg = lc_messages[-1]

        try:
            while tool_round < MAX_TOOL_ROUNDS:
                tool_round += 1
                # Rebuild lc_messages: frozen prefix + tail
                lc_messages = self._cached_prefix + [tail_msg]

                with _timer(f"LLM stream (tool round {tool_round})"):
                    response_text, tool_calls, ai_message, interrupted = self._stream_response(
                        lc_messages, event_waiter, use_tools=True
                    )

                # If interrupted with too little text — discard and retry
                if interrupted and len(response_text.strip()) < MIN_RESPONSE_LENGTH:
                    print(f"[CHAT AGENT] Discarding short interrupted response ({len(response_text.strip())} chars), retrying...")
                    continue

                # Append AI response to cached prefix (frozen for future calls)
                self._cached_prefix.append(ai_message)

                # Collect text from this round
                if response_text.strip():
                    final_text = response_text

                if not tool_calls:
                    break

                print(f"\n[CHAT AGENT] Executing {len(tool_calls)} tool call(s)...")
                last_chat_message = ""
                for tc in tool_calls:
                    tool_name = tc["name"]
                    tool_args = tc.get("args", {})
                    tool_id = tc.get("id", tool_name)
                    if tool_name == "chat" and tool_args.get("message"):
                        last_chat_message = str(tool_args["message"])

                    tools_config._current_tool_call_id = tool_id
                    with _timer(f"tool {tool_name}"):
                        print(f"  -> {tool_name}({tool_args})")
                        result = _execute_tool_call(self.tool_bank, tool_name, tool_args)
                        print(f"  <- {result}")
                    tools_config._current_tool_call_id = None

                    self._cached_prefix.append(ToolMessage(content=str(result), tool_call_id=tool_id))

                # If we have voice text + tool calls done — we're done
                if final_text.strip():
                    break

                # Some models (GLM) only call tools and never write voice text:
                # speak the chat message instead of spending more rounds on it
                if VOICE_CHAT_FALLBACK and last_chat_message:
                    final_text = last_chat_message
                    print("[CHAT AGENT] No voice text, speaking the chat message")
                    break

                if not DISABLE_INTERRUPTS and event_waiter.check():
                    print("[CHAT AGENT] Interrupted by new event")
                    break

                # After tool results, Qwen requires HumanMessage before next assistant turn
                self._cached_prefix.append(
                    HumanMessage(content="Tools executed. Continue or finish with voice *emotion*.")
                )

        except Exception as e:
            print(f"\n[CHAT AGENT] Tool calling error: {e}")
            traceback.print_exc()
            print("[CHAT AGENT] Falling back to plain text mode...")
            lc_messages = self._cached_prefix + [tail_msg]
            final_text = self._respond_plain_text(lc_messages, event_waiter)
            return final_text

        # Send text output as speech
        if final_text.strip():
            self._send_as_speech(final_text)

        return final_text

    def _respond_plain_text(self, lc_messages: List[BaseMessage], event_waiter: SmartEventWaiter) -> str:
        """Fallback: plain text response, sent as speech.
        Does NOT append AI response to prefix (no tool loop to continue)."""
        for attempt in range(MAX_INTERRUPTS_PER_CALL + 1):
            try:
                with _timer(f"LLM stream (plain text, attempt {attempt + 1})"):
                    response_text, _, ai_message, interrupted = self._stream_response(
                        lc_messages, event_waiter, use_tools=False
                    )
                if interrupted and len(response_text.strip()) < MIN_RESPONSE_LENGTH:
                    print(f"[CHAT AGENT] Discarding short interrupted response, retrying...")
                    continue
            except Exception as e:
                print(f"\n[CHAT AGENT] Plain text error: {e}")
                traceback.print_exc()
                return ""

            # Append AI response to prefix for cache continuity
            if ai_message and ai_message.content:
                self._cached_prefix.append(ai_message)

            if response_text.strip():
                self._send_as_speech(response_text)
            return response_text

        return ""

    def _send_as_speech(self, text: str):
        """Parse *emotion* from text, clean up, send to TTS only.

        Chat is handled separately by the LLM via chat tool call.
        """
        try:
            import re
            from agent.tools.base_tools import _parse_emotion, voice_send

            emotion, clean_text = _parse_emotion(text)

            # Validate emotion — if not in allowed set, default to neutral
            if not emotion or emotion not in SpeakEmotions:
                if emotion:
                    print(f"[CHAT AGENT] Invalid emotion '{emotion}', using neutral")
                emotion = "neutral"

            # Strip roleplay markers like *walks away*, *follows player*
            clean_text = re.sub(r"\*[^*]{2,}\*", "", clean_text)

            # Strip command-like lines
            lines = clean_text.split("\n")
            speech_lines = [l for l in lines if not l.strip().startswith(("@", "/", "!"))]
            clean_text = "\n".join(speech_lines).strip()

            # Collapse multiple whitespace/newlines
            clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)
            clean_text = clean_text.strip()

            if not clean_text:
                return
            print(f"[CHAT AGENT] 🟢 TTS: {clean_text} *{emotion}*")
            # NOTE: no _add_tool_run_log here — AI response is already in _cached_prefix.
            # Logging speak to ctx_chat would create a duplicate AIMessage in prefix.
            voice_send([{"text": clean_text, "emotion": emotion}])
        except Exception as e:
            print(f"[CHAT AGENT] Speech send error: {e}")
            traceback.print_exc()

    def _stream_response(
        self,
        messages: List[BaseMessage],
        event_waiter: SmartEventWaiter,
        use_tools: bool = True,
    ) -> tuple:
        """Stream LLM response. Returns (text, tool_calls, ai_message).

        Interrupt logic:
        - Checks event_waiter per chunk, but respects cooldown and max count
        - If interrupted too early (< MIN_RESPONSE_LENGTH chars), the text
          is discarded by the caller so it doesn't pollute conversation history
        """
        response_text = ""
        tool_calls = []
        chunks = []
        was_interrupted = False

        llm = self.llm_with_tools if use_tools else self.llm

        print("[CHAT AGENT] ", end="", flush=True)
        t_first_token = time.time()
        got_first_token = False

        stream_gen = None
        try:
            stream_gen = llm.stream(messages, max_tokens=MAX_RESPONSE_TOKENS)
            for chunk in stream_gen:
                if not got_first_token:
                    ttft = time.time() - t_first_token
                    print(f"[TTFT: {ttft:.2f}s] ", end="", flush=True)
                    got_first_token = True

                chunks.append(chunk)

                if chunk.content:
                    print(chunk.content, end="", flush=True)
                    response_text += chunk.content
                elif use_tools and hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
                    for tc in chunk.tool_call_chunks:
                        if tc.get("name"):
                            print(f"\n[→ {tc['name']}] ", end="", flush=True)
                        if tc.get("args"):
                            print(tc["args"], end="", flush=True)

                # Check interrupt — only if cooldown allows
                if self._can_interrupt() and event_waiter.check():
                    self._record_interrupt()
                    was_interrupted = True
                    print(f"\n[CHAT AGENT] Generation interrupted ({len(response_text)} chars so far)")
                    break

            if not got_first_token:
                elapsed = time.time() - t_first_token
                print(f"[NO TOKENS after {elapsed:.2f}s]", flush=True)

            print()

        except Exception as e:
            print(f"\n[CHAT AGENT] Stream error: {e}")
            traceback.print_exc()
            # Try non-streaming invoke as last resort
            try:
                print("[CHAT AGENT] Trying non-streaming invoke...")
                with _timer("LLM invoke fallback"):
                    response = llm.invoke(messages)
                response_text = response.content or ""
                print(f"[CHAT AGENT] {response_text}")
                if hasattr(response, "tool_calls") and response.tool_calls:
                    tool_calls = response.tool_calls
                return response_text, tool_calls, response, False
            except Exception as e2:
                print(f"[CHAT AGENT] Invoke also failed: {e2}")
                traceback.print_exc()
                return "", [], AIMessage(content=""), False
        finally:
            # CRITICAL: drain remaining stream chunks before closing.
            # LM Studio only saves KV cache when stream is fully consumed.
            # Breaking early without draining = cache miss on next request (~2.5s penalty).
            if stream_gen is not None:
                try:
                    for _ in stream_gen:
                        pass  # discard remaining chunks
                except Exception:
                    pass

        # Merge chunks
        ai_message = self._merge_chunks(chunks, response_text)
        if use_tools and hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
            tool_calls = ai_message.tool_calls

        return response_text, tool_calls, ai_message, was_interrupted

    def _merge_chunks(self, chunks: list, response_text: str) -> AIMessage:
        """Merge streamed chunks into a single AIMessage (not AIMessageChunk).
        AIMessage is required for proper serialization in cached prefix."""
        if not chunks:
            return AIMessage(content=response_text)
        try:
            merged = chunks[0]
            for chunk in chunks[1:]:
                merged = merged + chunk
            # Convert AIMessageChunk → AIMessage for stable serialization
            tc = getattr(merged, "tool_calls", None)
            if tc:
                return AIMessage(content=merged.content or "", tool_calls=tc)
            return AIMessage(content=merged.content or response_text)
        except Exception:
            return AIMessage(content=response_text)

    def _to_langchain_messages(self, messages: List[dict]) -> List[BaseMessage]:
        result = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                result.append(SystemMessage(content=content))
            elif role == "assistant":
                result.append(AIMessage(content=content))
            else:
                result.append(HumanMessage(content=content))
        return result

    def step(self):
        try:
            tracker = self.ctx_handler.activity_tracker
            t_wait_start = time.time()
            activity_level = tracker.level
            print(f"[CHAT AGENT] Waiting for trigger... ({tracker.to_prompt_string()})")

            def extended_wait_check(event: CtxEventBase) -> bool:
                # Unconditional: mentions and directives always wake
                if nick_mention_check(event, SELF_NAME):
                    return True
                if interrupt_event_check(event):
                    return True

                # Queue backlog (existing logic)
                def check_timeout():
                    return time.time() - t_wait_start > 30.0
                if len(self.ctx_handler.ctx_swarm["chat_queue"]) > 1 and not check_timeout():
                    return False
                if len(self.ctx_handler.ctx_swarm["tts_queue"]) > 1 and not check_timeout():
                    return False

                level = tracker.level

                if level == "NO_ACTIVITY":
                    # Starved for input — accept anything
                    return True

                if level == "HIGH_ACTIVITY":
                    # Only kills/deaths (mentions already handled above)
                    return event.type in ("kill", "death")

                # LOW_ACTIVITY — default + flood suppression
                if event.type == "chat" and not event.user:
                    return not tracker.is_flood("server_msg")
                if event.type == "mc_damaged":
                    return not tracker.is_flood("damage")

                return default_event_check(event)

            # Shorten timeout when approaching idle self-activation
            remaining_to_idle = max(0, IDLE_SELF_ACTIVATE - tracker.idle_seconds)
            if remaining_to_idle < 15.0:
                wait_timeout = min(15.0, remaining_to_idle + 1.0)
            else:
                wait_timeout = 15.0

            event: CtxEventBase = self.ctx_handler.wait_for_sync(
                check=extended_wait_check, timeout=wait_timeout, raise_timeout=False
            )

            wait_elapsed = time.time() - t_wait_start
            if event:
                print(f"[CHAT AGENT ⏱️ waited {wait_elapsed:.1f}s] Triggered by: {event.user}: {event.msg[:80]}")
                self.respond(event)
            elif tracker.idle_seconds >= IDLE_SELF_ACTIVATE:
                print(f"[CHAT AGENT ⏱️ waited {wait_elapsed:.1f}s] Idle self-activation ({tracker.idle_seconds:.0f}s idle)")
                self._handle_idle_activation()
            else:
                print(f"[CHAT AGENT ⏱️ waited {wait_elapsed:.1f}s] No trigger (timeout, {activity_level})")

        except Exception as e:
            print(f"[CHAT AGENT] Error: {e}")
            traceback.print_exc()

    def _handle_idle_activation(self):
        """Self-activate when nothing has happened for a while."""
        idle_event = CtxEventBase(
            msg="[Nothing is happening. You are idle and bored.]",
            user="",
            date="",
            processing_timestamp=time.time_ns(),
            type="system",
            env="minecraft",
            additional_fields={
                "filter_results": {"acceptable": True},
                "idle_trigger": True,
            },
        )
        self.ctx_handler.activity_tracker.reset_idle()
        self.respond(idle_event)

    def run(self):
        print("[CHAT AGENT] Starting...")
        self.running = True
        while self.ctx_swarm["env"]["actived"] and self.running:
            self.step()

    def stop(self):
        self.running = False
        print("[CHAT AGENT] Stopped")

    def cleanup(self):
        if self.running:
            self.stop()
        if hasattr(self, "llm"):
            del self.llm
