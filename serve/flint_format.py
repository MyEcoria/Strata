"""serve/flint_format.py - Flint's text for the server: the tool calls it writes, and the <think> it opens with.

Flint (GGUF architecture flint) writes a tool call as (the model's own chat template; every tag is one token)

    <tool_call>get_weather<arg_key>city</arg_key><arg_value>Zagreb</arg_value><arg_key>days</arg_key><arg_value>3</arg_value></tool_call>

A string value is written as it is, every other type as JSON.  And it opens its turn with the reasoning token itself:
`<|assistant|>` ends the prompt and the model writes `<think>`, the reasoning, `</think>` and the answer (or just the answer).

The server's OutputParser (serve/frontend.py) reads the Qwen form `<tool_call><function=NAME><parameter=P>...`, with the
streaming of the arguments, the calls inside the reasoning and its recovery rules, and it expects the reasoning to start
at once (a Qwen prompt already ends with `<think>`).  FlintTranslator rewrites the Flint text into that form as it streams
(holding back only a tag that is still arriving) and tells the parser whether the turn opened with `<think>`;
FlintOutputParser is the OutputParser behind it.  So the OpenAI and Anthropic paths, tool_choice, the streaming arguments
and the reasoning / answer split behave for Flint as they do for the other families (the same design as serve/dsml.py).
"""
from __future__ import annotations

from serve.frontend import Event, OutputParser

THINK = "<think>"
_TAGS = ("<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>")
_NAME_MAX = 400                 # a tool or parameter name longer than this is not held any further


def _tag_at(s: str):
    """The tag `s` starts with -> (tag, None); a start that may still become one -> (None, "wait"); neither -> (None, None)."""
    for t in _TAGS:
        if s.startswith(t):
            return t, None
    for t in _TAGS:
        if t.startswith(s):
            return None, "wait"
    return None, None


class FlintTranslator:
    """Flint's text -> the Qwen tool-call text, a delta at a time.  Text outside a call passes through unchanged.

    on_start(thinking) is called once, with the first text: True when the turn opens with `<think>` (dropped here; the
    parser then starts in the reasoning), False when it does not (the parser starts in the answer)."""

    def __init__(self, on_start=None):
        self.buf = ""
        self.state = "start"       # start | text | name | key | between | value | call
        self.name = ""
        self.key = ""
        self.on_start = on_start
        self.started = False

    @property
    def held(self) -> str:
        """Text held back because it may be the start of a tag."""
        return self.buf

    def feed(self, delta: str) -> str:
        self.buf += delta
        return self._run(final=False)

    def flush(self) -> str:
        """The end of the output: whatever is held is text."""
        return self._run(final=True)

    # ------------------------------------------------------------------------------------------------
    def _begin(self, thinking: bool) -> None:
        self.state = "text"
        if not self.started:
            self.started = True
            if self.on_start is not None:
                self.on_start(thinking)

    def _run(self, final: bool) -> str:
        out: list[str] = []
        while self.buf:
            if self.state == "start":
                s = self.buf.lstrip()
                if s.startswith(THINK):
                    self.buf = s[len(THINK):]
                    self._begin(True)
                    continue
                if (not s or THINK.startswith(s)) and not final:
                    break                                   # still arriving: the first token decides
                self._begin(False)
                continue
            lt = self.buf.find("<")
            if lt < 0:
                self._text(self.buf, out)
                self.buf = ""
                break
            if lt:
                self._text(self.buf[:lt], out)
                self.buf = self.buf[lt:]
            tag, wait = _tag_at(self.buf)
            if wait and not final and len(self.buf) <= len(_TAGS[-1]):
                break
            if tag is None:                                 # an ordinary '<'
                self._text("<", out)
                self.buf = self.buf[1:]
                continue
            self.buf = self.buf[len(tag):]
            self._tag(tag, out)
        if final and self.state == "start":
            self._begin(False)
        if final:
            self._close(out)
        return "".join(out)

    def _text(self, text: str, out: list[str]) -> None:
        if self.state in ("text", "value"):
            out.append(text)
        elif self.state == "name":
            self.name += text
        elif self.state == "key":
            self.key += text
        # between / call: the layout between tags (stray text there is dropped: the call goes on)

    def _open(self, out: list[str]) -> None:
        """The call's name is known: its opening in the Qwen form."""
        out.append(f"<tool_call>\n<function={self.name.strip()}>\n")

    def _end_value(self, out: list[str]) -> None:
        out.append("\n</parameter>\n")
        self.state = "call"

    def _close(self, out: list[str]) -> None:
        """The output ends inside a call: close what is open."""
        if self.state == "value":
            self._end_value(out)
        if self.state in ("name", "key", "between", "call", "value"):
            if self.state == "name":
                self._open(out)
            out.append("</function>\n</tool_call>\n")
            self.state = "text"

    def _tag(self, tag: str, out: list[str]) -> None:
        st = self.state
        if tag == "<tool_call>":
            if st in ("text",):
                self.state, self.name, self.key = "name", "", ""
            elif st in ("name", "key", "between", "value", "call"):   # the previous call's closing tag was left out
                self._close(out)
                self.state, self.name, self.key = "name", "", ""
            return
        if tag == "</tool_call>":
            if st in ("name", "key", "between", "value", "call"):
                self._close(out)
            else:
                out.append(tag)
            return
        if tag == "<arg_key>":
            if st == "name":
                self._open(out)
                self.state, self.key = "key", ""
            elif st in ("call", "between"):
                self.state, self.key = "key", ""
            elif st == "value":                              # the value's closing tag was left out
                self._end_value(out)
                self.state, self.key = "key", ""
            else:
                out.append(tag)
            return
        if tag == "</arg_key>":
            if st == "key":
                self.state = "between"
            else:
                out.append(tag) if st in ("text", "value") else None
            return
        if tag == "<arg_value>":
            if st in ("between", "key"):
                out.append(f"<parameter={self.key.strip()}>\n")
                self.state = "value"
            else:
                out.append(tag) if st in ("text", "value") else None
            return
        if tag == "</arg_value>":
            if st == "value":
                self._end_value(out)
            else:
                out.append(tag) if st == "text" else None
            return


class FlintOutputParser(OutputParser):
    """OutputParser for Flint: the model's tool calls are read as the template-form calls they are, and the turn starts
    in the reasoning only when it opens with `<think>`."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flint = FlintTranslator(self._on_start)
        # A turn that stops (EOS) inside the reasoning with no answer and no call: the text is the answer, as for DeepSeek.
        self._reasoning: list[str] = []
        self._answered = False

    def _on_start(self, thinking: bool) -> None:
        self.state = "reasoning" if thinking else "content"

    def _watch(self, evs):
        for ev in evs:
            if ev.kind == "reasoning":
                self._reasoning.append(ev.text)
            elif ev.kind in ("tool_start", "tool_call") or (ev.kind == "content" and ev.text.strip()):
                self._answered = True
        return evs

    def feed(self, delta: str):
        return self._watch(super().feed(self.flint.feed(delta)))

    def finish(self, reason: str | None = None):
        rest = self.flint.flush()
        out = self._watch(super().feed(rest) if rest else [])
        out += self._watch(super().finish(reason))
        text = "".join(self._reasoning).strip()
        if reason in (None, "stop") and text and not self._answered:
            self._answered = True
            out.append(Event("content", text))
        return out

    @property
    def clean(self) -> bool:
        """No tag is held back anywhere: a clean point to stop the thinking at."""
        return not self.buf and not self.flint.held


def to_flint_force(force: str | None) -> str | None:
    """forced_call's opening of a call in the Qwen form (`<tool_call>\\n<function=NAME>\\n`, or cut after the name) -> the
    same opening in Flint's spelling (`<tool_call>NAME`), so the model goes on with the first argument."""
    if not force:
        return force
    marker = "<function="
    at = force.find(marker)
    name = force[at + len(marker):] if at >= 0 else ""
    if name.endswith(">\n"):
        name = name[:-2]
    return "<tool_call>" + name
