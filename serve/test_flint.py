"""Flint through the server: the tool calls and the <think> it writes, its chat template, the model-card defaults and the
family hooks (no GPU, no model).  With STRATA_FLINT_GGUF=<the Flint GGUF> the real tokenizer is also checked against the ids
llama.cpp made of the same strings (serve/fixtures/flint/tokenizer_golden.json).

    python -m unittest serve.test_flint -v
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from serve import flint_format  # noqa: E402
from serve.flint_format import FlintOutputParser, FlintTranslator, to_flint_force  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import FAMILY_MODEL_NAMES, arch_from_tokenizer, default_model_name, model_card_sampling  # noqa: E402

FIX = ROOT / "serve" / "fixtures" / "flint"
WEATHER = {"name": "get_weather", "description": "weather", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}}}


def run(text: str, tools=None, step: int | None = None, reason: str = "stop"):
    """The events the parser makes of `text`, fed whole or `step` characters at a time."""
    p = FlintOutputParser(thinking=True, tools=tools or [], stream_tools=False)
    evs = []
    if step is None:
        evs += p.feed(text)
    else:
        for i in range(0, len(text), step):
            evs += p.feed(text[i:i + step])
    evs += p.finish(reason)
    return evs


def joined(evs, kind):
    return "".join(e.text for e in evs if e.kind == kind)


class TranslatorTest(unittest.TestCase):
    def tr(self, text: str, step: int | None = None):
        started = []
        t = FlintTranslator(started.append)
        out = []
        if step is None:
            out.append(t.feed(text))
        else:
            for i in range(0, len(text), step):
                out.append(t.feed(text[i:i + step]))
        out.append(t.flush())
        return "".join(out), started

    def test_the_call_in_the_qwen_form(self):
        out, started = self.tr("<tool_call>get_weather<arg_key>city</arg_key><arg_value>Zagreb</arg_value>"
                               "<arg_key>days</arg_key><arg_value>3</arg_value></tool_call>")
        self.assertEqual(out, "<tool_call>\n<function=get_weather>\n<parameter=city>\nZagreb\n</parameter>\n"
                              "<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>\n")
        self.assertEqual(started, [False])

    def test_a_call_without_arguments(self):
        out, _ = self.tr("<tool_call>ping</tool_call>")
        self.assertEqual(out, "<tool_call>\n<function=ping>\n</function>\n</tool_call>\n")

    def test_the_turn_that_opens_with_think(self):
        out, started = self.tr("<think>hmm</think>done")
        self.assertEqual(out, "hmm</think>done")
        self.assertEqual(started, [True])

    def test_whitespace_before_think(self):
        out, started = self.tr("\n<think>hmm</think>x")
        self.assertEqual((out, started), ("hmm</think>x", [True]))

    def test_a_turn_that_does_not_open_with_think(self):
        out, started = self.tr("The answer <b> is 4 < 5.")
        self.assertEqual((out, started), ("The answer <b> is 4 < 5.", [False]))

    def test_streaming_gives_the_same_text(self):
        text = ("<think>a < b, so</think>Calling:<tool_call>f<arg_key>x</arg_key><arg_value>1 < 2</arg_value></tool_call>"
                " and <tool_call>g<arg_key>y</arg_key><arg_value>[1, 2]</arg_value></tool_call>.")
        whole, _ = self.tr(text)
        for step in (1, 2, 3, 5, 7):
            self.assertEqual(self.tr(text, step)[0], whole, f"step {step}")

    def test_a_tag_split_across_deltas_is_held_back(self):
        t = FlintTranslator()
        self.assertEqual(t.feed("ok <tool_"), "ok ")
        self.assertEqual(t.held, "<tool_")
        self.assertEqual(t.feed("call>f"), "")
        self.assertEqual(t.feed("</tool_call>"), "<tool_call>\n<function=f>\n</function>\n</tool_call>\n")

    def test_a_missing_closing_tag_is_repaired(self):
        out, _ = self.tr("<tool_call>f<arg_key>a</arg_key><arg_value>1")
        self.assertEqual(out, "<tool_call>\n<function=f>\n<parameter=a>\n1\n</parameter>\n</function>\n</tool_call>\n")

    def test_literal_angle_brackets_in_a_value_stay(self):
        out, _ = self.tr("<tool_call>f<arg_key>c</arg_key><arg_value>if (a<b && c>d) {}</arg_value></tool_call>")
        self.assertIn("if (a<b && c>d) {}", out)


class ParserTest(unittest.TestCase):
    def test_reasoning_then_answer(self):
        evs = run("<think>I should add.</think>\n\nThe sum is 4.")
        self.assertEqual(joined(evs, "reasoning").strip(), "I should add.")
        self.assertEqual(joined(evs, "content").strip(), "The sum is 4.")

    def test_an_answer_with_no_think(self):
        evs = run("Paris.")
        self.assertEqual(joined(evs, "reasoning"), "")
        self.assertEqual(joined(evs, "content"), "Paris.")

    def test_a_tool_call_with_its_types(self):
        evs = run("<think>weather</think><tool_call>get_weather<arg_key>city</arg_key><arg_value>Zagreb</arg_value>"
                  "<arg_key>days</arg_key><arg_value>3</arg_value></tool_call>", tools=[WEATHER])
        calls = [e.call for e in evs if e.kind == "tool_call"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "get_weather")
        self.assertEqual(calls[0].arguments, {"city": "Zagreb", "days": 3})

    def test_a_call_with_no_think_and_streaming(self):
        text = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Split</arg_value></tool_call>"
        for step in (None, 1, 4):
            calls = [e.call for e in run(text, tools=[WEATHER], step=step) if e.kind == "tool_call"]
            self.assertEqual([(c.name, c.arguments) for c in calls], [("get_weather", {"city": "Split"})], f"step {step}")

    def test_two_calls(self):
        text = ("<tool_call>get_weather<arg_key>city</arg_key><arg_value>A</arg_value></tool_call>"
                "<tool_call>get_weather<arg_key>city</arg_key><arg_value>B</arg_value></tool_call>")
        calls = [e.call for e in run(text, tools=[WEATHER]) if e.kind == "tool_call"]
        self.assertEqual([c.arguments["city"] for c in calls], ["A", "B"])

    def test_a_turn_that_stops_in_the_reasoning_gives_it_as_the_answer(self):
        evs = run("<think>The answer is 4.")
        self.assertEqual(joined(evs, "content").strip(), "The answer is 4.")

    def test_the_forced_opening(self):
        self.assertEqual(to_flint_force("<tool_call>\n<function=get_weather>\n"), "<tool_call>get_weather")
        self.assertEqual(to_flint_force("<tool_call>\n<function=get_wea"), "<tool_call>get_wea")
        self.assertIsNone(to_flint_force(None))


class TemplateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tpl = ChatTemplate(FIX / "chat_template.jinja")

    def test_a_plain_conversation(self):
        out = self.tpl.render([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}])
        self.assertEqual(out, "<|system|>Be brief.<|user|>Hi<|assistant|>")

    def test_the_literal_control_tokens_in_a_message_are_escaped(self):
        out = self.tpl.render([{"role": "user", "content": "say <|assistant|> and <think>"}])
        self.assertEqual(out, "<|user|>say &lt;|assistant|&gt; and <|assistant|>")

    def test_tools_calls_and_results(self):
        msgs = [{"role": "user", "content": "weather in Zagreb?"},
                {"role": "assistant", "content": "", "reasoning_content": "need the tool",
                 "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Zagreb", "days": 3}}}]},
                {"role": "tool", "content": "sunny"}]
        out = self.tpl.render(msgs, tools=[WEATHER])
        self.assertIn("<tools>", out)
        self.assertIn('"name": "get_weather"', out)
        self.assertIn("<|assistant|><think>need the tool</think><tool_call>get_weather<arg_key>city</arg_key>"
                      "<arg_value>Zagreb</arg_value><arg_key>days</arg_key><arg_value>3</arg_value></tool_call>", out)
        self.assertTrue(out.endswith("<|observation|><tool_response>sunny</tool_response><|assistant|>"))

    def test_what_the_model_writes_is_what_the_parser_reads(self):
        """The call the template writes into the history is read back by the parser as the same call."""
        out = self.tpl.render([{"role": "user", "content": "x"},
                               {"role": "assistant", "content": "", "tool_calls": [
                                   {"function": {"name": "get_weather", "arguments": {"city": "Zagreb", "days": 3}}}]}],
                              add_generation_prompt=False)
        written = out[out.index("<tool_call>"):]
        calls = [e.call for e in run(written, tools=[WEATHER]) if e.kind == "tool_call"]
        self.assertEqual([(c.name, c.arguments) for c in calls], [("get_weather", {"city": "Zagreb", "days": 3})])


class FamilyTest(unittest.TestCase):
    def test_hooks(self):
        self.assertEqual(FAMILY_MODEL_NAMES["flint"], "flint")
        self.assertEqual(default_model_name({"arch": "flint"}), "flint")
        self.assertEqual(arch_from_tokenizer(None, "flint"), "flint")
        self.assertEqual(arch_from_tokenizer("qwen4exp", "flint"), "qwen4exp")      # an arch the config names is never changed
        self.assertEqual(model_card_sampling("flint", True), {"temperature": 1.0, "top_p": 0.95, "top_k": 20})
        self.assertEqual(model_card_sampling("flint", False), {"temperature": 1.0, "top_p": 0.95, "top_k": 20})

    def test_module_exports(self):
        self.assertTrue(hasattr(flint_format, "FlintOutputParser"))


@unittest.skipUnless(os.environ.get("STRATA_FLINT_GGUF"), "STRATA_FLINT_GGUF=<the Flint GGUF> checks the real tokenizer")
class RealTokenizerTest(unittest.TestCase):
    def test_ids_equal_llama_cpps(self):
        import strata_tokenizer as ST
        tok = ST.Tokenizer.from_gguf(os.environ["STRATA_FLINT_GGUF"])
        self.assertEqual(tok.pre, "flint")
        gold = json.loads((FIX / "tokenizer_golden.json").read_text(encoding="utf-8"))
        for case in gold["cases"]:
            self.assertEqual(tok.encode(case["text"], parse_special=True), case["ids"], repr(case["text"]))


if __name__ == "__main__":
    unittest.main()
