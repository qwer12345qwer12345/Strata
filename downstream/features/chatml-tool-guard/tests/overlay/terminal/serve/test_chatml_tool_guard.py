import json
import unittest
import urllib.request

from serve.server import ByteTokenizer, ChatTemplate, MockEngine, Service, serve
from serve import test_server as fixtures

ROOT = fixtures.ROOT
UnfinishedToolCall = fixtures.UnfinishedToolCall

MARKER = '<|im_start|>'
TOOLS = [{'type': 'function', 'function': {'name': 'write', 'parameters': {
    'type': 'object', 'properties': UnfinishedToolCall.PROPS}}}]


class ChatMLToolGuard(unittest.TestCase):
    def replay(self, script, *, plain_marker=False, guard=True, tools=TOOLS, max_tokens=768, stream=True, tokenizer=None):
        tok = tokenizer or ByteTokenizer()
        engine = MockEngine(tok, script)
        if plain_marker:
            engine.script = tok.encode(script, parse_special=False) + tok.encode('<|im_end|>', parse_special=True)
        svc = Service(engine, tok, ChatTemplate(ROOT / 'serve/chat_template.jinja'))
        svc.chatml_tool_guard = guard
        svc.undeclared_tool_calls = 'call'
        httpd = serve(svc, port=0)
        try:
            body = dict(messages=[{'role': 'user', 'content': 'Save my notes'}],
                        tools=tools, model='x', stream=stream, max_tokens=max_tokens,
                        chat_template_kwargs={'enable_thinking': False})
            req = urllib.request.Request(f'http://127.0.0.1:{httpd.server_port}/v1/chat/completions',
                                         data=json.dumps(body).encode(),
                                         headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=10) as r:
                raw = r.read()
        finally:
            httpd.shutdown()
            httpd.server_close()
        if not stream:
            return json.loads(raw)['choices']
        return [choice for l in raw.splitlines() if l.startswith(b'data: {')
                for choice in json.loads(l[6:])['choices']]

    def test_real_role_token_does_not_escape_in_agent_reply(self):
        reply = self.replay('preamble' + MARKER + 'user\nwrong')
        self.assertEqual(''.join(c['delta'].get('content') or '' for c in reply), 'preamble')
        self.assertEqual(reply[-1]['finish_reason'], 'stop')

    def test_role_token_in_truncated_tool_does_not_publish_a_tool(self):
        partial = UnfinishedToolCall.CUT.removeprefix('</think>\n\n')
        for stream in (True, False):
            with self.subTest(stream=stream):
                reply = self.replay(partial + MARKER + 'assistant\nwrong', stream=stream)
                self.assertFalse(any(c.get('delta', c.get('message', {})).get('tool_calls') for c in reply))
                self.assertEqual(reply[-1]['finish_reason'], 'length')

    def test_complete_tool_before_role_token_keeps_tool_finish(self):
        good = UnfinishedToolCall.WHOLE.removeprefix('</think>\n\n')
        reply = self.replay(good + MARKER + 'user\nwrong')
        calls = [tc for c in reply for tc in c['delta'].get('tool_calls', [])]
        self.assertEqual(len(calls), 1)
        self.assertEqual(reply[-1]['finish_reason'], 'tool_calls')

    def test_plain_marker_in_valid_file_contents_is_preserved(self):
        good = UnfinishedToolCall.WHOLE.removeprefix('</think>\n\n').replace('all of it', MARKER)
        reply = self.replay(good, plain_marker=True)
        calls = [tc for c in reply for tc in c['delta'].get('tool_calls', [])]
        self.assertEqual(len(calls), 1)
        self.assertEqual(json.loads(calls[0]['function']['arguments'])['content'], MARKER)
        self.assertEqual(reply[-1]['finish_reason'], 'tool_calls')

    def test_undeclared_complete_call_preserved(self):
        good = UnfinishedToolCall.WHOLE.removeprefix('</think>\n\n').replace('function=write', 'function=deferred_write')
        reply = self.replay(good)
        calls = [tc for c in reply for tc in c['delta'].get('tool_calls', [])]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]['function']['name'], 'deferred_write')
        self.assertEqual(json.loads(calls[0]['function']['arguments'])['path'], 'notes.txt')

    def test_agent_token_limit_does_not_publish_partial_json(self):
        partial = UnfinishedToolCall.CUT.removeprefix('</think>\n\n')
        reply = self.replay(partial + 'more text', max_tokens=len(partial))
        self.assertFalse(any(c['delta'].get('tool_calls') for c in reply))
        self.assertEqual(reply[-1]['finish_reason'], 'length')

    def test_no_tools_keeps_literal_output(self):
        reply = self.replay(MARKER, tools=[])
        self.assertEqual(''.join(c['delta'].get('content') or '' for c in reply), MARKER)

    def test_disabled_guard_preserves_role_output(self):
        script = 'preamble' + MARKER + 'user\nnext turn'
        reply = self.replay(script, guard=False)
        self.assertEqual(''.join(c['delta'].get('content') or '' for c in reply), script)

    def test_disabled_guard_preserves_incremental_tool_stream(self):
        good = UnfinishedToolCall.WHOLE.removeprefix('</think>\n\n')
        reply = self.replay(good, guard=False)
        calls = [tc for c in reply for tc in c['delta'].get('tool_calls', [])]
        self.assertGreater(len(calls), 1)
        args = ''.join(tc.get('function', {}).get('arguments', '') for tc in calls)
        self.assertEqual(json.loads(args)['path'], 'notes.txt')
        self.assertEqual(reply[-1]['finish_reason'], 'tool_calls')

    def test_complete_nonstream_tool_before_boundary(self):
        good = UnfinishedToolCall.WHOLE.removeprefix('</think>\n\n')
        reply = self.replay(good + MARKER + 'user\nnext turn', stream=False)
        calls = reply[0]['message']['tool_calls']
        self.assertEqual(len(calls), 1)
        self.assertEqual(json.loads(calls[0]['function']['arguments'])['path'], 'notes.txt')
        self.assertEqual(reply[0]['finish_reason'], 'tool_calls')


if __name__ == '__main__':
    unittest.main()
