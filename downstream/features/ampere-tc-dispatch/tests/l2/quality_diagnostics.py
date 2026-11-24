"""Synthetic quality diagnostics for the explicit Tail Skip x ADAPT_FETCH study.
Repetition is reported, never treated as failure by itself. Token IDs are preserved only if the API supplies them.
"""
import ast
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import resource
import shutil
import subprocess
import tempfile

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
CODE_PROMPT = (
    "Implement one Python standard-library module for a durable key-value store. "
    "Return exactly one fenced Python code block, with no prose outside it. Define "
    "KVStore(path, max_entries=128), set(key, value, ttl_seconds=None), get(key, default=None), "
    "delete(key) returning bool, and transaction() as a context manager yielding an object with "
    "set/get/delete. Make it thread-safe, persist values in an append-only newline-delimited journal, "
    "recover complete valid records while ignoring a malformed or partial trailing record, expire TTLs "
    "across process restarts, and evict least-recently-used entries after max_entries. Include at least "
    "eight real unittest cases runnable with python candidate.py; no placeholders, TODOs, or skipped tests."
)
TOOL_SCHEMA = [{
    "type": "function",
    "function": {
        "name": "create_reservation",
        "description": "Create a mock reservation; it is not connected to any external system.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "enum": ["Oslo", "Kyoto", "Lima"]},
                "check_in": {"type": "string", "enum": ["2027-05-11", "2027-05-18"]},
                "nights": {"type": "integer", "enum": [2, 3, 4]},
                "cancellation": {"type": "string", "enum": ["flexible", "nonrefundable"]},
            },
            "required": ["city", "check_in", "nights", "cancellation"],
            "additionalProperties": False,
        },
    },
}]
TOOL_COLD_EXPECTED = {"city":"Kyoto","check_in":"2027-05-18","nights":3,"cancellation":"flexible"}
TOOL_HOT_PREVIOUS = {"city":"Oslo","check_in":"2027-05-11","nights":2,"cancellation":"nonrefundable"}
TOOL_HOT_EXPECTED = {"city":"Lima","check_in":"2027-05-18","nights":4,"cancellation":"flexible"}
TOOL_COLD_PROMPT = (
    "Choose exactly one create_reservation tool call for a mock trip to Kyoto, checking in "
    "2027-05-18 for 3 nights with flexible cancellation. Do not write explanatory text and do not execute it."
)
TOOL_HOT_PROMPT = (
    "This is a new turn in the same mock conversation. Make exactly one new reservation for Lima, "
    "checking in 2027-05-18 for 4 nights with flexible cancellation. Do not repeat or modify the "
    "previous reservation. Do not write explanatory text and do not execute the call."
)

FIXED_TESTS = r'''
import sys
sys.path.insert(0, "/work")
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from candidate import KVStore

class FixedBehavior(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmp.name) / "store.jsonl")
    def tearDown(self):
        self.tmp.cleanup()
    def test_persists_across_reopen(self):
        s = KVStore(self.path); s.set("alpha", {"v": 3})
        self.assertEqual(KVStore(self.path).get("alpha"), {"v": 3})
    def test_ttl_expires(self):
        s = KVStore(self.path); s.set("short", "value", ttl_seconds=0.08)
        time.sleep(0.12); self.assertIsNone(s.get("short"))
    def test_lru_evicts_least_recently_used(self):
        s = KVStore(self.path, max_entries=2); s.set("a", 1); s.set("b", 2)
        self.assertEqual(s.get("a"), 1); s.set("c", 3)
        self.assertIsNone(s.get("b")); self.assertEqual((s.get("a"), s.get("c")), (1, 3))
    def test_delete_reports_presence(self):
        s = KVStore(self.path); s.set("x", 1)
        self.assertTrue(s.delete("x")); self.assertFalse(s.delete("x"))
    def test_transaction_commits_all_updates(self):
        s = KVStore(self.path)
        with s.transaction() as tx: tx.set("a", 1); tx.set("b", 2)
        self.assertEqual((s.get("a"), s.get("b")), (1, 2))
    def test_transaction_rolls_back_on_exception(self):
        s = KVStore(self.path); s.set("stable", 1)
        with self.assertRaises(RuntimeError):
            with s.transaction() as tx:
                tx.set("stable", 2); tx.set("new", 3); raise RuntimeError("rollback")
        self.assertEqual(s.get("stable"), 1); self.assertIsNone(s.get("new"))
    def test_threaded_writes_are_visible(self):
        s = KVStore(self.path, max_entries=64)
        def write(i):
            s.set("k" + str(i), i); return s.get("k" + str(i))
        with ThreadPoolExecutor(max_workers=8) as pool:
            self.assertEqual(list(pool.map(write, range(32))), list(range(32)))
    def test_partial_trailing_record_preserves_prior_values(self):
        s = KVStore(self.path); s.set("safe", "yes")
        with open(self.path, "ab") as f: f.write(b'{"partial":')
        self.assertEqual(KVStore(self.path).get("safe"), "yes")

if __name__ == "__main__":
    unittest.main()
'''

def objective_workloads():
    faults = ("checksum mismatch", "stale index", "clock skew", "orphaned segment")
    rows = [
        f"Record {20000+i}, shard {(i*101+7)%4093}; fault={faults[i%4]}; "
        f"journal offset {(i*3301+17)%1000003}; operator {(i*3)%11}; page {(i*17)%2000}."
        for i in range(160)
    ]
    probes = []
    prior_call_id = "call_synthetic_previous"
    prior_history = [
        {"role":"assistant","content":None,"tool_calls":[{
            "id":prior_call_id,"type":"function","function":{
                "name":"create_reservation",
                "arguments":json.dumps(TOOL_HOT_PREVIOUS,sort_keys=True)}}]},
        {"role":"tool","tool_call_id":prior_call_id,"name":"create_reservation",
         "content":json.dumps({"mock":"fixture only","executed":False})},
    ]
    tool_options = {"tools":TOOL_SCHEMA,
        "tool_choice":{"type":"function","function":{"name":"create_reservation"}}}
    for i in range(1, 9):
        probes.append((f"agent_tool_cold_{i:02d}", TOOL_COLD_PROMPT, 192, tool_options))
        probes.append((f"agent_tool_hot_{i:02d}", TOOL_HOT_PROMPT, 192,
                       {**tool_options,"messages":prior_history+[{"role":"user","content":TOOL_HOT_PROMPT}]}))
    return [
        ("kv_store", CODE_PROMPT, 6144, {}),
        ("math_logic",
         'Return JSON only with keys "liters" and "equation". A tank starts with 18 liters, '
         'leaks 2.5 liters per hour for 4 hours, then receives 7 liters. Give the exact final amount '
         'and arithmetic equation.', 192, {}),
        ("long_needle",
         "Find Record 20117 in these synthetic incident records and return JSON only with keys "
         "record_id, shard, fault, journal_offset, operator, page. Copy each value exactly.\n"
         + "\n".join(rows), 256, {}),
        *probes,
    ]


def _tokens(text):
    return TOKEN_RE.findall((text or "").casefold())

def repetition_metrics(text):
    tokens = _tokens(text)
    n = len(tokens)
    if not n:
        return {"tokenization": "Unicode word/punctuation tokens from text; not model token IDs",
                "text_token_count": 0, "max_consecutive_same_token": 0,
                "repeat_coverage_ratio": 0.0, "first_repeat_token_offset": None,
                "longest_repeat": None, "candidate_count": 0}
    max_run = 1
    i = 0
    while i < n:
        j = i + 1
        while j < n and tokens[j] == tokens[i]: j += 1
        max_run = max(max_run, j-i); i = j
    intervals = []
    for period in range(1, min(32, n//4) + 1):
        i = 0
        while i + 2*period <= n:
            if tokens[i:i+period] == tokens[i+period:i+2*period]:
                end = i + 2*period
                while end+period <= n and tokens[end:end+period] == tokens[end-period:end]: end += period
                cycles = (end-i)//period
                if cycles >= 4 and end-i >= 16: intervals.append((i,end,period,cycles))
                i += max(1, end-i-period)
            else: i += 1
    gram = min(64, n//2)
    if gram >= 32:
        seen = {}
        for i in range(n-gram+1):
            key = tuple(tokens[i:i+gram])
            previous = seen.get(key)
            if previous is None:
                seen[key] = i; continue
            end = i + gram
            while end < n and previous+(end-i) < n and tokens[end] == tokens[previous+(end-i)]: end += 1
            intervals.append((previous,end,gram,max(2,(end-previous)//gram)))
    merged = []
    for start,end,period,cycles in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else: merged.append((start,end))
    coverage = sum(end-start for start,end in merged)
    longest = max(intervals, key=lambda x:x[1]-x[0]) if intervals else None
    detail = None
    if longest:
        start,end,period,cycles = longest
        detail = {"start_token":start, "end_token_exclusive":end, "span_tokens":end-start,
                  "period_tokens":period, "cycles":cycles,
                  "sample":" ".join(tokens[start:min(end,start+80)])[:600]}
    return {"tokenization":"Unicode word/punctuation tokens from text; not model token IDs",
            "text_token_count":n, "max_consecutive_same_token":max_run,
            "repeat_coverage_ratio":round(coverage/n,6), "repeat_coverage_tokens":coverage,
            "first_repeat_token_offset":min((a for a,_ in merged),default=None),
            "longest_repeat":detail, "candidate_count":len(intervals)}

def _parse_json(text):
    text = (text or "").strip(); start = text.find("{"); end = text.rfind("}")
    if start < 0 or end < start: return None
    try: return json.loads(text[start:end+1])
    except (ValueError, TypeError): return None

def _resource_limits():
    resource.setrlimit(resource.RLIMIT_CPU,(6,6))
    resource.setrlimit(resource.RLIMIT_AS,(512*1024*1024,512*1024*1024))
    resource.setrlimit(resource.RLIMIT_FSIZE,(2*1024*1024,2*1024*1024))
    resource.setrlimit(resource.RLIMIT_NOFILE,(64,64))

def _candidate_tests(content):
    fence = chr(96)*3
    match = re.search(fence+r"python\s*(.*?)"+fence, content or "", re.S|re.I)
    if not match: return {"passed":False,"reason":"missing one fenced Python implementation"}
    source = match.group(1)
    if len(source.encode("utf-8")) > 100_000:
        return {"passed":False,"reason":"candidate source exceeded the 100 KiB safety limit"}
    try: ast.parse(source)
    except SyntaxError as exc: return {"passed":False,"reason":f"candidate syntax error at line {exc.lineno}"}
    bwrap, python = shutil.which("bwrap"), "/usr/bin/python3"
    if not bwrap or not Path(python).is_file():
        return {"passed":False,"reason":"isolated sandbox runtime unavailable","sandbox":"not_run"}
    with tempfile.TemporaryDirectory(prefix="strata-quality-") as temp:
        root = Path(temp)
        (root/"candidate.py").write_text(source)
        (root/"fixed_tests.py").write_text(FIXED_TESTS)
        base = [bwrap,"--unshare-all","--die-with-parent"]
        for path in ("/usr","/bin","/lib","/lib64"):
            if Path(path).exists(): base += ["--ro-bind",path,path]
        base += ["--dev","/dev","--proc","/proc","--tmpfs","/tmp","--bind",temp,"/work",
                 "--chdir","/work","--clearenv","--setenv","PATH","/usr/bin:/bin",
                 "--setenv","HOME","/work","--setenv","PYTHONDONTWRITEBYTECODE","1","--"]
        commands = [
            ['-c','import sys,runpy;sys.path.insert(0,"/work");runpy.run_path("/work/candidate.py",run_name="__main__")'],
            ['-c','import sys,runpy;sys.path.insert(0,"/work");runpy.run_path("/work/fixed_tests.py",run_name="__main__")'],
        ]
        results = []
        for args in commands:
            try:
                r = subprocess.run(base+[python,"-I"]+args, capture_output=True, text=True, timeout=12,
                                   preexec_fn=_resource_limits, env={"PATH":"/usr/bin:/bin","HOME":"/work"})
                results.append({"exit_code":r.returncode,"stdout_tail":r.stdout[-3000:],
                                "stderr_tail":r.stderr[-3000:]})
            except (subprocess.TimeoutExpired,OSError) as exc:
                results.append({"exit_code":None,"error":type(exc).__name__})
    def count(result):
        m = re.search(r"Ran (\d+) tests?",result.get("stderr_tail","")+result.get("stdout_tail",""))
        return int(m.group(1)) if m else None
    candidate_count = count(results[0]) if results else None
    fixed_count = count(results[1]) if len(results)>1 else None
    passed = (len(results)==2 and all(r.get("exit_code")==0 for r in results)
              and candidate_count is not None and candidate_count>=8 and fixed_count==8)
    return {"passed":passed,
            "sandbox":"bubblewrap: isolated user/mount/network namespaces; read-only system; temporary writable workspace",
            "candidate_tests":candidate_count,"independent_fixed_tests":fixed_count,"runs":results}

def _tool_evaluation(tool_calls, hot=False):
    if len(tool_calls or []) != 1:
        return {"passed":False,"reason":f"expected exactly one new tool call, observed {len(tool_calls or [])}"}
    fn = tool_calls[0].get("function",{})
    try: args = json.loads(fn.get("arguments",""))
    except (ValueError,TypeError): return {"passed":False,"reason":"tool arguments were not valid JSON"}
    expected = TOOL_HOT_EXPECTED if hot else TOOL_COLD_EXPECTED
    return {"passed":fn.get("name")=="create_reservation" and args==expected,
            "tool_name":fn.get("name"),"arguments":args,"expected":expected,"executed":False,
            "context":"synthetic prior tool call and fixture-only tool result" if hot else "cold single-turn"}

def _math_equation_matches(equation):
    if not isinstance(equation, str):
        return False
    sides = equation.split("=")
    if len(sides) not in (1, 2):
        return False
    try:
        actual = ast.parse(sides[0].strip(), mode="eval").body
        expected = ast.parse("18 - 2.5 * 4 + 7", mode="eval").body
        result = ast.parse(sides[1].strip(), mode="eval").body if len(sides) == 2 else None
    except (SyntaxError, TypeError, ValueError):
        return False
    if ast.dump(actual, include_attributes=False) != ast.dump(expected, include_attributes=False):
        return False
    if result is None:
        return True
    return (isinstance(result, ast.Constant) and
            isinstance(result.value, (int, float)) and not isinstance(result.value, bool) and
            math.isclose(float(result.value), 15.0))


def analyze_case(row):
    content = row.get("content") or ""; case = row.get("case")
    task = {"applicable":False,"passed":None}
    if case == "kv_store": task = {"applicable":True,**_candidate_tests(content)}
    elif case == "math_logic":
        answer = _parse_json(content)
        try: valid_liters = math.isclose(float(answer.get("liters")),15.0) if answer else False
        except (ValueError,TypeError): valid_liters = False
        valid_equation = _math_equation_matches((answer or {}).get("equation"))
        task = {"applicable":True,"passed":valid_liters and valid_equation,"answer":answer,
                "expected":{"liters":15,"equation":"18 - 2.5 * 4 + 7"}}
    elif case == "long_needle":
        i=117
        expected={"record_id":20000+i,"shard":(i*101+7)%4093,
                  "fault":("checksum mismatch","stale index","clock skew","orphaned segment")[i%4],
                  "journal_offset":(i*3301+17)%1000003,"operator":(i*3)%11,"page":(i*17)%2000}
        answer = _parse_json(content)
        task={"applicable":True,"passed":answer==expected,"answer":answer,"expected":expected}
    elif case.startswith("agent_tool_"):
        task = {"applicable":True,**_tool_evaluation(row.get("tool_calls") or [],
                                                    hot=case.startswith("agent_tool_hot_"))}
    return {"content_repetition":repetition_metrics(content),
            "reasoning_repetition":repetition_metrics(row.get("reasoning_content") or ""),
            "task_evaluation":task,"generated_token_ids":row.get("generated_token_ids"),
            "token_id_source":row.get("token_id_source"),"finish_reason":row.get("finish"),
            "generation_failure_count":row.get("generation_failure_count",0),
            "generation_recovery_count":row.get("generation_recovery_count",0)}

def _output_tokens(row):
    text=row.get("content") or ""
    if row.get("tool_calls"): text+=json.dumps(row["tool_calls"],sort_keys=True,ensure_ascii=False)
    return _tokens(text)

def compare_runs(runs):
    selected=[r for r in runs if r.get("settings",{}).get("group")=="tail-fetch"]
    baseline=next((r for r in selected if r.get("settings",{}).get("tail")==0
                   and r.get("settings",{}).get("fetch")==0 and r.get("cases")),None)
    tool_probes=defaultdict(lambda: {"total":0,"cold":0,"hot_history":0,"passed":0})
    for run in selected:
        s=run.get("settings",{}); arm=f"tail={s.get('tail')} fetch={s.get('fetch')}"
        for row in run.get("cases",[]):
            case=row.get("case","")
            if not case.startswith("agent_tool_"): continue
            counts=tool_probes[arm]; counts["total"]+=1
            counts["hot_history" if case.startswith("agent_tool_hot_") else "cold"]+=1
            if row.get("quality",{}).get("task_evaluation",{}).get("passed") is True:
                counts["passed"]+=1
    out={"run_count":len(selected),
         "token_comparison_source":"Unicode text tokens from response text/tool calls; not generated model token IDs",
         "agent_tool_probe_counts":dict(tool_probes),
         "agent_tool_hot_mode":"synthetic conversation history replay; conversation-cache slots are disabled and no tool is executed",
         "arm_run_counts":{},"first_text_token_divergence_vs_first_A":{},"runs":[]}
    by_case=defaultdict(list)
    for order,run in enumerate(selected,1):
        s=run.get("settings",{}); arm=f"tail={s.get('tail')} fetch={s.get('fetch')}"
        out["arm_run_counts"][arm]=out["arm_run_counts"].get(arm,0)+1
        out["runs"].append({"run_order":run.get("run_index",order),"arm":arm,"status":run.get("status"),
                            "quality_failure_cases":run.get("quality_failure_cases",[]),
                            "generation_recovery_cases":run.get("generation_recovery_cases",[])})
        for row in run.get("cases",[]): by_case[row["case"]].append((run,row))
    base={}
    if baseline:
        for row in baseline.get("cases",[]): base[row["case"]]=_output_tokens(row)
    for case,values in by_case.items():
        ref=base.get(case)
        if ref is None: continue
        diffs=[]
        for run,row in values:
            other=_output_tokens(row)
            i=next((i for i,(a,b) in enumerate(zip(ref,other)) if a!=b),min(len(ref),len(other)))
            s=run.get("settings",{})
            diffs.append({"run_order":run.get("run_index"),"arm":f"tail={s.get('tail')} fetch={s.get('fetch')}",
                          "first_text_token_divergence":i if i<max(len(ref),len(other)) else None,
                          "same":ref==other,"baseline_text_token_count":len(ref),
                          "run_text_token_count":len(other)})
        out["first_text_token_divergence_vs_first_A"][case]=diffs
    return out
