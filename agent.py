"""
agent.py  -  the copilot's agent layer.

ONE agent, FOUR tools, and an LLM used only where language is the problem:

  tools (deterministic Python, schema-validated arguments)
    get_forecast        read next-7-day P10/P50/P90 for a route from data/forecasts.csv (never trains anything)
    get_network         assets and costs of a route (flights, charter, trailers)
    run_optimizer       build the MILP inputs, apply approved levers, solve -> plan or diagnosis
    propose_relaxations rank the permitted levers that could fix an infeasible plan (each one is test-solved)

  where the LLM is used (and where it is NOT)
    interpret_notes     free text -> structured adjustments from a CLOSED menu (validated, capped, shown for confirmation)
    explain             plain-language summary / trade-off explanation of tool results
    answer_followup     what-if questions: the LLM chooses a tool + arguments, code validates and runs it
    The LLM never approves a relaxation, never edits solver parameters, never invents numbers.

  guardrails
    * tools read the validated request state - the LLM does not copy numbers into solver calls
    * every number in LLM text must appear in tool outputs, otherwise the deterministic template is used instead
    * no-LLM mode: same plan numbers, template explanation
"""
import os as _os; _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))   # always run relative to the project folder
import copy, json, os, re
from types import SimpleNamespace

import pandas as pd

import optimizer as opt

FC_PATH, NET_PATH = "data/forecasts.csv", "data/network.json"
CASES = {"P50": "Expected demand (P50)", "P90": "High-demand risk (P90)"}
SETTING_RANGES = {"spend_cap": (0, 5_000_000), "min_margin_segment": (0, 0.9), "min_margin_lane": (0, 0.9),
                  "min_fill": (0, 0.95), "max_share": (0, 1.0)}
SETTING_BOOLS = {"allow_downgrade", "allow_unmet"}
BANDS = [0.0, 0.05, 0.10, 0.15, 0.20]

# ------------------------------------------------------------------------------------------ data access (cached by mtime)
_cache = {}


def _load(path, reader):
    m = os.path.getmtime(path)
    if _cache.get(path, (None,))[0] != m:
        _cache[path] = (m, reader(path))
    return _cache[path][1]


def network(): return _load(NET_PATH, opt.load_network)
def forecasts(): return _load(FC_PATH, pd.read_csv)
def money(x): return f"₹{x:,.0f}"
ASSET_LABEL = {"CH": "Charter flight", "TR": "Trailers"}
DIAG_LABEL = {"spend_over_Rs": "spend above the cap (₹)", "lane_margin_short_Rs": "lane margin shortfall (₹)"}


def diag_text(diag):
    return ", ".join(f"{DIAG_LABEL.get(k, k.replace('_', ' '))}: {v:,.0f}" + (" kg of demand cannot be carried" if k.startswith("unmet") else "") for k, v in (diag or {}).items())
def route_name(rid):
    n = network()["states"]; o, d = rid.split("-"); return f"{n[o].split(' (')[0]} to {n[d].split(' (')[0]}"


# ------------------------------------------------------------------------------------------ validation of anything the LLM / UI hands us
def clean_settings(raw, base=None):
    out = dict(base or {})
    for k, v in (raw or {}).items():
        if k in SETTING_RANGES:
            lo, hi = SETTING_RANGES[k]; out[k] = float(min(max(float(v), lo), hi))
        elif k in SETTING_BOOLS: out[k] = bool(v)
        elif k == "price_band": out[k] = min(BANDS, key=lambda b: abs(b - float(v)))
    return out


def clean_request(raw, base):
    """full, validated request: route_id / target_date / demand_case / settings"""
    net, fc = network(), forecasts()
    r = copy.deepcopy(base)
    rid = raw.get("route_id", r["route_id"])
    if rid not in net["routes"]: raise ValueError(f"unknown route {rid}")
    r["route_id"] = rid
    d = raw.get("target_date", r["target_date"])
    if d not in set(fc.target_date): raise ValueError(f"no forecast for date {d}")
    r["target_date"] = d
    c = raw.get("demand_case", r["demand_case"])
    if c not in CASES: raise ValueError(f"demand_case must be one of {list(CASES)}")
    r["demand_case"] = c
    r["settings"] = clean_settings(raw.get("settings"), r.get("settings"))
    return r


def validate_adjustments(raw, route_id):
    """closed menu + hard ranges.  Returns (clean_adjustments, echo_lines, rejected_lines)."""
    net = network(); cats = set(net["categories"]); flights = set(net["routes"][route_id]["air_flights"])
    adj, echo, rej = {"demand_change": [], "asset_outage": [], "cost_multiplier": [], "set": {}}, [], []
    for d in raw.get("demand_change", []) or []:
        try:
            cat, pct = str(d["category"]).upper(), float(d["pct"])
            if cat in ("MEDICINE", "ELECTRICAL", "APPAREL"): cat = {"MEDICINE": "MED", "ELECTRICAL": "ELEC", "APPAREL": "APP"}[cat]
            assert cat in cats | {"ALL"} and -50 <= pct <= 100
            adj["demand_change"].append({"category": cat, "pct": pct})
            echo.append(f"{'All categories' if cat == 'ALL' else net['categories'][cat]['name']} demand {pct:+g}%")
        except Exception: rej.append(f"demand change {d} (category must be MED/ELEC/APP/ALL, change between -50 and +100 %)")
    for o in raw.get("asset_outage", []) or []:
        try:
            a = str(o["asset"])
            a = {"f1": "F1", "f2": "F2", "charter": "charter", "owned_trailer": "owned_trailer", "rentable_trailers": "rentable_trailers"}.get(a.lower(), a)
            if a in flights: adj["asset_outage"].append({"asset": a}); echo.append(f"Flight {a} unavailable (cancelled)")
            elif a == "charter": adj["asset_outage"].append({"asset": a}); echo.append("Charter flight unavailable")
            elif a == "owned_trailer": adj["asset_outage"].append({"asset": a}); echo.append("Owned trailer unavailable")
            elif a == "rentable_trailers":
                lim = int(o.get("limit", 0)); assert 0 <= lim <= net["routes"][route_id]["trailers"]["rentable"]
                adj["asset_outage"].append({"asset": a, "limit": lim}); echo.append(f"Rentable trailers limited to {lim}")
            else: raise ValueError
        except Exception: rej.append(f"asset change {o} (allowed: {sorted(flights)}, charter, owned_trailer, rentable_trailers)")
    for m in raw.get("cost_multiplier", []) or []:
        try:
            a, f = str(m["asset"]).lower(), float(m["factor"]); assert a in ("air", "charter", "trailers") and 0.5 <= f <= 3
            adj["cost_multiplier"].append({"asset": a, "factor": f}); echo.append(f"{a.capitalize()} costs x{f:g}")
        except Exception: rej.append(f"cost change {m} (asset air/charter/trailers, factor between 0.5 and 3)")
    for k, v in (raw.get("set") or {}).items():
        try:
            if k in SETTING_RANGES:
                lo, hi = SETTING_RANGES[k]; assert lo <= float(v) <= hi; adj["set"][k] = float(v); echo.append(f"Set {k.replace('_', ' ')} to {float(v):g}")
            elif k in SETTING_BOOLS | {"allow_downgrade"}:
                adj["set"][k] = bool(v); echo.append(f"Set {k.replace('_', ' ')} to {bool(v)}")
            else: raise ValueError
        except Exception: rej.append(f"setting {k}={v} (not allowed or out of range)")
    for u in raw.get("unrecognized", []) or []: rej.append(f"not understood: {u}")
    return adj, echo, rej


# ------------------------------------------------------------------------------------------ the four tools
def tool_get_forecast(route_id, target_date=None):
    fc = forecasts(); d = fc[fc.route_id == route_id]
    if target_date: d = d[d.target_date == target_date]
    cols = ["target_date", "category", "weight_p10", "weight_p50", "weight_p90", "volume_p50", "items_p50",
            "our_price_per_kg", "backtest_wape_weight", "coverage_p10_p90_weight", "w_tft_weight", "model_version"]
    return {"route_id": route_id, "rows": d[cols].round(3).to_dict("records")}


def tool_get_network(route_id):
    r = network()["routes"][route_id]
    return {"route_id": route_id, **{k: r[k] for k in ("distance_km", "air_flights", "charter", "trailers")}}


def _params(req, adj):
    net, fc = network(), forecasts()
    g = fc[(fc.route_id == req["route_id"]) & (fc.target_date == req["target_date"])]
    P = opt.build_params(net, req["route_id"], g, req["demand_case"], req["settings"])
    return opt.apply_adjustments(P, adj) if adj and any(adj.values()) else P


_solve_cache = {}


def _propose(req, approved, adj):
    key = json.dumps([req, list(approved), adj], sort_keys=True, default=str)
    if key not in _solve_cache:
        if len(_solve_cache) > 200: _solve_cache.clear()
        _solve_cache[key] = opt.propose(_params(req, adj), approved)
    return _solve_cache[key]


def tool_run_optimizer(req, approved=(), adj=None):
    r = _propose(req, approved, adj)
    out = {"status": r["status"], "applied_changes": r.get("applied", [])}
    if r["status"] == "OK":
        out.update(r["result"]); out["solver_status"] = out["status"]; out["status"] = "OK"
    else:
        out["diagnosis"] = r.get("diag"); out["message"] = r.get("message")
    return out


def tool_propose_relaxations(req, approved=(), adj=None):
    r = _propose(req, approved, adj)
    return {"status": r["status"], "diagnosis": r.get("diag"), "options": r.get("options", []), "message": r.get("message")}


TOOL_SCHEMAS = [   # what the LLM sees in follow-up Q&A (it can only ask for WHAT-IF runs; it cannot approve anything)
    {"type": "function", "function": {"name": "get_forecast", "description": "Read the stored 7-day forecast (P10/P50/P90 weight, volume, items) for a route, optionally one date.",
        "parameters": {"type": "object", "properties": {"route_id": {"type": "string"}, "target_date": {"type": "string", "description": "YYYY-MM-DD"}}, "required": []}}},
    {"type": "function", "function": {"name": "run_optimizer", "description": "Run a WHAT-IF plan. Only pass the things that differ from the current request. Pass plain numbers, never expressions.",
        "parameters": {"type": "object", "properties": {
            "route_id": {"type": "string"}, "target_date": {"type": "string"}, "demand_case": {"type": "string", "enum": ["P50", "P90"]},
            "settings": {"type": "object", "properties": {"spend_cap": {"type": "number"}, "min_margin_lane": {"type": "number"}, "min_margin_segment": {"type": "number"},
                                                          "min_fill": {"type": "number"}, "price_band": {"type": "number"}, "allow_downgrade": {"type": "boolean"}}},
            "adjustments": {"type": "object", "properties": {
                "demand_change": {"type": "array", "items": {"type": "object", "properties": {"category": {"type": "string", "enum": ["MED", "ELEC", "APP", "ALL"]}, "pct": {"type": "number"}}}},
                "asset_outage": {"type": "array", "items": {"type": "object", "properties": {"asset": {"type": "string"}, "limit": {"type": "integer"}}}},
                "cost_multiplier": {"type": "array", "items": {"type": "object", "properties": {"asset": {"type": "string", "enum": ["air", "charter", "trailers"]}, "factor": {"type": "number"}}}}}}},
            "required": []}}},
    {"type": "function", "function": {"name": "propose_relaxations", "description": "For a what-if that is infeasible, list the permitted relaxations and their effect (same arguments as run_optimizer).",
        "parameters": {"type": "object", "properties": {"route_id": {"type": "string"}, "target_date": {"type": "string"}, "demand_case": {"type": "string", "enum": ["P50", "P90"]},
                                                         "settings": {"type": "object"}, "adjustments": {"type": "object"}}, "required": []}}},
]


# ------------------------------------------------------------------------------------------ the agent run (tool calls in a fixed policy)
def run_plan(req, approved=(), adj=None):
    """Executes the tool sequence for the dashboard request and returns everything the UI needs, plus a trace."""
    trace, adj = [], adj or {}
    fc = tool_get_forecast(req["route_id"], req["target_date"])
    trace.append({"tool": "get_forecast", "args": f"{req['route_id']} {req['target_date']}", "result": f"{len(fc['rows'])} categories, model {fc['rows'][0]['model_version']}"})
    net = tool_get_network(req["route_id"])
    trace.append({"tool": "get_network", "args": req["route_id"], "result": f"{len(net['air_flights'])} flights, charter, trailers (owned {net['trailers']['owned']}, rentable {net['trailers']['rentable']})"})
    res = tool_run_optimizer(req, approved, adj)
    trace.append({"tool": "run_optimizer", "args": f"{CASES[req['demand_case']]}; approved levers {list(approved) or 'none'}",
                  "result": res["status"] + (f", profit {money(res['profit'])}" if res["status"] == "OK" else "")})
    state = {"request": req, "approved": list(approved), "adjustments": adj, "forecast": fc, "network": net, "status": res["status"],
             "result": res if res["status"] == "OK" else None, "diag": res.get("diagnosis"), "message": res.get("message"),
             "applied": res.get("applied_changes", []), "options": []}
    if res["status"] == "NEEDS_APPROVAL":
        rel = tool_propose_relaxations(req, approved, adj)
        state["options"] = rel["options"]
        trace.append({"tool": "propose_relaxations", "args": f"diagnosis {rel['diagnosis']}", "result": f"{len(rel['options'])} permitted option(s) test-solved, waiting for human approval"})
    state["trace"] = trace
    return state


# ------------------------------------------------------------------------------------------ LLM wrapper (Groq) + mock-friendly interface
class LLM:
    """Thin wrapper.  .ok is False when no key / package -> the app runs in no-LLM mode with identical plan numbers.
    If Groq says the configured model does not exist (models get retired), it asks Groq for the models this key can use,
    picks a suitable chat model, and retries once."""
    PREFERRED = ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "openai/gpt-oss-20b", "meta-llama/llama-4-scout-17b-16e-instruct",
                 "qwen/qwen3-32b", "llama-3.1-8b-instant"]
    SKIP = ("whisper", "guard", "tts", "playai", "embed", "orpheus", "distil")

    def __init__(self):
        self.ok, self.model, self.err = False, os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"), ""
        try:
            from dotenv import load_dotenv; load_dotenv()
            self.model = os.getenv("GROQ_MODEL", self.model).strip()
            key = (os.getenv("GROQ_API_KEY") or "").strip()
            if not key: self.err = "GROQ_API_KEY not set"; return
            from groq import Groq
            self.client, self.ok = Groq(api_key=key), True
        except Exception as e: self.err = str(e)

    def _pick_model(self):
        ids = [m.id for m in self.client.models.list().data if not any(w in m.id.lower() for w in self.SKIP)]
        for p in self.PREFERRED:
            if p in ids and p != self.model: return p
        rest = [i for i in ids if i != self.model]
        if not rest: raise RuntimeError("no usable chat model found for this Groq key")
        return rest[0]

    def _call(self, messages, tools, json_mode, max_tokens):
        kw = dict(model=self.model, messages=messages, temperature=0, max_tokens=max_tokens)
        if tools: kw.update(tools=tools, tool_choice="auto")
        if json_mode: kw["response_format"] = {"type": "json_object"}
        return self.client.chat.completions.create(**kw).choices[0].message

    def chat(self, messages, tools=None, json_mode=False, max_tokens=600):
        try:
            return self._call(messages, tools, json_mode, max_tokens)
        except Exception as e:
            if type(e).__name__ not in ("NotFoundError", "BadRequestError") and "model" not in str(e).lower(): raise
            old, self.model = self.model, self._pick_model()
            print(f"[LLM] model '{old}' failed ({str(e)[:120]}); switching to '{self.model}'")
            return self._call(messages, tools, json_mode, max_tokens)


# ------------------------------------------------------------------------------------------ number-grounding guard
_NUM = re.compile(r"(?<![\w.])-?\d[\d,]*\.?\d*")


def numbers_in(text):
    t = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", " ", text)           # dates
    t = re.sub(r"\b[A-Za-z]+\d+\b", " ", t)                   # F1, F2, P50, P90, T1
    t = re.sub(r"(?m)^\s*\d+[.)]\s", " ", t)                  # list numbering
    out = []
    for m in _NUM.findall(t):
        s = m.replace(",", "").rstrip(".")
        if s in ("", "-"): continue
        out.append((float(s), len(s.split(".")[1]) if "." in s else 0))
    return out


def _collect(x, acc):
    if isinstance(x, bool) or x is None: return
    if isinstance(x, (int, float)): acc.add(float(x))
    elif isinstance(x, dict):
        for v in x.values(): _collect(v, acc)
    elif isinstance(x, (list, tuple)):
        for v in x: _collect(v, acc)
    elif isinstance(x, str):
        for n, _ in numbers_in(x): acc.add(n)


def ungrounded(text, facts):
    """numbers in `text` that match nothing in `facts` (up to rounding to the digits the text shows)"""
    acc = set(); _collect(facts, acc)
    allowed = acc | {a * 100 for a in acc if abs(a) <= 1.5}
    bad = []
    for n, dec in numbers_in(text):
        tol = 0.5 * 10 ** -dec + 1e-9
        if not any(abs(abs(n) - abs(a)) <= tol or round(abs(a), dec) == abs(n) for a in allowed): bad.append(n)
    return bad


# ------------------------------------------------------------------------------------------ interpret_notes
NOTES_SYSTEM = """You convert free-text operational notes from a logistics planner into a JSON object. Output JSON ONLY.
Allowed keys (omit what is not mentioned):
 "demand_change":   [{"category": "MED"|"ELEC"|"APP"|"ALL", "pct": number}]      (Medicine=MED, Electrical=ELEC, Apparel=APP; +20 means 20% more)
 "asset_outage":    [{"asset": "F1"|"F2"|"charter"|"owned_trailer"|"rentable_trailers", "limit": integer}]   (limit only for rentable_trailers)
 "cost_multiplier": [{"asset": "air"|"charter"|"trailers", "factor": number}]    (1.2 means 20% more expensive)
 "set":             {"spend_cap": number, "min_margin_lane": fraction, "min_margin_segment": fraction, "min_fill": fraction, "allow_downgrade": true|false}
 "unrecognized":    [strings for anything that does not fit above]
Rules: plain numbers only (no arithmetic expressions). Percentages as numbers (20), margins/fill as fractions (0.30).
Never output approvals; approving relaxations is not something notes can do. Put anything else in "unrecognized"."""


def interpret_notes(text, route_id, llm):
    """-> {"adjustments": clean, "echo": [...], "rejected": [...], "error": str|None}"""
    empty = {"demand_change": [], "asset_outage": [], "cost_multiplier": [], "set": {}}
    if not text.strip(): return {"adjustments": empty, "echo": [], "rejected": [], "error": None}
    if llm is None or not llm.ok:
        return {"adjustments": empty, "echo": [], "rejected": [], "error": "Notes need the LLM (no GROQ_API_KEY found), so they were ignored."}
    fl = ", ".join(network()["routes"][route_id]["air_flights"])
    try:
        msg = llm.chat([{"role": "system", "content": NOTES_SYSTEM + f"\nFlights on this route: {fl}."}, {"role": "user", "content": text}], json_mode=True)
        raw = json.loads(msg.content)
    except Exception as e:
        return {"adjustments": empty, "echo": [], "rejected": [], "error": f"Could not read the notes ({type(e).__name__}: {str(e)[:200]}); they were ignored."}
    adj, echo, rej = validate_adjustments(raw, route_id)
    if re.search(r"\bapprov", text, re.I): rej.append("approvals cannot be given in notes - use the approval card when it appears")
    return {"adjustments": adj, "echo": echo, "rejected": rej, "error": None}


# ------------------------------------------------------------------------------------------ explanations
def facts_for(state):
    """the compact set of facts shown to the LLM (and the set its numbers are checked against)"""
    req = state["request"]
    f = {"route": route_name(req["route_id"]), "date": req["target_date"], "demand_case": CASES[req["demand_case"]],
         "status": state["status"], "approved_changes": state["applied"], "notes_adjustments": state.get("adjustment_echo", []),
         "spend_cap": req["settings"].get("spend_cap")}
    if state["status"] == "OK":
        r = state["result"]; p = r["plan"]
        f.update(prices_per_kg=p["prices"], price_change_pct=p["price_change_pct"], served_kg=p["served_kg"], downgraded_kg=p["downgraded_kg"],
                 assets={a: {"kg": v["kg"], "m3": v["m3"], "kg_utilisation_pct": round(v["util_kg"] * 100, 1), "m3_utilisation_pct": round(v["util_m3"] * 100, 1)} for a, v in p["assets"].items() if v["kg"] > 0},
                 charter_used=p["charter_used"], rented_trailers=p["rented_trailers"], owned_trailers=p["owned_trailers"],
                 revenue=r["revenue"], cost=r["cost"], profit=r["profit"], lane_margin_pct=round(r["lane_margin"] * 100, 1), added_spend=r["added_spend"],
                 bottlenecks=r.get("binding", []))
    elif state["status"] == "NEEDS_APPROVAL":
        f.update(diagnosis=state["diag"], options=[{k: o[k] for k in ("label", "feasible", "profit", "added_spend")} for o in state["options"]])
    else:
        f.update(message=state["message"], diagnosis=state["diag"])
    return f


def template_summary(state):
    """deterministic explanation; used in no-LLM mode and whenever the LLM text fails the number check"""
    req, st = state["request"], state["status"]
    head = f"{route_name(req['route_id'])} on {req['target_date']} - {CASES[req['demand_case']]}."
    if st == "OK":
        r = state["result"]; p = r["plan"]; net = network()["categories"]
        prices = "; ".join(f"{net[c]['name']} {v:.2f}/kg ({p['price_change_pct'][c]:+.1f}% vs today)" for c, v in p["prices"].items())
        used = [a for a, v in p["assets"].items() if v["kg"] > 0]
        cap = "; ".join(f"{ASSET_LABEL.get(a, a)} carries {p['assets'][a]['kg']:,.0f} kg ({p['assets'][a]['util_kg'] * 100:.0f}% of weight, {p['assets'][a]['util_m3'] * 100:.0f}% of volume capacity)" for a in used)
        extra = []
        if p["charter_used"]: extra.append("a charter flight is used")
        if p["rented_trailers"]: extra.append(f"{p['rented_trailers']} trailer(s) rented")
        if p["downgraded_kg"] > 0: extra.append(f"{p['downgraded_kg']:,.0f} kg of Electrical is downgraded from air to ground")
        txt = (f"{head} Prices: {prices}. Capacity: {cap}. " + ("Added capacity: " + ", ".join(extra) + ". " if extra else "No extra capacity was bought. ")
               + f"Result: revenue {money(r['revenue'])}, cost {money(r['cost'])}, profit {money(r['profit'])}, lane margin {r['lane_margin'] * 100:.1f}%, "
               f"added spend {money(r['added_spend'])}.")
        if state["applied"]: txt += " Approved changes applied: " + "; ".join(state["applied"]) + "."
        return txt
    if st == "NEEDS_APPROVAL":
        d = diag_text(state["diag"])
        feas = [o for o in state["options"] if o["feasible"]]
        txt = f"{head} No plan satisfies all current limits. What breaks: {d}. "
        if feas: txt += "Options that make it feasible: " + "; ".join(f"{o['label']} (profit {money(o['profit'])}, added spend {money(o['added_spend'])})" for o in feas) + ". Nothing is applied until you approve one."
        else: txt += "None of the single options below fixes it on its own; approving one may reduce the gap, or change an input."
        return txt
    return f"{head} {state['message']}"


SUMMARY_SYSTEM = """You are the explanation layer of a logistics planning copilot. Explain the result in plain language for a network planner, at most 120 words.
STRICT RULES: use ONLY numbers that appear in FACTS and write them exactly as they appear (no abbreviations like 'k' or 'lakh', no new calculations, no rounding changes);
do not invent causes; say which limits are binding when 'bottlenecks' is present; if status is NEEDS_APPROVAL say clearly that nothing is applied until the human approves;
never claim that you approved or changed anything."""


def explain(state, llm):
    """-> (text, source)  source in {'llm','template','template (llm text failed number check)'}"""
    facts = facts_for(state)
    if llm is None or not llm.ok: return template_summary(state), "template"
    try:
        msg = llm.chat([{"role": "system", "content": SUMMARY_SYSTEM}, {"role": "user", "content": "FACTS:\n" + json.dumps(facts, default=str)}], max_tokens=350)
        text = (msg.content or "").strip()
    except Exception: return template_summary(state), "template (llm unavailable)"
    bad = ungrounded(text, {"facts": facts, "request": state["request"]})
    return (text, "llm") if text and not bad else (template_summary(state), f"template (llm text failed number check: {bad[:4]})")


# ------------------------------------------------------------------------------------------ follow-up what-if Q&A (real tool selection by the LLM)
FOLLOW_SYSTEM = """You answer follow-up questions about a logistics plan by calling tools. All optimizer runs you make are WHAT-IFS: they do not change the live plan and you cannot approve anything.
Only pass arguments that the question changes; pass plain numbers (never expressions like 7194*1.15). For a question that needs no tool, answer from CURRENT_STATE.
After tool results arrive, answer in at most 100 words using ONLY numbers present in the tool results or CURRENT_STATE, written exactly as they appear. Always say it is a what-if."""


def exec_tool(name, args, state):
    """validate + run one tool call requested by the LLM against the current state"""
    base, approved, adj = state["request"], state["approved"], state.get("adjustments") or {}
    if name == "get_forecast":
        rid = args.get("route_id", base["route_id"])
        if rid not in network()["routes"]: raise ValueError("unknown route")
        return tool_get_forecast(rid, args.get("target_date"))
    if name in ("run_optimizer", "propose_relaxations"):
        req = clean_request({k: args[k] for k in ("route_id", "target_date", "demand_case", "settings") if k in args}, base)
        extra, _, rej = validate_adjustments(args.get("adjustments") or {}, req["route_id"]) if args.get("adjustments") else ({}, [], [])
        merged = {k: list(adj.get(k, [])) + list(extra.get(k, [])) for k in ("demand_change", "asset_outage", "cost_multiplier")}
        merged["set"] = {**adj.get("set", {}), **extra.get("set", {})}
        fn = tool_run_optimizer if name == "run_optimizer" else tool_propose_relaxations
        out = fn(req, approved, merged)         # approvals always come from the human's state, never from the LLM
        out["what_if_request"] = {"route_id": req["route_id"], "target_date": req["target_date"], "demand_case": req["demand_case"], "settings": req["settings"], "extra_adjustments": extra}
        if rej: out["ignored_inputs"] = rej
        return out
    raise ValueError(f"unknown tool {name}")


def answer_followup(question, state, llm, max_rounds=3):
    """-> (text, trace, source)"""
    if llm is None or not llm.ok: return "Follow-up questions need the LLM (set GROQ_API_KEY in .env). The plan above is unaffected.", [], "none"
    ctx = json.dumps(facts_for(state), default=str)
    msgs = [{"role": "system", "content": FOLLOW_SYSTEM}, {"role": "user", "content": f"CURRENT_STATE: {ctx}\nQUESTION: {question}"}]
    trace, evidence = [], {"current": json.loads(ctx), "request": state["request"]}
    try:
        for _ in range(max_rounds):
            m = llm.chat(msgs, tools=TOOL_SCHEMAS)
            calls = getattr(m, "tool_calls", None)
            if not calls:
                text = (m.content or "").strip()
                bad = ungrounded(text, evidence)
                return (text, trace, "llm") if text and not bad else (_followup_template(trace, state), trace, f"template (number check failed: {bad[:4]})")
            msgs.append({"role": "assistant", "content": m.content or "", "tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in calls]})
            for c in calls:
                try:
                    args = json.loads(c.function.arguments or "{}")
                    res = exec_tool(c.function.name, args, state)
                except Exception as e:
                    args, res = {}, {"error": f"{type(e).__name__}: {e}"}
                trace.append({"tool": c.function.name, "args": args, "result": res})
                evidence[f"call{len(trace)}"] = {"args": args, "result": res}
                msgs.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(res, default=str)})
        return _followup_template(trace, state), trace, "template (too many tool rounds)"
    except Exception as e:
        return f"The LLM call failed ({type(e).__name__}: {str(e)[:300]}). The plan above is unaffected.", trace, "error"


def _followup_template(trace, state):
    runs = [t for t in trace if t["tool"] in ("run_optimizer", "propose_relaxations") and "error" not in t["result"]]
    if not runs: return "I could not run a valid what-if for that question."
    r = runs[-1]["result"]
    cur = state["result"]["profit"] if state["status"] == "OK" else None
    if r["status"] == "OK":
        s = f"What-if (not applied): feasible, profit {money(r['profit'])}, lane margin {r['lane_margin'] * 100:.1f}%, added spend {money(r['added_spend'])}."
        return s + (f" Current plan profit is {money(cur)}." if cur is not None else "")
    opts = "; ".join(f"{o['label']} (feasible: {o['feasible']})" for o in r.get("options", []))
    return f"What-if (not applied): status {r['status']}. " + (f"Options: {opts}." if opts else (r.get("message") or ""))
