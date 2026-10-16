"""
optimizer.py  -  ONE joint MILP for capacity + pricing, plus the closed-loop re-planning controller.

    python optimizer.py                       # demo: 3 scenarios on TN-MH for the first forecast day

Scope of one solve: ONE route, ONE day.  A 7-day plan is 7 independent solves (plan_week).
Solver: OR-Tools pywraplp.  SCIP by default; CBC/HiGHS work too:  set MILP_SOLVER=CBC (or HIGHS).

Decisions
  z[c,k]   binary   which price on the grid each category charges           (pricing)
  q[c,k]   kg       kg sold at that price = demand(price) if chosen         (constant-elasticity demand)
  x[c,a]   kg       category c kg carried on asset a (only compatible pairs exist)
  g        kg       ELEC downgraded air->ground (<= max_share of baseline forecast)
  y_charter binary, n_owned int, n_rented int   which extra capacity is bought
  u[c]     kg       unmet demand (0 unless allowed)
Constraints: demand conservation, weight AND volume cap on every asset, min fill on ADDED assets,
             spend cap (charter + rentals), per-category margin, lane margin.
Objective:   revenue - variable cost - fixed cost - goodwill penalty on unmet kg.

Re-planning: if infeasible -> elastic solve (soft constraints) says WHAT breaks -> each permitted lever is tried
by actually re-solving -> ranked options go to a HUMAN.  Hard goals (margins, min fill, compatibility,
zero unmet) are never relaxed by the system.
"""
import copy, hashlib, json, math, os, sys
from ortools.linear_solver import pywraplp

INF = float("inf")

# --------------------------------------------------------------------------- default parameters (also used by the tests)
DEFAULT = {
    "segments": {   # demand = forecast kg at today's price p_ref
        "MED":  {"demand": 3000, "rho": 0.004,  "p_ref": 62, "elast": -0.4, "mode": "air",    "unmet_penalty": 150},
        "ELEC": {"demand": 4000, "rho": 0.005,  "p_ref": 55, "elast": -1.2, "mode": "air",    "unmet_penalty": 60},
        "APP":  {"demand": 9000, "rho": 0.0078, "p_ref": 22, "elast": -1.8, "mode": "ground", "unmet_penalty": 10},
    },
    "price_grid": [0.90, 0.95, 1.00, 1.05, 1.10],
    "downgrade": {"from": "ELEC", "allowed": True, "price": 28, "rho": 0.005, "max_share": 0.30},
    "air_flights": {"F1": {"cap_w": 4500, "cap_v": 28, "var_cost": 30}, "F2": {"cap_w": 2000, "cap_v": 10, "var_cost": 30}},
    "charter":  {"cap_w": 4000, "cap_v": 20, "fixed": 140000, "var_cost": 0, "available": 1},
    "trailers": {"cap_w": 8000, "cap_v": 40, "owned": 1, "owned_trip": 60000, "rentable": 3, "rent_trip": 68000, "handling": 3},
    "min_fill": 0.60, "min_margin_segment": 0.25, "min_margin_lane": 0.25,
    "spend_cap": 200000, "allow_unmet": False,
}


# --------------------------------------------------------------------------- the MILP
def solve(P, time_limit_s=20, elastic=False, fixed=None):
    """elastic=True : unmet demand / spend cap / lane margin become soft with big penalties (diagnosis only)
       fixed=dict   : re-solve as a pure LP with every integer fixed -> shadow prices"""
    S, G = P["segments"], P["price_grid"]
    ch, tr, dg, fl = P["charter"], P["trailers"], P["downgrade"], P["air_flights"]
    if fixed:
        s = pywraplp.Solver.CreateSolver("GLOP")
        mk_bool = lambda nm: s.NumVar(fixed[nm], fixed[nm], nm)
        mk_int = lambda lo, hi, nm: s.NumVar(fixed[nm], fixed[nm], nm)
    else:
        s = pywraplp.Solver.CreateSolver(os.environ.get("MILP_SOLVER", "SCIP")) or pywraplp.Solver.CreateSolver("CBC")
        s.SetTimeLimit(time_limit_s * 1000)
        mk_bool, mk_int = s.BoolVar, s.IntVar
    ints, cons = {}, {}

    def BV(nm): v = mk_bool(nm); ints[nm] = v; return v
    def NI(lo, hi, nm): v = mk_int(lo, hi, nm); ints[nm] = v; return v
    def NV(lo, hi, nm): return s.NumVar(lo, hi, nm)
    def C(name, lo, hi): c = s.Constraint(lo, hi, name); cons[name] = c; return c

    # ---- price choice: exactly one grid price per category; q = kg sold at that price
    z, q, D = {}, {}, {}
    for n, seg in S.items():
        for k, m in enumerate(G):
            D[n, k] = seg["demand"] * m ** seg["elast"]              # constant elasticity: D = D0 * (p/p_ref)^e
            z[n, k] = BV(f"z_{n}_{k}"); q[n, k] = NV(0, INF, f"q_{n}_{k}")
            c = C(f"link_{n}_{k}", -INF, 0); c.SetCoefficient(q[n, k], 1); c.SetCoefficient(z[n, k], -D[n, k])
        c = C(f"onePrice_{n}", 1, 1)
        for k in range(len(G)): c.SetCoefficient(z[n, k], 1)

    # ---- capacity purchase decisions
    y_ch = BV("y_charter")
    n_own = NI(0, tr["owned"], "n_owned")
    n_rent = NI(0, tr["rentable"], "n_rented")
    a_w_ch, a_v_ch, a_w_rt, a_v_rt = BV("fill_w_charter"), BV("fill_v_charter"), BV("fill_w_rent"), BV("fill_v_rent")

    # ---- flows: x[c,a] exists only for compatible (category, asset) pairs
    x = {}
    for n, seg in S.items():
        for a in (list(fl) + ["CH"] if seg["mode"] == "air" else ["TR"]): x[n, a] = NV(0, INF, f"x_{n}_{a}")
    dg_on = bool(dg.get("allowed")) and dg.get("from") in S
    g = NV(0, INF, "x_DOWN_TR") if dg_on else None
    u = {n: NV(0, INF if (P["allow_unmet"] or elastic) else 0, f"unmet_{n}") for n in S}
    BIG = 1e4
    sl_spend = NV(0, INF if elastic else 0, "slack_spend")
    sl_margin = NV(0, INF if elastic else 0, "slack_lane_margin")

    # ---- demand conservation:  carried + downgraded + unmet = D(chosen price);  carried = sold
    for n in S:
        c = C(f"demand_{n}", 0, 0)
        for (nn, a), v in x.items():
            if nn == n: c.SetCoefficient(v, 1)
        c.SetCoefficient(u[n], 1)
        if g is not None and n == dg["from"]: c.SetCoefficient(g, 1)
        for k in range(len(G)): c.SetCoefficient(z[n, k], -D[n, k])
        c2 = C(f"served_{n}", 0, 0)
        for (nn, a), v in x.items():
            if nn == n: c2.SetCoefficient(v, 1)
        for k in range(len(G)): c2.SetCoefficient(q[n, k], -1)
    if g is not None:     # only a share of the BASELINE forecast accepts a downgrade (stops the "downgrade everything" loophole)
        c = C("downgrade_share", -INF, dg["max_share"] * S[dg["from"]]["demand"]); c.SetCoefficient(g, 1)

    # ---- capacity: weight AND volume on every asset
    def load_w(asset): return [(v, 1) for (n, a), v in x.items() if a == asset]
    def load_v(asset): return [(v, S[n]["rho"]) for (n, a), v in x.items() if a == asset]
    for f, spec in fl.items():
        cw, cv = C(f"cap_{f}_w", -INF, spec["cap_w"]), C(f"cap_{f}_v", -INF, spec["cap_v"])
        for v, w in load_w(f): cw.SetCoefficient(v, w)
        for v, w in load_v(f): cv.SetCoefficient(v, w)
    cw, cv = C("cap_charter_w", -INF, 0), C("cap_charter_v", -INF, 0)
    for v, w in load_w("CH"): cw.SetCoefficient(v, w)
    for v, w in load_v("CH"): cv.SetCoefficient(v, w)
    cw.SetCoefficient(y_ch, -ch["cap_w"]); cv.SetCoefficient(y_ch, -ch["cap_v"])
    c = C("charter_avail", -INF, ch["available"]); c.SetCoefficient(y_ch, 1)

    tw = load_w("TR") + ([(g, 1)] if g is not None else [])
    tv = load_v("TR") + ([(g, dg["rho"])] if g is not None else [])
    cw, cv = C("cap_trailers_w", -INF, 0), C("cap_trailers_v", -INF, 0)
    for v, w in tw: cw.SetCoefficient(v, w)
    for v, w in tv: cv.SetCoefficient(v, w)
    for nv in (n_rent, n_own): cw.SetCoefficient(nv, -tr["cap_w"]); cv.SetCoefficient(nv, -tr["cap_v"])

    # ---- minimum fill on ADDED assets only:  (weight fill >= f) OR (volume fill >= f)   [big-M]
    f = P["min_fill"]
    c = C("fillpick_charter", 0, INF); c.SetCoefficient(a_w_ch, 1); c.SetCoefficient(a_v_ch, 1); c.SetCoefficient(y_ch, -1)
    c = C("fillW_charter", 0, INF)
    for v, w in load_w("CH"): c.SetCoefficient(v, w)
    c.SetCoefficient(a_w_ch, -f * ch["cap_w"])
    c = C("fillV_charter", 0, INF)
    for v, w in load_v("CH"): c.SetCoefficient(v, w)
    c.SetCoefficient(a_v_ch, -f * ch["cap_v"])
    if tr["rentable"] > 0:
        # rented trailers (pooled): load beyond what owned trailers can carry must fill the rented ones to >= f
        c = C("fillpick_rent", 0, INF); c.SetCoefficient(a_w_rt, 1); c.SetCoefficient(a_v_rt, 1); c.SetCoefficient(n_rent, -1 / tr["rentable"])
        M_w, M_v = f * tr["cap_w"] * tr["rentable"], f * tr["cap_v"] * tr["rentable"]
        cW, cV = C("fillW_rent", -M_w, INF), C("fillV_rent", -M_v, INF)
        for v, w in tw: cW.SetCoefficient(v, w)
        for v, w in tv: cV.SetCoefficient(v, w)
        cW.SetCoefficient(n_own, -tr["cap_w"]); cV.SetCoefficient(n_own, -tr["cap_v"])
        cW.SetCoefficient(n_rent, -f * tr["cap_w"]); cV.SetCoefficient(n_rent, -f * tr["cap_v"])
        cW.SetCoefficient(a_w_rt, -M_w); cV.SetCoefficient(a_v_rt, -M_v)

    # ---- money
    rev_c = {n: s.Sum([S[n]["p_ref"] * G[k] * q[n, k] for k in range(len(G))]) for n in S}
    revenue = s.Sum(list(rev_c.values()))
    down_rev = dg["price"] * g if g is not None else 0
    unit_var = lambda n, a: fl[a]["var_cost"] if a in fl else (ch["var_cost"] if a == "CH" else tr["handling"])
    air_var = s.Sum([fl[a]["var_cost"] * v for (n, a), v in x.items() if a in fl])
    ch_var = s.Sum([ch["var_cost"] * v for (n, a), v in x.items() if a == "CH"])
    gr_var = s.Sum([tr["handling"] * v for (n, a), v in x.items() if a == "TR"] + ([tr["handling"] * g] if g is not None else []))
    fixed_added = ch["fixed"] * y_ch + tr["rent_trip"] * n_rent
    fixed_all = fixed_added + tr["owned_trip"] * n_own
    goodwill = s.Sum([S[n]["unmet_penalty"] * u[n] for n in S])

    c = C("spend_cap", -INF, P["spend_cap"])
    c.SetCoefficient(y_ch, ch["fixed"]); c.SetCoefficient(n_rent, tr["rent_trip"]); c.SetCoefficient(sl_spend, -1)

    for n in S:      # per-category margin on VARIABLE cost; fixed costs are not allocated to a category
        vc = s.Sum([unit_var(n, a) * v for (nn, a), v in x.items() if nn == n])
        s.Add(rev_c[n] - vc >= P["min_margin_segment"] * rev_c[n])
    total_rev, total_cost = revenue + down_rev, air_var + ch_var + gr_var + fixed_all      # lane margin on ALL avoidable cost
    cons["lane_margin"] = s.Add(total_rev - total_cost + sl_margin >= P["min_margin_lane"] * total_rev)
    pen = s.Sum([BIG * u[n] for n in S]) + 10 * sl_spend + 10 * sl_margin if elastic else 0
    s.Maximize(total_rev - total_cost - goodwill - pen)

    st = s.Solve()
    names = {pywraplp.Solver.OPTIMAL: "OPTIMAL", pywraplp.Solver.FEASIBLE: "FEASIBLE", pywraplp.Solver.INFEASIBLE: "INFEASIBLE"}
    out = {"status": names.get(st, str(st))}
    if st not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return out

    val = lambda v: v.solution_value()
    R, Cst = total_rev.solution_value(), total_cost.solution_value()
    prices, served, dem, alloc, pchg = {}, {}, {}, {}, {}
    for n in S:
        k = [kk for kk in range(len(G)) if val(z[n, kk]) > .5][0]
        prices[n] = round(S[n]["p_ref"] * G[k], 2); pchg[n] = round((G[k] - 1) * 100, 1); dem[n] = round(D[n, k], 1)
        alloc[n] = {a: round(val(v), 1) for (nn, a), v in x.items() if nn == n and val(v) > 1e-6}
        served[n] = round(sum(alloc[n].values()), 1)
    down_kg = round(val(g), 1) if g is not None else 0.0
    nco, nrt, now = int(round(val(y_ch))), int(round(val(n_rent))), int(round(val(n_own)))
    assets = {}
    for a in list(fl) + ["CH", "TR"]:
        kg = sum(val(v) for (n, aa), v in x.items() if aa == a) + (down_kg if a == "TR" else 0)
        m3 = sum(val(v) * S[n]["rho"] for (n, aa), v in x.items() if aa == a) + (down_kg * dg["rho"] if a == "TR" and g is not None else 0)
        if a in fl: cw_, cv_ = fl[a]["cap_w"], fl[a]["cap_v"]
        elif a == "CH": cw_, cv_ = ch["cap_w"] * nco, ch["cap_v"] * nco
        else: cw_, cv_ = tr["cap_w"] * (nrt + now), tr["cap_v"] * (nrt + now)
        assets[a] = {"kg": round(kg, 1), "m3": round(m3, 2), "cap_kg": cw_, "cap_m3": cv_,
                     "util_kg": round(kg / cw_, 3) if cw_ else 0.0, "util_m3": round(m3 / cv_, 3) if cv_ else 0.0}
    out["plan"] = {"prices": prices, "price_change_pct": pchg, "demand_at_price": dem, "served_kg": served, "allocation": alloc,
                   "downgraded_kg": down_kg, "assets": assets, "charter_used": nco, "rented_trailers": nrt, "owned_trailers": now}
    out.update(revenue=round(R), cost=round(Cst), profit=round(R - Cst), lane_margin=round((R - Cst) / R, 4) if R else 0.0,
               added_spend=round(fixed_added.solution_value()),
               cost_breakdown={"air_variable": round(air_var.solution_value()), "charter_variable": round(ch_var.solution_value()),
                               "ground_handling": round(gr_var.solution_value()), "charter_fixed": round(ch["fixed"] * nco),
                               "trailer_rental": round(tr["rent_trip"] * nrt), "owned_trailer_trip": round(tr["owned_trip"] * now)})
    out["ints"] = {nm: round(val(v)) for nm, v in ints.items()}
    out["violations"] = {k: round(v, 1) for k, v in {**{f"unmet_{n}_kg": val(u[n]) for n in S}, "spend_over_Rs": val(sl_spend),
                                                     "lane_margin_short_Rs": val(sl_margin)}.items() if v > 1e-6}
    if fixed:
        out["shadow_prices"] = {nm: round(c.dual_value(), 3) for nm, c in cons.items()
                                if not nm.startswith(("link_", "onePrice", "fillpick")) and abs(c.dual_value()) > 1e-6}
    return out


# --------------------------------------------------------------------------- shadow prices -> "what is the bottleneck"
def binding_constraints(P, res, top=3):
    """Re-solve as an LP with the integer decisions of `res` fixed; duals = Rs of profit per extra unit of that limit."""
    r = solve(P, fixed=res["ints"])
    if "shadow_prices" not in r: return []
    lab = {"spend_cap": ("spend cap", "Rs"), "downgrade_share": ("downgrade share limit", "kg"), "lane_margin": ("lane margin floor", "Rs")}
    out = []
    for nm, d in r["shadow_prices"].items():
        if nm.startswith(("demand_", "served_", "charter_avail")): continue
        if nm.startswith("cap_charter") and not res["plan"]["charter_used"]: continue      # unused asset: not a real bottleneck
        if nm in lab: name, unit = lab[nm]
        elif nm.startswith("cap_"):
            _, asset, kind = nm.split("_"); name, unit = f"{asset} {'weight' if kind == 'w' else 'volume'}", "kg" if kind == "w" else "m3"
        elif nm.startswith("fill"): name, unit = "minimum-fill rule on added capacity", ""
        else: continue
        out.append({"constraint": name, "unit": unit, "profit_per_unit_Rs": d})
    return sorted(out, key=lambda o: -abs(o["profit_per_unit_Rs"]))[:top]


# --------------------------------------------------------------------------- re-planning levers (all need a human)
GRID_STEP, MAX_BAND = 0.05, 0.20


def _lever_spend(P, over, to=None):
    new = to if to is not None else P["spend_cap"] + math.ceil(over / 1000) * 1000
    P["spend_cap"] = new

def _lever_downgrade(P, *_):  P["downgrade"]["allowed"] = True

def _lever_band(P, *_):
    lo = max(min(P["price_grid"]) - GRID_STEP, 1 - MAX_BAND)        # never wider than +-20%
    hi = min(max(P["price_grid"]) + GRID_STEP, 1 + MAX_BAND)
    P["price_grid"] = [round(1 + GRID_STEP * i, 2) for i in range(int(round((lo - 1) / GRID_STEP)), int(round((hi - 1) / GRID_STEP)) + 1)]

LEVERS = {   # name -> (tier, applier, label)
    "widen_price_band": ("T1-pricing", _lever_band, "Widen the allowed price band by one step (5%) on each side"),
    "allow_downgrade": ("T1-service", _lever_downgrade, "Allow part of the Electrical air volume to be downgraded to ground"),
    "raise_spend_cap": ("T1-budget", _lever_spend, "Raise the spend cap"),
}
CANON = ["widen_price_band", "allow_downgrade", "raise_spend_cap"]


def _key(P): return hashlib.md5(json.dumps(P, sort_keys=True).encode()).hexdigest()


def _remaining(Q):
    """distance from feasible in elastic units (0 = feasible)"""
    d = solve(Q, elastic=True).get("violations")
    if d is None: return 1e9
    return 1e4 * sum(v for k, v in d.items() if k.startswith("unmet")) + 10 * d.get("spend_over_Rs", 0) + 10 * d.get("lane_margin_short_Rs", 0)


def _is_noop(name, P):
    Q = copy.deepcopy(P); LEVERS[name][1](Q, 0)
    return _key(Q) == _key(P)


def apply_levers(P, tokens):
    """tokens: 'allow_downgrade' | 'widen_price_band' | 'raise_spend_cap' (amount = what is needed now) | 'raise_spend_cap=68000'.
    Returns (new P, list of human-readable changes)."""
    P, notes = copy.deepcopy(P), []
    for t in tokens:
        name, _, arg = t.partition("=")
        if name not in LEVERS: raise ValueError(f"unknown lever {name}")
        before = copy.deepcopy(P)
        if name == "raise_spend_cap":
            if arg: _lever_spend(P, 0, to=float(arg))
            else:
                v = solve(P, elastic=True).get("violations") or {}
                _lever_spend(P, v.get("spend_over_Rs", 0))
            if P["spend_cap"] != before["spend_cap"]:
                notes.append(f"Spend cap raised from ₹{before['spend_cap']:,.0f} to ₹{P['spend_cap']:,.0f}")
        else:
            LEVERS[name][1](P)
            if _key(P) != _key(before):
                notes.append({"widen_price_band": f"Price band widened to {min(P['price_grid']):.2f}x - {max(P['price_grid']):.2f}x of today's price",
                              "allow_downgrade": "Downgrade of part of the Electrical air volume to ground allowed"}[name])
    return P, notes


def propose(P, approved=()):
    """Stateless controller used by the agent tool.  Applies levers a HUMAN already approved, solves, and if the problem is
    still infeasible returns the diagnosis plus ranked, human-approvable options."""
    P, applied = apply_levers(P, approved)
    r = solve(P)
    if r["status"] in ("OPTIMAL", "FEASIBLE"):
        r["binding"] = binding_constraints(P, r)
        return {"status": "OK", "result": r, "applied": applied, "params": P}
    el = solve(P, elastic=True)
    if "violations" not in el:   # infeasible even with soft constraints -> only hard goals can be the cause
        return {"status": "HARD_GOAL_INFEASIBLE", "diag": {}, "applied": applied, "params": P,
                "message": "Even with spend, unmet-demand and lane-margin made flexible there is no plan: a hard goal "
                           "(category margin floor, minimum fill, or compatibility) conflicts with demand/capacity."}
    diag = el["violations"]
    only_margin = "lane_margin_short_Rs" in diag and not any(k.startswith(("unmet", "spend")) for k in diag)
    if only_margin:
        return {"status": "HARD_GOAL_INFEASIBLE", "diag": diag, "applied": applied, "params": P,
                "message": "The lane margin target cannot be met by any plan. It is a hard goal, so the system will not relax it: lower it or change the inputs."}
    over = diag.get("spend_over_Rs", 0)
    options = []
    for name in CANON:
        if name in [a.split("=")[0] for a in approved] and name != "raise_spend_cap": continue
        if _is_noop(name, P) and not (name == "raise_spend_cap" and over > 0): continue
        if name == "raise_spend_cap" and over <= 0: continue
        Q = copy.deepcopy(P); LEVERS[name][1](Q, over)
        rr = solve(Q); ok = rr["status"] in ("OPTIMAL", "FEASIBLE")
        token = f"raise_spend_cap={Q['spend_cap']:.0f}" if name == "raise_spend_cap" else name
        label = (f"Raise the spend cap from ₹{P['spend_cap']:,.0f} to ₹{Q['spend_cap']:,.0f}" if name == "raise_spend_cap" else LEVERS[name][2])
        options.append({"lever": name, "token": token, "label": label, "tier": LEVERS[name][0], "feasible": ok,
                        "profit": rr.get("profit"), "added_spend": rr.get("added_spend"),
                        "remaining_violation": 0 if ok else round(_remaining(Q))})
    options.sort(key=lambda o: (not o["feasible"], o["remaining_violation"], -(o["profit"] or 0)))
    if not options:
        return {"status": "HARD_GOAL_INFEASIBLE", "diag": diag, "applied": applied, "params": P,
                "message": "No permitted relaxation is left that changes the outcome; change a hard goal or the inputs."}
    return {"status": "NEEDS_APPROVAL", "diag": diag, "options": options, "applied": applied, "params": P}


# --------------------------------------------------------------------------- building the solver input from files + user settings
def load_network(path="data/network.json"): return json.load(open(path))


def band_grid(band):
    n = int(round(band / GRID_STEP))
    return [round(1 + GRID_STEP * i, 2) for i in range(-n, n + 1)]


def build_params(network, route_id, fc_rows, case="P50", settings=None):
    """fc_rows: forecast rows (one per category) for ONE route and ONE date.  case: 'P50' (expected) or 'P90' (high-demand risk)."""
    st = {**network["defaults"], "spend_cap": network["routes"][route_id]["typical_spend_cap"], "price_band": 0.10, "max_share": network["downgrade"]["max_share"]}
    st.update(settings or {})
    rt, dgn = network["routes"][route_id], network["downgrade"]
    p = case.lower()
    seg = {}
    for _, r in fc_rows.iterrows():
        cat = r["category"]; cfg = network["categories"][cat]
        w, v = float(r[f"weight_{p}"]), float(r[f"volume_{p}"])
        rho = v / w if w > 0 else cfg["rho"]
        rho = min(max(rho, 0.5 * cfg["rho"]), 2.0 * cfg["rho"])         # guard against a silly volume/weight pair
        seg[cat] = {"demand": round(w, 1), "rho": round(rho, 6), "p_ref": float(r["our_price_per_kg"]), "elast": cfg["elasticity"],
                    "mode": cfg["mode"], "unmet_penalty": cfg["unmet_penalty"]}
    dg_cat = dgn["category"] if dgn["category"] in seg else None
    return {"segments": seg, "price_grid": band_grid(st["price_band"]),
            "downgrade": {"from": dg_cat, "allowed": bool(st["allow_downgrade"]) and dg_cat is not None,
                          "price": round(dgn["price_ratio"] * seg[dg_cat]["p_ref"], 2) if dg_cat else 0, "rho": dgn["rho"], "max_share": st["max_share"]},
            "air_flights": copy.deepcopy(rt["air_flights"]), "charter": copy.deepcopy(rt["charter"]), "trailers": copy.deepcopy(rt["trailers"]),
            "min_fill": st["min_fill"], "min_margin_segment": st["min_margin_segment"], "min_margin_lane": st["min_margin_lane"],
            "spend_cap": st["spend_cap"], "allow_unmet": bool(st["allow_unmet"])}


def apply_adjustments(P, adj):
    """Deterministically apply VALIDATED notes-adjustments (see agent.validate_adjustments). Returns a new P."""
    P = copy.deepcopy(P)
    for d in adj.get("demand_change", []):
        for c, seg in P["segments"].items():
            if d["category"] in ("ALL", c): seg["demand"] = round(seg["demand"] * (1 + d["pct"] / 100), 1)
    for o in adj.get("asset_outage", []):
        a, lim = o["asset"], o.get("limit", 0)
        if a in P["air_flights"]: P["air_flights"][a].update(cap_w=0, cap_v=0)
        elif a == "charter": P["charter"]["available"] = 0
        elif a == "owned_trailer": P["trailers"]["owned"] = 0
        elif a == "rentable_trailers": P["trailers"]["rentable"] = int(lim)
    for m in adj.get("cost_multiplier", []):
        f, a = m["factor"], m["asset"]
        if a == "air":
            for fl in P["air_flights"].values(): fl["var_cost"] = round(fl["var_cost"] * f, 2)
        elif a == "charter": P["charter"]["fixed"] = round(P["charter"]["fixed"] * f)
        elif a == "trailers":
            for k in ("owned_trip", "rent_trip"): P["trailers"][k] = round(P["trailers"][k] * f)
            P["trailers"]["handling"] = round(P["trailers"]["handling"] * f, 2)
    for k, v in adj.get("set", {}).items():
        if k == "allow_downgrade":
            P["downgrade"]["allowed"] = bool(v) and P["downgrade"]["from"] is not None
        else:
            P[k] = v
    return P


def plan_week(network, fc, route_id, case, settings, approved=(), adj=None):
    """7 independent one-day solves with the same settings -> small table for the outlook strip"""
    rows = []
    for d, g in fc[fc.route_id == route_id].groupby("target_date"):
        P = build_params(network, route_id, g, case, settings)
        if adj: P = apply_adjustments(P, adj)
        r = propose(P, approved)
        res = r.get("result", {})
        rows.append({"date": d, "status": r["status"], "profit": res.get("profit"), "added_spend": res.get("added_spend"),
                     "lane_margin": res.get("lane_margin")})
    return rows


# --------------------------------------------------------------------------- demo
if __name__ == "__main__":
    import pandas as pd
    os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.stdout.reconfigure(encoding="utf-8")
    net = load_network()
    fc = pd.read_csv("data/forecasts.csv")
    day = fc.target_date.min()
    g = fc[(fc.route_id == "TN-MH") & (fc.target_date == day)]
    for title, case, st in [("Expected demand, defaults", "P50", {}), ("High-demand risk (P90)", "P90", {}),
                            ("Tight spend cap Rs 30,000", "P50", {"spend_cap": 30000})]:
        P = build_params(net, "TN-MH", g, case, st)
        r = propose(P)
        print(f"\n=== {title}: {r['status']}")
        if r["status"] == "OK":
            x = r["result"]; print("prices", x["plan"]["prices"], "profit", x["profit"], "added spend", x["added_spend"], "margin", x["lane_margin"])
        else:
            print(json.dumps({k: v for k, v in r.items() if k in ("diag", "options", "message")}, indent=1)[:900])
