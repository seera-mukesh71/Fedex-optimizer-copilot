"""
app.py  -  Streamlit dashboard.   Run:  streamlit run app.py

Flow: form -> (optional notes -> confirmation card) -> agent runs its tools -> plan
      -> if infeasible: approval card (human decides) -> re-plan -> explanation -> follow-up what-ifs
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import altair as alt
import pandas as pd
import streamlit as st

import agent as A
import optimizer as opt

st.set_page_config(page_title="Network Capacity & Pricing Copilot", page_icon="✈️", layout="wide")

NET, FC = A.network(), A.forecasts()
CAT = {k: v["name"] for k, v in NET["categories"].items()}
ASSET = {"CH": "Charter flight", "TR": "Trailers (owned + rented)"}
CANON = opt.CANON


@st.cache_resource
def get_llm():
    return A.LLM()


llm = get_llm()
ss = st.session_state
for k, v in dict(state=None, pending=None, adj={}, adj_echo=[], approved=[], rejected=False, answers=[], week=None).items():
    ss.setdefault(k, v)

# ------------------------------------------------------------------------------------------------ header
st.title("Network Capacity & Pricing Copilot")
meta = FC.iloc[0]
c1, c2, c3 = st.columns(3)
c1.caption(f"Forecast model: **{meta.model_version}**  ·  generated {meta.generated_at}")
c2.caption("LLM: **on** (" + llm.model + ")" if llm.ok else f"LLM: **off** - {llm.err}. Plans are identical; explanations use templates.")
use_llm = c3.toggle("Use LLM for explanations / notes / follow-ups", value=llm.ok, disabled=not llm.ok)
active_llm = llm if (llm.ok and use_llm) else None
st.caption("Costs, capacities and elasticities are illustrative assumptions, not real FedEx data.")

# ------------------------------------------------------------------------------------------------ form
with st.container(border=True):
    a, b, c, d = st.columns(4)
    frm = a.selectbox("From", list(NET["states"]), format_func=lambda s: NET["states"][s])
    to = b.selectbox("To", [s for s in NET["states"] if s != frm], format_func=lambda s: NET["states"][s])
    route = f"{frm}-{to}"
    dates = sorted(FC.target_date.unique())
    date = c.selectbox("Date (next 7 days)", dates)
    case = d.radio("Demand case", list(A.CASES), format_func=lambda k: A.CASES[k], horizontal=False)

    e, f, g, h = st.columns(4)
    m_seg = e.number_input("Min margin per category (%)", 0.0, 90.0, 25.0, 1.0)
    m_lane = f.number_input("Min lane margin (%)", 0.0, 90.0, 25.0, 1.0)
    cap = g.number_input("Spend cap on charter + rentals (₹)", 0, 5_000_000, int(NET["routes"][route]["typical_spend_cap"]), 1000, key=f"cap_{route}")
    fill = h.number_input("Min fill on added capacity (%)", 0.0, 95.0, 60.0, 5.0)

    i, j, k_, l = st.columns(4)
    band = i.selectbox("Price band (± of today's price)", A.BANDS, index=2, format_func=lambda x: f"±{x * 100:.0f}%")
    dg = j.checkbox("Allow Electrical air → ground downgrade", True)
    share = k_.number_input("Max downgrade share (%)", 0.0, 100.0, 30.0, 5.0, disabled=not dg)
    unmet = l.checkbox("Allow unmet demand", False, help="Off = every kg must be carried (hard goal).")

    st.markdown("**Pre-approve relaxations** (only if you tick them can the system use them without asking):")
    p1, p2, p3 = st.columns(3)
    pre = []
    if p1.checkbox("Raise spend cap if needed"): pre.append("raise_spend_cap")
    if p2.checkbox("Widen price band if needed"): pre.append("widen_price_band")
    if p3.checkbox("Allow downgrade if needed"): pre.append("allow_downgrade")
    pre = [x for x in CANON if x in pre]

    notes = st.text_area("Anything else? (optional)", placeholder="e.g. Electrical demand up 20% for the festival rush, flight F2 cancelled, trailer costs up 15%",
                         help="Turned into structured adjustments by the LLM from a closed menu. You confirm them before they are used.")
    go = st.button("Plan", type="primary")


def current_request():
    return {"route_id": route, "target_date": date, "demand_case": case,
            "settings": {"min_margin_segment": m_seg / 100, "min_margin_lane": m_lane / 100, "spend_cap": float(cap), "min_fill": fill / 100,
                         "price_band": band, "allow_downgrade": dg, "max_share": share / 100, "allow_unmet": unmet}}


def run(req, approved, adj, echo):
    s = A.run_plan(req, approved, adj)
    s["adjustment_echo"] = echo
    s["explanation"] = A.explain(s, active_llm)
    ss.state, ss.approved, ss.adj, ss.adj_echo, ss.rejected, ss.answers, ss.week = s, list(approved), adj, echo, False, [], None


# ------------------------------------------------------------------------------------------------ button handling
if go:
    ss.pending = None
    req = current_request()
    if notes.strip():
        if active_llm is None:
            st.warning("Notes need the LLM, so they were ignored. Planning with the form settings only.")
            run(req, pre, {}, [])
        else:
            with st.spinner("Reading your notes..."):
                ss.pending = {"req": req, "pre": pre, "interp": A.interpret_notes(notes, route, active_llm)}
    else:
        run(req, pre, {}, [])

if ss.pending:
    p = ss.pending; it = p["interp"]
    with st.container(border=True):
        st.subheader("Please confirm how I understood your notes")
        if it["error"]: st.error(it["error"])
        for line in it["echo"]: st.markdown(f"- ✅ {line}")
        for line in it["rejected"]: st.markdown(f"- ⚠️ ignored: {line}")
        if not it["echo"] and not it["error"]: st.info("Nothing in your notes could be turned into an allowed adjustment.")
        cc1, cc2 = st.columns(2)
        if cc1.button("Confirm and plan", type="primary", key="confirm_notes"):
            ss.pending = None; run(p["req"], p["pre"], it["adjustments"], it["echo"]); st.rerun()
        if cc2.button("Discard notes and plan without them", key="discard_notes"):
            ss.pending = None; run(p["req"], p["pre"], {}, []); st.rerun()

s = ss.state
if not s:
    st.info("Choose a route and date, then press **Plan**.")
    st.stop()

req, res = s["request"], s["result"]
st.divider()
st.subheader(f"{A.route_name(req['route_id'])} · {req['target_date']} · {A.CASES[req['demand_case']]}")
if s["adjustment_echo"]: st.caption("Adjustments from your notes: " + "; ".join(s["adjustment_echo"]))
if s["applied"]: st.caption("Approved changes applied: " + "; ".join(s["applied"]))

# ------------------------------------------------------------------------------------------------ approval card
if s["status"] == "NEEDS_APPROVAL" and not ss.rejected:
    with st.container(border=True):
        st.warning("No plan satisfies every limit you set. **Nothing has been changed - you decide.**")
        st.markdown("**What breaks:** " + A.diag_text(s["diag"]))
        feas = [o for o in s["options"] if o["feasible"]]
        if not feas: st.markdown("None of these fixes it on its own. Approving one may shrink the gap, or change an input above.")
        for n_, o in enumerate(s["options"]):
            col1, col2 = st.columns([4, 1])
            tag = "✅ makes it feasible" if o["feasible"] else f"⚠️ still short (remaining violation {o['remaining_violation']:,})"
            extra = f" · profit ₹{o['profit']:,.0f} · added spend ₹{o['added_spend']:,.0f}" if o["feasible"] else ""
            col1.markdown(f"**{o['label']}**  \n{tag}{extra}  \n<small>{o['tier']}</small>", unsafe_allow_html=True)
            if col2.button("Approve", key=f"approve_{n_}"):
                run(req, ss.approved + [o["token"]], ss.adj, ss.adj_echo); st.rerun()
        if st.button("Reject all (keep my limits)", key="reject"):
            ss.rejected = True; st.rerun()
elif s["status"] == "NEEDS_APPROVAL":
    st.info("You rejected the options. No plan was produced - change an input above and press Plan.")
elif s["status"] == "HARD_GOAL_INFEASIBLE":
    st.error(s["message"])

# ------------------------------------------------------------------------------------------------ results
if s["status"] == "OK":
    p = res["plan"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Profit", f"₹{res['profit']:,.0f}")
    k2.metric("Revenue", f"₹{res['revenue']:,.0f}")
    k3.metric("Lane margin", f"{res['lane_margin'] * 100:.1f}%")
    k4.metric("Added spend", f"₹{res['added_spend']:,.0f}", f"cap ₹{req['settings']['spend_cap']:,.0f}", delta_color="off")

    t_plan, t_fc, t_week, t_trace = st.tabs(["Plan", "Forecast", "7-day outlook", "Agent trace"])
    with t_plan:
        fr = s["forecast"]["rows"]
        base = {r["category"]: r for r in fr}
        rows = []
        for c, price in p["prices"].items():
            rows.append({"Category": CAT[c], "Today ₹/kg": base[c]["our_price_per_kg"], "Planned ₹/kg": price, "Change %": p["price_change_pct"][c],
                         "Demand at planned price (kg)": p["demand_at_price"][c], "Carried (kg)": p["served_kg"][c],
                         "Downgraded to ground (kg)": p["downgraded_kg"] if c == "ELEC" else 0.0})
        st.markdown("**Pricing**"); st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        arows = [{"Asset": ASSET.get(a, a), "Load (kg)": v["kg"], "Capacity (kg)": v["cap_kg"], "Weight util %": round(v["util_kg"] * 100, 1),
                  "Load (m³)": v["m3"], "Capacity (m³)": v["cap_m3"], "Volume util %": round(v["util_m3"] * 100, 1)} for a, v in p["assets"].items()]
        st.markdown("**Capacity**"); st.dataframe(pd.DataFrame(arows), hide_index=True, width="stretch")
        extra = []
        if p["charter_used"]: extra.append("charter flight")
        if p["rented_trailers"]: extra.append(f"{p['rented_trailers']} rented trailer(s)")
        extra.append(f"{p['owned_trailers']} owned trailer(s) dispatched")
        st.caption("Bought / dispatched: " + ", ".join(extra))
        al = pd.DataFrame(p["allocation"]).T.fillna(0.0).rename(index=CAT, columns=ASSET)
        st.markdown("**Who rides what (kg)**"); st.dataframe(al, width="stretch")
        cb = {k.replace("_", " ").capitalize(): v for k, v in res["cost_breakdown"].items() if v}
        c_a, c_b = st.columns(2)
        c_a.markdown("**Cost breakdown (₹)**"); c_a.dataframe(pd.DataFrame({"₹": cb}), width="stretch")
        c_b.markdown("**What limits profit here** (shadow prices)")
        if res.get("binding"):
            for b_ in res["binding"]:
                c_b.markdown(f"- {b_['constraint']}: one more {b_['unit'] or 'unit'} would add about ₹{b_['profit_per_unit_Rs']:,.2f} profit")
        else: c_b.markdown("- No capacity limit is binding at the chosen prices.")

    with t_fc:
        d = FC[FC.route_id == req["route_id"]].copy()
        d["Category"] = d.category.map(CAT)
        band_ = alt.Chart(d).mark_area(opacity=0.25).encode(x=alt.X("target_date:N", title="date"), y=alt.Y("weight_p10:Q", title="weight (kg)"), y2="weight_p90:Q", color="Category:N")
        line = alt.Chart(d).mark_line(point=True).encode(x="target_date:N", y="weight_p50:Q", color="Category:N")
        rule = alt.Chart(pd.DataFrame({"target_date": [req["target_date"]]})).mark_rule(strokeDash=[4, 4]).encode(x="target_date:N")
        st.altair_chart((band_ + line + rule).properties(height=320), width="stretch")
        st.caption("Line = P50, band = P10-P90. Dashed line = selected date. Backtested on the last 8 weeks; interval widened so coverage is about 80%.")
        q = d[d.target_date == req["target_date"]][["Category", "weight_p10", "weight_p50", "weight_p90", "volume_p50", "items_p50", "w_tft_weight", "backtest_wape_weight", "coverage_p10_p90_weight"]]
        q.columns = ["Category", "P10 kg", "P50 kg", "P90 kg", "P50 m³", "P50 items", "TFT blend weight", "Backtest WAPE", "P10-P90 coverage"]
        st.dataframe(q, hide_index=True, width="stretch")

    with t_week:
        if st.button("Compute 7-day outlook with these settings"):
            with st.spinner("Solving 7 days..."):
                ss.week = opt.plan_week(NET, FC, req["route_id"], req["demand_case"], req["settings"], ss.approved, ss.adj if ss.adj and any(ss.adj.values()) else None)
        if ss.week: st.dataframe(pd.DataFrame(ss.week), hide_index=True, width="stretch"); st.caption("Each day is an independent solve. Days marked NEEDS_APPROVAL need you to open that date and approve.")

    with t_trace:
        st.dataframe(pd.DataFrame(s["trace"]), hide_index=True, width="stretch")
        st.caption("The agent picks tools in a fixed policy for the dashboard run; the LLM chooses tools only in follow-up questions.")

# ------------------------------------------------------------------------------------------------ explanation + follow-up
if s.get("explanation"):
    text, src = s["explanation"]
    st.markdown("### Explanation")
    st.write(text)
    st.caption(f"Source: {src}")

st.markdown("### Ask a follow-up (what-if)")
if active_llm is None: st.caption("Needs the LLM. Add GROQ_API_KEY to .env and restart.")
with st.form("followup"):
    q_ = st.text_input("e.g. What if flight F2 is cancelled? What if the spend cap is 300000?", disabled=active_llm is None)
    asked = st.form_submit_button("Ask", disabled=active_llm is None)
if asked and q_.strip():
    with st.spinner("Thinking..."):
        ss.answers.insert(0, (q_, *A.answer_followup(q_, s, active_llm)))
for q_txt, ans, tr, src in ss.answers:
    with st.container(border=True):
        st.markdown(f"**Q:** {q_txt}"); st.write(ans); st.caption(f"Source: {src}")
        if tr:
            with st.expander("Tools the LLM called"):
                for t in tr: st.code(f"{t['tool']}({t['args']})", language="python")
