# Network Capacity & Pricing Copilot

A prototype of a FedEx-style planning copilot, written in Python only (Streamlit front end).
For one route and one day it decides **what price to charge each category** and **which capacity to use or buy**
(scheduled flights, a charter, owned and rented trailers) in **one joint MILP**, driven by a 7-day
XGBoost + Temporal Fusion Transformer demand forecast. If the limits you set cannot all be met, it
diagnoses what breaks, test-solves the permitted fixes and **asks a human to approve one** before re-planning.

> All data is synthetic. Costs, capacities, price elasticities and the downgrade rule are illustrative
> assumptions in `data_gen.py`, not real FedEx figures.

## Run it (Windows PowerShell, from the project folder)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1            # if blocked:  Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements-tft.txt     # or requirements.txt for XGBoost only (then add --no-tft below)
copy .env.example .env                  # optional: put your GROQ_API_KEY in .env

python data_gen.py                      # synthetic history + network.json
python forecast.py train                # weekly job: backtest, blend weights, refit, save (about 6 min on 2 CPU cores with the TFT)
python forecast.py forecast             # nightly job: writes data/forecasts.csv
streamlit run app.py
```

XGBoost only: `python forecast.py train --no-tft`. Without a Groq key the app still works: plans are identical and
explanations use a template (notes box and follow-up questions need the LLM).

## Files

| File | Job |
|---|---|
| `data_gen.py` | 3 states -> 6 ordered routes x 3 categories (Medicine, Electrical, Apparel) = 18 series, 3 years daily + 7 future rows; writes `data/demand_data.csv` and `data/network.json` |
| `forecast.py` | `train` (weekly) / `forecast` (nightly). XGBoost per series + one global TFT, per-series blend weight chosen on the last 8 weeks, writes `data/forecasts.csv` and `data/backtest_metrics.csv` |
| `optimizer.py` | the joint capacity + pricing MILP (OR-Tools), diagnosis, ranked relaxation levers |
| `agent.py` | four tools, `interpret_notes`, explanations, follow-up what-ifs, number-grounding check |
| `app.py` | Streamlit dashboard |

## How it works

**Forecast.** Queries never train anything; they read `forecasts.csv`. Each series has an XGBoost model (quantile
objective, P10/P50/P90, every lag at least 7 days so one model covers horizons 1-7). One TFT is trained on all 18
series. For weight and volume the two are blended per series with a weight in {0, 0.1, ..., 1} chosen by pinball loss on
the last 8 weeks; item count uses XGBoost only. The P10-P90 interval is widened by a factor chosen on the same
holdout so coverage is about 80% (this makes the reported coverage slightly optimistic).

**Optimizer.** Binary variables pick one price per category from a grid (multiples of today's price); demand at each
price is `D0 * m^elasticity`. Category kg are assigned to compatible assets only (Medicine and Electrical fly, Apparel
goes by trailer). Every asset has a weight cap and a volume cap. Added assets (charter, rented trailers) must be at least
60% full by weight or volume. Spend cap applies to charter + rentals. Margin floors are per category (on variable cost)
and per lane (on all avoidable cost). Electrical can be downgraded to ground up to a share of the baseline forecast.
Objective: revenue minus variable and fixed cost minus a goodwill penalty on unmet kg.

**Re-planning (human in the loop).** Infeasible -> an elastic solve shows what breaks -> each permitted lever (raise
spend cap, allow downgrade, widen price band) is re-solved -> ranked options appear on the approval card. Margin floors,
minimum fill, compatibility and "no unmet demand" are never relaxed by the system. The LLM cannot approve anything.

**LLM, only where language is the problem.** (1) Notes box -> structured adjustments from a closed menu, validated,
capped and shown for confirmation; (2) plain-language explanation; (3) follow-up what-ifs where it chooses a tool and
arguments that code validates. Every number in LLM text must appear in tool output, otherwise the template is used.

## Backtest on the synthetic data (last 8 weeks, mean of 18 series)

| Target | WAPE XGBoost | WAPE TFT | WAPE blend | P10-P90 coverage |
|---|---|---|---|---|
| weight_kg | 12.2% | 10.9% | 10.8% | 83.5% |
| volume_m3 | 12.8% | 11.5% | 11.4% | 83.0% |
| item_count | 12.9% | - | 12.9% | 81.8% |

The TFT helping here says little about real data: the series come from a generator, so a model that learns its
patterns will look good. Re-run the backtest on real history before claiming anything.

## Known limits

- One route and one day per solve; the 7-day outlook is 7 independent solves. No multi-leg network or time-expanded model.
- Each route has its own asset pool; flights are static residual space; trailers are interchangeable.
- Demand elasticity comes from config, it is not estimated. Competitor price is held at its last value in the forecast.
- `item_count` is forecast but not used by the solver.
- The dashboard run uses a fixed tool policy; the LLM chooses tools only in follow-up questions.
- The Groq calls (notes, explanation, follow-ups) were tested with a stand-in model, not against the live API.
