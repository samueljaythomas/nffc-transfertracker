#!/usr/bin/env python3
"""
Transfer tracker for an FPL Draft league.

Projection model
----------------
Every player gets a PRIOR from preseason-knowable signals (per-90 output
across recent seasons, FPL's draft_rank, set-piece duty) and an OBSERVED
rate from this season (actual points per 90 blended with underlying
xG/xA/clean-sheet numbers). They combine Bayesian style:

    weight = minutes / (minutes + K)
    rate   = weight * observed + (1 - weight) * prior

With K_POINTS = 700, a player with 90 minutes played is ~89% prior; at
700 minutes it's an even split. Minutes get the same treatment with a
smaller K, since playing time stabilises faster than scoring.

Last-season history comes from the classic element-summary endpoint and
is cached in history_cache.json, which the workflow commits.
"""

import json
import math
import os
import time
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

LEAGUE_ID = int(os.environ.get("LEAGUE_ID", "6206"))
MY_TEAM = os.environ.get("MY_TEAM", "Nottinghamburglars")
TZ = ZoneInfo(os.environ.get("TZ_NAME", "America/Chicago"))
LONG_HORIZON = int(os.environ.get("LONG_HORIZON", "5"))

# Bayesian shrinkage. Bigger K = trust this season more slowly.
K_POINTS = float(os.environ.get("K_POINTS", "700"))
K_MINUTES = float(os.environ.get("K_MINUTES", "320"))

W_UNDERLYING = 0.45      # within observed: underlying vs raw points
W_LAST_SEASON = 0.65     # within prior: history vs rank curve

FIXTURE_CLAMP = (0.78, 1.28)
SWAP_THRESHOLD = float(os.environ.get("SWAP_THRESHOLD", "0.8"))

CACHE_FILE = "history_cache.json"
MAX_HISTORY_FETCH = int(os.environ.get("MAX_HISTORY_FETCH", "220"))
HISTORY_SLEEP = 0.22

POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
GOAL_PTS = {1: 10, 2: 6, 3: 5, 4: 4}
CS_PTS = {1: 4, 2: 4, 3: 1, 4: 0}
DEFCON_THRESHOLD = {1: 999, 2: 10, 3: 12, 4: 12}
ATT_SHARE = {1: 0.05, 2: 0.35, 3: 0.75, 4: 0.95}

DRAFT = "https://draft.premierleague.com/api"
CLASSIC = "https://fantasy.premierleague.com/api"

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (compatible; fpl-transfer-tracker/2.0)",
    "Accept": "application/json",
})


def get(url):
    r = session.get(url, timeout=40)
    r.raise_for_status()
    return r.json()


def fnum(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _key(s):
    return "".join(c for c in (s or "").lower() if c.isalnum())


print("Fetching draft data...")
draft_boot = get(f"{DRAFT}/bootstrap-static")
details = get(f"{DRAFT}/league/{LEAGUE_ID}/details")
choices = get(f"{DRAFT}/draft/{LEAGUE_ID}/choices")

print("Fetching classic data...")
classic_boot = get(f"{CLASSIC}/bootstrap-static/")
fixtures = get(f"{CLASSIC}/fixtures/")

league_name = details["league"]["name"]
entries = {e["id"]: e["entry_name"] for e in details["league_entries"]}

me = next((e for e in details["league_entries"]
           if _key(e["entry_name"]) == _key(MY_TEAM)), None)
if me is None:
    me = next((e for e in details["league_entries"]
               if _key(MY_TEAM) and _key(MY_TEAM) in _key(e["entry_name"])), None)
if me is None:
    raise SystemExit(f"Could not find a team called {MY_TEAM!r}. "
                     f"Teams in this league: {list(entries.values())}")

my_ids = {v for v in (me.get("id"), me.get("entry_id")) if v is not None}
MY_TEAM = me["entry_name"]

owner_of = {es["element"]: es.get("owner")
            for es in choices.get("element_status", [])}

classic_by_code = {e["code"]: e for e in classic_boot["elements"]}
classic_teams = {t["id"]: t for t in classic_boot.get("teams", [])}
draft_teams = {t["id"]: t for t in draft_boot.get("teams", [])}

classic_team_by_name = {}
for cid, t in classic_teams.items():
    classic_team_by_name[_key(t.get("name"))] = cid
    classic_team_by_name[_key(t.get("short_name"))] = cid

draft_to_classic_team = {}
for did, t in draft_teams.items():
    cid = (classic_team_by_name.get(_key(t.get("name")))
           or classic_team_by_name.get(_key(t.get("short_name"))))
    if cid:
        draft_to_classic_team[did] = cid

team_name = {did: (t.get("short_name") or t.get("name"))
             for did, t in draft_teams.items()}


def avg(vals):
    vals = [v for v in vals if v]
    return sum(vals) / len(vals) if vals else 1.0


AVG_ATT = avg([t.get("strength_attack_home") for t in classic_teams.values()]
              + [t.get("strength_attack_away") for t in classic_teams.values()])
AVG_DEF = avg([t.get("strength_defence_home") for t in classic_teams.values()]
              + [t.get("strength_defence_away") for t in classic_teams.values()])

finished_events = [f["event"] for f in fixtures
                   if f.get("finished") and f.get("event")]
last_done = max(finished_events) if finished_events else 0
gws_done = max(1, last_done)
upcoming = sorted({f["event"] for f in fixtures
                   if f.get("event") and f["event"] > last_done})
next_gw = upcoming[0] if upcoming else None
horizon = upcoming[:LONG_HORIZON]

team_fix = defaultdict(lambda: defaultdict(list))
for f in fixtures:
    gw = f.get("event")
    if not gw or gw <= last_done:
        continue
    team_fix[f["team_h"]][gw].append((f["team_a"], True))
    team_fix[f["team_a"]][gw].append((f["team_h"], False))


def fixture_factor(draft_team_id, gws, pos_id):
    """Expected fixtures per gameweek, weighted by opponent strength."""
    cid = draft_to_classic_team.get(draft_team_id)
    if not cid or not gws:
        return 1.0, []
    att_share = ATT_SHARE[pos_id]
    total, detail = 0.0, []
    for gw in gws:
        games = team_fix[cid].get(gw, [])
        wk, chips = 0.0, []
        for opp_id, is_home in games:
            opp = classic_teams.get(opp_id, {})
            opp_def = opp.get("strength_defence_away" if is_home
                              else "strength_defence_home") or AVG_DEF
            opp_att = opp.get("strength_attack_away" if is_home
                              else "strength_attack_home") or AVG_ATT
            mult = (att_share * (AVG_DEF / opp_def)
                    + (1 - att_share) * (AVG_ATT / opp_att))
            mult *= 1.05 if is_home else 0.96
            mult = max(FIXTURE_CLAMP[0], min(FIXTURE_CLAMP[1], mult))
            wk += mult
            chips.append((opp.get("short_name", "?"), is_home, mult))
        total += wk
        detail.append((gw, chips))
    return total / len(gws), detail


if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, encoding="utf-8") as fh:
        cache = json.load(fh)
else:
    cache = {"version": 2, "seasons": {}, "checked": []}
cache.setdefault("seasons", {})
cache.setdefault("checked", [])
checked = set(cache["checked"])

wanted = sorted(draft_boot["elements"], key=lambda e: e.get("draft_rank") or 9999)
to_fetch = [e for e in wanted
            if str(e["code"]) not in checked and e["code"] in classic_by_code]

fetched = 0
for el in to_fetch[:MAX_HISTORY_FETCH]:
    cid = classic_by_code[el["code"]]["id"]
    try:
        summary = get(f"{CLASSIC}/element-summary/{cid}/")
    except Exception as exc:
        print(f"  history fetch failed for {el.get('web_name')}: {exc}")
        continue
    cache["seasons"][str(el["code"])] = [
        {"season": h.get("season_name"), "minutes": h.get("minutes", 0),
         "points": h.get("total_points", 0), "starts": h.get("starts", 0)}
        for h in summary.get("history_past", [])
    ]
    checked.add(str(el["code"]))
    fetched += 1
    time.sleep(HISTORY_SLEEP)

cache["checked"] = sorted(checked)
with open(CACHE_FILE, "w", encoding="utf-8") as fh:
    json.dump(cache, fh, separators=(",", ":"))
print(f"History: {fetched} fetched this run, {len(checked)} cached, "
      f"{max(0, len(to_fetch) - fetched)} still to collect")


def past_rates(code):
    """Weighted per-90 points and minutes-per-week from recent seasons."""
    rows = [r for r in cache["seasons"].get(str(code), []) if r["minutes"] >= 400]
    if not rows:
        return None, None, 0
    rows = rows[-3:]
    weights = [0.15, 0.3, 0.55][-len(rows):]
    tot_w = sum(weights)
    p90 = sum(w * (r["points"] / r["minutes"] * 90)
              for w, r in zip(weights, rows)) / tot_w
    mpg = sum(w * (r["minutes"] / 38.0) for w, r in zip(weights, rows)) / tot_w
    return p90, mpg, sum(r["minutes"] for r in rows)


samples = defaultdict(list)
for el in draft_boot["elements"]:
    rank = el.get("draft_rank")
    if not rank:
        continue
    p90, _m, mins = past_rates(el["code"])
    if p90 is None or mins < 900:
        continue
    samples[el["element_type"]].append((math.log(rank), p90))

rank_fit, pos_avg_p90 = {}, {}
for pos_id, pts in samples.items():
    pos_avg_p90[pos_id] = sum(y for _, y in pts) / len(pts)
    if len(pts) < 8:
        continue
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    denom = sum((x - mx) ** 2 for x, _ in pts)
    if denom <= 0:
        continue
    b = sum((x - mx) * (y - my) for x, y in pts) / denom
    rank_fit[pos_id] = (my - b * mx, b, n)

if rank_fit:
    print("Rank curve fitted: "
          + ", ".join(f"{POS[p]}(n={rank_fit[p][2]})" for p in sorted(rank_fit)))
else:
    print("Rank curve: not enough cached history yet, using position averages")


def rank_p90(pos_id, rank):
    if rank and pos_id in rank_fit:
        a, b, _ = rank_fit[pos_id]
        return max(0.5, a + b * math.log(rank))
    return pos_avg_p90.get(pos_id, 3.0)


team_pos_minutes = defaultdict(float)
for el in draft_boot["elements"]:
    team_pos_minutes[(el["team"], el["element_type"])] += (el.get("minutes") or 0)


def set_piece_bonus(el, pos_id):
    bonus, tags = 0.0, []
    if el.get("penalties_order") == 1 and pos_id != 1:
        bonus += 0.45
        tags.append("pens")
    if el.get("direct_freekicks_order") == 1:
        bonus += 0.12
        tags.append("FKs")
    if el.get("corners_and_indirect_freekicks_order") == 1:
        bonus += 0.18
        tags.append("corners")
    return bonus, tags


def project(el):
    pos_id = el["element_type"]
    mins = el.get("minutes") or 0
    starts = el.get("starts") or 0
    tp = el.get("total_points") or 0
    rank = el.get("draft_rank")

    status = el.get("status", "a")
    chance = el.get("chance_of_playing_next_round")
    if status in ("i", "s", "u", "n"):
        return dict(proj=0.0, prior_share=1.0, prior_p90=0.0, obs_p90=0.0,
                    exp_min=0.0, notes="unavailable", tags=[], hist_mins=0)
    avail = 1.0 if chance is None else max(0.0, chance / 100.0)

    hist_p90, hist_mpg, hist_mins = past_rates(el["code"])
    curve_p90 = rank_p90(pos_id, rank)
    if hist_p90 is None:
        prior_p90 = curve_p90
        prior_mpg = 45.0 if (rank or 999) < 250 else 20.0
    else:
        hw = W_LAST_SEASON * min(1.0, hist_mins / 1800.0)
        prior_p90 = hw * hist_p90 + (1 - hw) * curve_p90
        prior_mpg = hist_mpg

    sp_bonus, sp_tags = set_piece_bonus(el, pos_id)
    prior_p90 += sp_bonus

    if mins >= 45:
        per90 = 90.0 / mins
        actual_p90 = tp * per90
        xg90 = fnum(el.get("expected_goals")) * per90
        xa90 = fnum(el.get("expected_assists")) * per90
        xgc90 = fnum(el.get("expected_goals_conceded")) * per90
        dc90 = (el.get("defensive_contribution") or 0) * per90
        sv90 = (el.get("saves") or 0) * per90
        und = xg90 * GOAL_PTS[pos_id] + xa90 * 3.0
        if CS_PTS[pos_id]:
            und += math.exp(-max(0.0, xgc90)) * CS_PTS[pos_id]
        if pos_id == 1:
            und += sv90 / 3.0
        thr = DEFCON_THRESHOLD[pos_id]
        if thr < 100 and dc90 > 0:
            und += 2.0 * min(1.0, dc90 / thr) ** 2
        obs_p90 = (1 - W_UNDERLYING) * actual_p90 + W_UNDERLYING * und
    else:
        obs_p90 = prior_p90

    kp = mins / (mins + K_POINTS)
    p90 = kp * obs_p90 + (1 - kp) * prior_p90

    obs_mpg = mins / gws_done
    km = mins / (mins + K_MINUTES)
    exp_min = km * obs_mpg + (1 - km) * prior_mpg
    exp_min = min(90.0, max(0.0, exp_min)) * avail

    pool = team_pos_minutes.get((el["team"], pos_id), 0)
    if pool > 0 and mins > 0:
        share = mins / pool
        expected_share = {1: 0.5, 2: 0.22, 3: 0.2, 4: 0.3}[pos_id]
        if share < expected_share * 0.55 and mins < 450:
            exp_min *= 0.82

    appearance = (min(1.0, exp_min / 15.0)
                  + max(0.0, min(1.0, (exp_min - 20) / 50.0)))
    proj = p90 * (exp_min / 90.0) + appearance

    notes = []
    if status == "d":
        notes.append(el.get("news") or "doubtful")
    if starts and mins > 0 and starts / gws_done < 0.6:
        notes.append("rotation risk")
    if kp < 0.25:
        notes.append(f"{int((1 - kp) * 100)}% prior")

    return dict(proj=max(0.0, proj), prior_share=1 - kp, prior_p90=prior_p90,
                obs_p90=obs_p90, exp_min=exp_min, notes=", ".join(notes),
                tags=sp_tags, hist_mins=hist_mins or 0)


players = {}
for el in draft_boot["elements"]:
    r = project(el)
    pos_id = el["element_type"]
    short_f, short_detail = fixture_factor(
        el["team"], [next_gw] if next_gw else [], pos_id)
    long_f, long_detail = fixture_factor(el["team"], horizon, pos_id)
    players[el["id"]] = {
        "id": el["id"], "name": el.get("web_name", "?"),
        "pos": POS.get(pos_id, "?"), "pos_id": pos_id,
        "team": team_name.get(el["team"], "?"),
        "owner": owner_of.get(el["id"]),
        "base": r["proj"],
        "short": r["proj"] * short_f,
        "long": r["proj"] * long_f,
        "short_fix": short_detail, "long_fix": long_detail,
        "prior_share": r["prior_share"], "prior_p90": r["prior_p90"],
        "obs_p90": r["obs_p90"], "exp_min": r["exp_min"],
        "notes": r["notes"], "tags": r["tags"], "hist_mins": r["hist_mins"],
        "minutes": el.get("minutes") or 0,
        "total_points": el.get("total_points") or 0,
        "rank": el.get("draft_rank"), "status": el.get("status", "a"),
    }

roster = [p for p in players.values() if p["owner"] in my_ids]
free = [p for p in players.values()
        if p["owner"] is None and p["status"] != "u"]


def build_swaps(key):
    out, by_pos = [], defaultdict(list)
    for p in free:
        by_pos[p["pos_id"]].append(p)
    for pos_id, pool in by_pos.items():
        pool = sorted(pool, key=lambda p: -p[key])[:6]
        mine = sorted([p for p in roster if p["pos_id"] == pos_id],
                      key=lambda p: p[key])
        if not mine:
            continue
        for cand in pool:
            gain = cand[key] - mine[0][key]
            if gain >= SWAP_THRESHOLD:
                out.append({"add": cand, "drop": mine[0], "gain": gain})
    out.sort(key=lambda s: -s["gain"])
    seen, dedup = set(), []
    for s in out:
        if s["add"]["id"] in seen:
            continue
        seen.add(s["add"]["id"])
        dedup.append(s)
    return dedup[:6]


swaps_short = build_swaps("short")
swaps_long = build_swaps("long")


# ------------------------------------------------------------------ html
def fixture_chips(detail):
    if not detail:
        return ""
    bits = []
    for gw, chips in detail:
        if not chips:
            bits.append(f'<i class="d0" title="GW{gw} blank">&mdash;</i>')
            continue
        for opp, home, mult in chips:
            cls = ("d1" if mult >= 1.15 else "d2" if mult >= 1.04
                   else "d3" if mult >= 0.96 else "d4" if mult >= 0.87 else "d5")
            label = opp.upper() if home else opp.lower()
            bits.append(f'<i class="{cls}" title="GW{gw} v {opp}">{label}</i>')
    return f'<span class="fixes">{"".join(bits)}</span>'


def confidence_bar(prior_share):
    pct = int(round((1 - prior_share) * 100))
    return (f'<span class="conf" title="{pct}% of this number comes from this '
            f'season, the rest from the prior">'
            f'<i style="width:{max(3, pct)}%"></i></span>')


def player_row(p, key):
    tags = "".join(f'<em class="tag-sp">{t}</em>' for t in p["tags"])
    note = f'<div class="note-inline">{p["notes"]}</div>' if p["notes"] else ""
    rank = f'#{p["rank"]}' if p["rank"] else "&mdash;"
    return (
        f'<div class="row">'
        f'<div class="pos p{p["pos_id"]}">{p["pos"]}</div>'
        f'<div class="who"><div class="pname">{p["name"]}'
        f'<span class="club">{p["team"]}</span>{tags}</div>'
        f'<div class="meta">rank {rank} &middot; {p["exp_min"]:.0f}\' expected '
        f'&middot; {p["minutes"]}\' played &middot; {p["total_points"]}pts</div>'
        f'{confidence_bar(p["prior_share"])}{note}</div>'
        f'{fixture_chips(p["long_fix"])}'
        f'<div class="val"><div class="num">{p[key]:.1f}</div>'
        f'<div class="unit">pts/gw</div></div></div>'
    )


def why(a, d):
    bits = []
    if a["prior_p90"] > d["prior_p90"] + 0.4:
        bits.append("better track record")
    if a["obs_p90"] > d["obs_p90"] + 0.6 and a["minutes"] > 150:
        bits.append("scoring better this season")
    if a["exp_min"] > d["exp_min"] + 12:
        bits.append("more secure minutes")
    if (a["base"] > 0 and d["base"] > 0
            and a["long"] / a["base"] > d["long"] / d["base"] + 0.05):
        bits.append("kinder fixtures")
    if a["tags"]:
        bits.append("on " + "/".join(a["tags"]))
    return " &middot; ".join(bits) or "higher projection overall"


def swap_card(s, key):
    a, d = s["add"], s["drop"]
    warn = (f'<div class="note-inline warn">{a["notes"]}</div>'
            if a["notes"] else "")
    return (
        f'<div class="swap">'
        f'<div class="swaphead"><span class="gain">+{s["gain"]:.1f}</span>'
        f'<span class="swaplabel">pts per gameweek</span></div>'
        f'<div class="leg add"><span class="tag">ADD</span>'
        f'<span class="pn">{a["name"]}</span>'
        f'<span class="cl">{a["pos"]} &middot; {a["team"]}</span>'
        f'<span class="pv">{a[key]:.1f}</span></div>'
        f'<div class="legfix">{fixture_chips(a["long_fix"])}</div>'
        f'<div class="leg drop"><span class="tag">DROP</span>'
        f'<span class="pn">{d["name"]}</span>'
        f'<span class="cl">{d["pos"]} &middot; {d["team"]}</span>'
        f'<span class="pv">{d[key]:.1f}</span></div>'
        f'<div class="whyline">{why(a, d)}</div>{warn}</div>'
    )


def swaps_pane(swaps, key, blurb):
    if not swaps:
        return (f'<section class="card"><div class="head">'
                f'<h2>No moves suggested</h2><p>{blurb}</p></div>'
                f'<div class="empty">Nothing available clears the '
                f'{SWAP_THRESHOLD} pts/gw threshold.</div></section>')
    return (f'<section class="card"><div class="head"><h2>Suggested moves</h2>'
            f'<p>{blurb}</p></div>'
            f'{"".join(swap_card(s, key) for s in swaps)}</section>')


roster_by_pos = defaultdict(list)
for p in roster:
    roster_by_pos[p["pos_id"]].append(p)

roster_html = ""
for pos_id in (1, 2, 3, 4):
    group = sorted(roster_by_pos.get(pos_id, []), key=lambda p: -p["long"])
    if group:
        roster_html += (f'<section class="card"><div class="head">'
                        f'<h2>{POS[pos_id]}</h2><p>{len(group)} on roster</p>'
                        f'</div>{"".join(player_row(p, "long") for p in group)}'
                        f'</section>')
if not roster_html:
    roster_html = ('<section class="card"><div class="head">'
                   '<h2>Roster not found</h2><p>Check MY_TEAM</p></div></section>')

free_html = ""
for pos_id in (1, 2, 3, 4):
    group = sorted([p for p in free if p["pos_id"] == pos_id],
                   key=lambda p: -p["long"])[:10]
    if group:
        free_html += (f'<section class="card"><div class="head">'
                      f'<h2>{POS[pos_id]}</h2><p>Top {len(group)} available</p>'
                      f'</div>{"".join(player_row(p, "long") for p in group)}'
                      f'</section>')

updated = datetime.now(TZ).strftime("%a %d %b, %I:%M %p").replace(" 0", " ")
gw_label = f"GW{next_gw}" if next_gw else "season end"
horizon_label = (f"GW{horizon[0]}&ndash;{horizon[-1]}"
                 if len(horizon) > 1 else gw_label)
rel = [p for p in players.values() if p["minutes"] > 0]
avg_prior = (sum(p["prior_share"] for p in rel) / len(rel)) if rel else 1.0

CSS = """
  * { box-sizing: border-box; }
  body {
    margin: 0;
    padding: calc(20px + env(safe-area-inset-top)) 13px
             calc(44px + env(safe-area-inset-bottom)) 13px;
    background: radial-gradient(900px 380px at 50% -170px, #10233a 0%, transparent 70%), #0a0c11;
    color: #f2f5fa;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  .wrap { max-width: 470px; margin: 0 auto; }
  .title { font-size: 25px; font-weight: 800; letter-spacing: -.025em; }
  .stampbar { display: flex; align-items: center; gap: 10px; margin: 8px 0 15px; }
  .stamp { font-size: 12px; color: #838ea3; flex: 1; }
  .stamp b { color: #4d9fff; font-weight: 600; }
  .reload { flex: none; width: 30px; height: 30px; border-radius: 9px;
    background: #141922; border: 1px solid #252c3a; color: #838ea3;
    font-size: 16px; font-family: inherit; }
  .tabs { display: flex; gap: 4px; background: #141922; border: 1px solid #252c3a;
    border-radius: 12px; padding: 4px; margin-bottom: 15px;
    overflow-x: auto; scrollbar-width: none; }
  .tabs::-webkit-scrollbar { display: none; }
  .tab { flex: 1 0 auto; padding: 9px 12px; border-radius: 9px; white-space: nowrap;
    font-size: 12.5px; font-weight: 600; color: #838ea3; border: none;
    background: none; font-family: inherit; }
  .tab.on { background: #2f7fe0; color: #fff; }
  .pane { display: none; }
  .pane.on { display: block; }
  .card { background: #141922; border: 1px solid #252c3a; border-radius: 15px;
    overflow: hidden; margin-bottom: 14px; }
  .head { padding: 14px 16px 11px; border-bottom: 1px solid #252c3a; }
  .head h2 { margin: 0; font-size: 17px; font-weight: 700; }
  .head p { margin: 3px 0 0; font-size: 12px; color: #838ea3; }
  .row { display: flex; align-items: center; gap: 10px; padding: 11px 14px;
    border-bottom: 1px solid #1c2029; }
  .row:last-of-type { border-bottom: none; }
  .pos { flex: none; width: 32px; text-align: center; font-size: 10px;
    font-weight: 800; padding: 3px 0; border-radius: 6px; }
  .p1 { background: #3b2f00; color: #f0c14b; }
  .p2 { background: #10304a; color: #6cc0ff; }
  .p3 { background: #113626; color: #4fd490; }
  .p4 { background: #3c1420; color: #ff7a92; }
  .who { flex: 1; min-width: 0; }
  .pname { font-size: 14px; font-weight: 600; }
  .club { font-size: 10.5px; color: #838ea3; margin-left: 6px; font-weight: 500; }
  .tag-sp { font-style: normal; font-size: 9px; font-weight: 700; margin-left: 5px;
    background: #2a2233; color: #c9a6f0; padding: 2px 5px; border-radius: 4px; }
  .meta { font-size: 10.5px; color: #838ea3; margin-top: 2px; }
  .conf { display: block; height: 3px; background: #232a36; border-radius: 2px;
    margin-top: 5px; max-width: 120px; overflow: hidden; }
  .conf i { display: block; height: 3px; background: #2f7fe0; }
  .note-inline { font-size: 10.5px; color: #f0b429; margin-top: 3px; }
  .note-inline.warn { padding: 6px 0 0; }
  .fixes { display: flex; gap: 2px; flex: none; }
  .fixes i { display: block; min-width: 24px; height: 15px; border-radius: 3px;
    font-size: 8.5px; font-weight: 700; line-height: 15px; text-align: center;
    font-style: normal; padding: 0 2px; }
  .d1 { background: #1b5e35; color: #9ff5c0; }
  .d2 { background: #24523a; color: #a8e8c0; }
  .d3 { background: #2c313c; color: #aab4c5; }
  .d4 { background: #5c2130; color: #ffb0be; }
  .d5 { background: #74152a; color: #ffc2cd; }
  .d0 { background: #1c2029; color: #5a6478; }
  .val { text-align: right; min-width: 44px; flex: none; }
  .num { font-size: 17px; font-weight: 800; font-variant-numeric: tabular-nums; }
  .unit { font-size: 9px; color: #838ea3; }
  .swap { border-bottom: 1px solid #1c2029; padding: 12px 14px 13px; }
  .swap:last-of-type { border-bottom: none; }
  .swaphead { display: flex; align-items: center; gap: 8px; margin-bottom: 9px; }
  .gain { background: #1e5f3a; color: #8bf0b5; font-size: 12px; font-weight: 800;
    padding: 3px 9px; border-radius: 20px; }
  .swaplabel { font-size: 10.5px; color: #838ea3; text-transform: uppercase;
    letter-spacing: .07em; }
  .leg { display: flex; align-items: center; gap: 8px; padding: 5px 0; }
  .legfix { padding: 2px 0 4px 50px; }
  .tag { font-size: 9px; font-weight: 800; padding: 3px 6px; border-radius: 5px;
    flex: none; width: 42px; text-align: center; }
  .add .tag { background: #14432c; color: #6ee7a6; }
  .drop .tag { background: #45202a; color: #ff9aad; }
  .pn { font-size: 14px; font-weight: 600; flex: 1; min-width: 0; }
  .cl { font-size: 10.5px; color: #838ea3; flex: none; }
  .pv { font-size: 14px; font-weight: 700; flex: none; min-width: 34px;
    text-align: right; font-variant-numeric: tabular-nums; }
  .whyline { font-size: 11px; color: #7fd4a2; margin-top: 7px; }
  .empty { padding: 30px 18px; text-align: center; font-size: 13px; color: #838ea3; }
  .note { padding: 12px 16px; font-size: 11.5px; color: #838ea3; line-height: 1.5;
    border-top: 1px solid #1c2029; }
  .note b { color: #f2f5fa; }
  footer { text-align: center; font-size: 11px; color: #4a5468; margin-top: 24px; }
"""

SCRIPT = """
<script>
(function () {
  var tabs = document.querySelectorAll('.tab');
  var panes = document.querySelectorAll('.pane');
  var order = ['moves', 'roster', 'free', 'model'];
  function show(n) {
    tabs.forEach(function (t) { t.classList.toggle('on', t.dataset.tab === n); });
    panes.forEach(function (p) { p.classList.toggle('on', p.dataset.pane === n); });
    try { history.replaceState(null, '', '#' + n); } catch (e) {}
    window.scrollTo(0, 0);
  }
  tabs.forEach(function (t) {
    t.addEventListener('click', function () { show(t.dataset.tab); });
  });
  var r = document.getElementById('reload');
  if (r) r.addEventListener('click', function () {
    location.replace(location.pathname + '?t=' + Date.now() + location.hash);
  });
  var s = (location.hash || '').replace('#', '');
  show(order.indexOf(s) >= 0 ? s : 'moves');
})();
</script>
"""

MODEL_NOTES = f"""
      <div class="note">
      Every player has a <b>prior</b> &mdash; what we'd expect before this
      season started &mdash; built from their points per 90 across the last
      three seasons (weighted toward the most recent), FPL's own
      <b>draft rank</b>, and set-piece duty. Penalty takers get a bump, as do
      first-choice corner and free-kick takers.
      </div>
      <div class="note">
      Against that sits the <b>observed</b> rate from this season: actual
      points per 90 blended with underlying numbers &mdash; expected goals and
      assists, clean-sheet odds from expected goals conceded, save rate for
      keepers, and defensive contributions against the position threshold.
      </div>
      <div class="note">
      The two combine according to how much football has actually been played:
      <b>weight = minutes &divide; (minutes + {int(K_POINTS)})</b>. At 90
      minutes played a player sits about 89% on the prior. At {int(K_POINTS)}
      minutes it's an even split. Playing time blends the same way with a
      smaller constant ({int(K_MINUTES)}), because minutes settle faster than
      scoring does. The blue bar under each name shows how much of that
      player's number comes from this season rather than the prior.
      </div>
      <div class="note">
      Fixtures use each opponent's <b>attack and defence strength ratings</b>,
      home and away, rather than the blunt 1&ndash;5 difficulty score.
      Forwards and midfielders are weighted toward the opponent's defensive
      rating; keepers and defenders toward their attacking rating. Chips show
      the opponent &mdash; uppercase for home, lowercase for away.
      </div>
      <div class="note">
      Last-season history is cached in the repo and fetched once per player,
      so the first few builds fill it in gradually. Only same-position swaps
      are offered, since draft squads stay legal at 2/5/5/3.
      </div>
"""

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Transfer tracker</title>
<meta name="theme-color" content="#0a0c11">
<meta name="robots" content="noindex, nofollow">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="Transfers">
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
  <div class="title">{MY_TEAM}</div>
  <div class="stampbar">
    <div class="stamp">Transfer tracker &middot; next <b>{gw_label}</b>
      &middot; updated {updated}</div>
    <button id="reload" class="reload">&#8635;</button>
  </div>

  <div class="tabs">
    <button class="tab on" data-tab="moves">Moves</button>
    <button class="tab" data-tab="roster">My roster</button>
    <button class="tab" data-tab="free">Free agents</button>
    <button class="tab" data-tab="model">Model</button>
  </div>

  <div class="pane on" data-pane="moves">
    {swaps_pane(swaps_short, "short", f"Best single-week move for {gw_label}")}
    {swaps_pane(swaps_long, "long", f"Best move across {horizon_label}")}
  </div>

  <div class="pane" data-pane="roster">{roster_html}</div>
  <div class="pane" data-pane="free">{free_html}</div>

  <div class="pane" data-pane="model">
    <section class="card"><div class="head"><h2>How the projection works</h2>
      <p>Currently {int(avg_prior * 100)}% prior on average</p></div>
      {MODEL_NOTES}
    </section>
  </div>

  <footer>Rebuilt twice daily &middot; prior-weighted projections</footer>
</div>
{SCRIPT}
</body>
</html>
"""

with open("index.html", "w", encoding="utf-8") as fh:
    fh.write(html)

owned_total = sum(1 for p in players.values() if p["owner"] is not None)
print(f"Matched team {MY_TEAM!r} -> ids {sorted(my_ids)}")
print(f"Roster: {len(roster)} | owned league-wide: {owned_total} "
      f"| free agents: {len(free)}")
print(f"Average prior weight (players with minutes): {avg_prior:.0%}")
print(f"Suggestions: {len(swaps_short)} short, {len(swaps_long)} long")
print(f"Built index.html for {league_name}, next {gw_label}")
