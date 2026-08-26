#!/usr/bin/env python3
"""
Builds index.html - a private transfer tracker for an FPL Draft league.

Pulls:
  draft.premierleague.com/api/bootstrap-static      player stats (incl. xG/xA)
  draft.premierleague.com/api/league/{id}/details   who's in the league
  draft.premierleague.com/api/draft/{id}/choices    ownership (who is free)
  fantasy.premierleague.com/api/bootstrap-static/   team names for joining
  fantasy.premierleague.com/api/fixtures/           fixture difficulty

Projects points per gameweek for every player, then suggests same-position
swaps between your roster and the free agent pool.
"""

import json
import math
import os
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

# ------------------------------------------------------------------ config
LEAGUE_ID = int(os.environ.get("LEAGUE_ID", "6206"))
MY_TEAM = os.environ.get("MY_TEAM", "Nottinghamburglars")
TZ = ZoneInfo(os.environ.get("TZ_NAME", "America/Chicago"))
LONG_HORIZON = int(os.environ.get("LONG_HORIZON", "5"))

# How much each signal counts toward the projection. Tune freely.
W_HISTORY = 0.40      # season points per 90, scaled by expected minutes
W_FORM = 0.30         # recent scoring
W_UNDERLYING = 0.30   # xG / xA / clean sheet odds / defensive contributions

# Points per fixture-difficulty step. 0.06 means a difficulty-1 fixture is
# worth ~12% more than a neutral one, difficulty-5 about 12% less.
FIXTURE_SWING = 0.06

# Minimum projected gain before a swap is worth suggesting (points per GW).
SWAP_THRESHOLD = 0.8

POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
GOAL_PTS = {1: 10, 2: 6, 3: 5, 4: 4}
CS_PTS = {1: 4, 2: 4, 3: 1, 4: 0}
DEFCON_THRESHOLD = {1: 99, 2: 10, 3: 12, 4: 12}   # 99 = not applicable

DRAFT = "https://draft.premierleague.com/api"
CLASSIC = "https://fantasy.premierleague.com/api"

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (compatible; fpl-transfer-tracker/1.0)",
    "Accept": "application/json",
})


def get(url):
    r = session.get(url, timeout=40)
    r.raise_for_status()
    return r.json()


# ------------------------------------------------------------------ fetch
print("Fetching draft data...")
draft_boot = get(f"{DRAFT}/bootstrap-static")
details = get(f"{DRAFT}/league/{LEAGUE_ID}/details")
choices = get(f"{DRAFT}/draft/{LEAGUE_ID}/choices")

print("Fetching fixtures...")
classic_boot = get(f"{CLASSIC}/bootstrap-static/")
fixtures = get(f"{CLASSIC}/fixtures/")

league_name = details["league"]["name"]
entries = {e["id"]: e["entry_name"] for e in details["league_entries"]}
my_id = next((i for i, n in entries.items()
              if n.strip().lower() == MY_TEAM.strip().lower()), None)
if my_id is None:
    raise SystemExit(f"Could not find a team called {MY_TEAM!r}. "
                     f"Teams in this league: {list(entries.values())}")

owner_of = {es["element"]: es.get("owner")
            for es in choices.get("element_status", [])}

# ------------------------------------------------------- team name joining
# Draft and classic use their own team ids, so join on name rather than id.
draft_teams = {t["id"]: t for t in draft_boot.get("teams", [])}
classic_teams = {t["id"]: t for t in classic_boot.get("teams", [])}


def norm(s):
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())


classic_by_name = {}
for cid, t in classic_teams.items():
    classic_by_name[norm(t.get("name"))] = cid
    classic_by_name[norm(t.get("short_name"))] = cid

draft_to_classic = {}
for did, t in draft_teams.items():
    cid = classic_by_name.get(norm(t.get("name"))) \
        or classic_by_name.get(norm(t.get("short_name")))
    if cid:
        draft_to_classic[did] = cid

team_name = {did: (t.get("short_name") or t.get("name"))
             for did, t in draft_teams.items()}

# ------------------------------------------------------------- gameweeks
finished_events = [f["event"] for f in fixtures
                   if f.get("finished") and f.get("event")]
last_done = max(finished_events) if finished_events else 0
upcoming = sorted({f["event"] for f in fixtures
                   if f.get("event") and f["event"] > last_done})
next_gw = upcoming[0] if upcoming else None
horizon = upcoming[:LONG_HORIZON]

# classic team id -> {gw: [difficulty, ...]}
team_fix = defaultdict(lambda: defaultdict(list))
for f in fixtures:
    gw = f.get("event")
    if not gw or gw <= last_done:
        continue
    team_fix[f["team_h"]][gw].append(f.get("team_h_difficulty") or 3)
    team_fix[f["team_a"]][gw].append(f.get("team_a_difficulty") or 3)


def fixture_factor(draft_team_id, gws):
    """Expected fixtures-per-gameweek, weighted by difficulty."""
    cid = draft_to_classic.get(draft_team_id)
    if not cid or not gws:
        return 1.0, []
    total, detail = 0.0, []
    for gw in gws:
        diffs = team_fix[cid].get(gw, [])
        wk = sum(1 + (3 - d) * FIXTURE_SWING for d in diffs)
        total += wk
        detail.append((gw, diffs))
    return total / len(gws), detail


# ------------------------------------------------------------ projections
gws_done = max(1, last_done)


def project(el):
    """Return (proj_per_gw_neutral, confidence 0-1, notes)."""
    pos = el["element_type"]
    mins = el.get("minutes") or 0
    starts = el.get("starts") or 0
    tp = el.get("total_points") or 0

    status = el.get("status", "a")
    chance = el.get("chance_of_playing_next_round")
    if status in ("i", "s", "u", "n"):
        return 0.0, 0.0, "unavailable"
    avail = 1.0 if chance is None else max(0.0, chance / 100.0)

    exp_mins = min(90.0, mins / gws_done) if gws_done else 0.0
    minute_share = exp_mins / 90.0

    p90 = (tp / mins * 90.0) if mins >= 60 else 0.0
    hist = p90 * minute_share

    try:
        form = float(el.get("form") or 0)
    except ValueError:
        form = 0.0

    # underlying per 90, converted to points
    def f(key):
        try:
            return float(el.get(key) or 0)
        except ValueError:
            return 0.0

    if mins >= 60:
        per90 = 90.0 / mins
        xg90 = f("expected_goals") * per90
        xa90 = f("expected_assists") * per90
        xgc90 = f("expected_goals_conceded") * per90
        dc90 = (el.get("defensive_contribution") or 0) * per90
        saves90 = (el.get("saves") or 0) * per90
    else:
        xg90 = xa90 = xgc90 = dc90 = saves90 = 0.0

    und = xg90 * GOAL_PTS[pos] + xa90 * 3.0
    if CS_PTS[pos]:
        und += math.exp(-xgc90) * CS_PTS[pos]
    if pos == 1:
        und += saves90 / 3.0
    thresh = DEFCON_THRESHOLD[pos]
    if thresh < 90 and dc90 > 0:
        # rough odds of clearing the defensive-contribution threshold
        und += 2.0 * min(1.0, dc90 / thresh) ** 2
    und += 2.0 if minute_share > 0.6 else 1.0     # appearance points
    und *= minute_share

    proj = W_HISTORY * hist + W_FORM * form + W_UNDERLYING * und
    proj *= avail

    # confidence: how much football we've actually seen
    conf = min(1.0, mins / 450.0)
    if conf < 1.0:
        # shrink toward a modest baseline when the sample is thin
        proj = proj * conf + (proj * 0.6) * (1 - conf)

    notes = []
    if status == "d":
        notes.append(el.get("news") or "doubtful")
    if starts and gws_done and starts / gws_done < 0.6:
        notes.append("rotation risk")
    if mins < 180:
        notes.append("small sample")
    return proj, conf, ", ".join(notes)


players = {}
for el in draft_boot["elements"]:
    proj, conf, notes = project(el)
    short_f, short_detail = fixture_factor(el["team"], [next_gw] if next_gw else [])
    long_f, long_detail = fixture_factor(el["team"], horizon)
    players[el["id"]] = {
        "id": el["id"],
        "name": el.get("web_name", "?"),
        "pos": POS.get(el["element_type"], "?"),
        "pos_id": el["element_type"],
        "team": team_name.get(el["team"], "?"),
        "team_id": el["team"],
        "owner": owner_of.get(el["id"]),
        "proj": proj,
        "conf": conf,
        "notes": notes,
        "short": proj * short_f,
        "long": proj * long_f,
        "short_fix": short_detail,
        "long_fix": long_detail,
        "form": el.get("form", "0.0"),
        "minutes": el.get("minutes") or 0,
        "starts": el.get("starts") or 0,
        "total_points": el.get("total_points") or 0,
        "status": el.get("status", "a"),
        "news": el.get("news") or "",
    }

roster = [p for p in players.values() if p["owner"] == my_id]
free = [p for p in players.values() if p["owner"] is None
        and p["status"] not in ("u",)]

# --------------------------------------------------------- swap suggestions
def build_swaps(key):
    out = []
    by_pos = defaultdict(list)
    for p in free:
        by_pos[p["pos_id"]].append(p)
    for pos_id, pool in by_pos.items():
        pool = sorted(pool, key=lambda p: -p[key])[:6]
        mine = sorted([p for p in roster if p["pos_id"] == pos_id],
                      key=lambda p: p[key])
        if not mine:
            continue
        for cand in pool:
            worst = mine[0]
            gain = cand[key] - worst[key]
            if gain >= SWAP_THRESHOLD:
                out.append({"add": cand, "drop": worst, "gain": gain})
    out.sort(key=lambda s: -s["gain"])
    # one suggestion per incoming player, max 6
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
    for gw, diffs in detail:
        if not diffs:
            bits.append('<i class="d0" title="Blank">&mdash;</i>')
            continue
        for d in diffs:
            bits.append(f'<i class="d{d}" title="GW{gw} difficulty {d}">{d}</i>')
    return f'<span class="fixes">{"".join(bits)}</span>'


def player_row(p, key, extra=""):
    conf_cls = "lowconf" if p["conf"] < 0.5 else ""
    note = f'<div class="note-inline">{p["notes"]}</div>' if p["notes"] else ""
    return (
        f'<div class="row {conf_cls}">'
        f'<div class="pos p{p["pos_id"]}">{p["pos"]}</div>'
        f'<div class="who"><div class="pname">{p["name"]}'
        f'<span class="club">{p["team"]}</span></div>'
        f'<div class="meta">form {p["form"]} &middot; {p["minutes"]}\' '
        f'&middot; {p["total_points"]}pts {extra}</div>{note}</div>'
        f'{fixture_chips(p["long_fix"])}'
        f'<div class="val"><div class="num">{p[key]:.1f}</div>'
        f'<div class="unit">pts/gw</div></div>'
        f'</div>'
    )


def swap_card(s, key, label):
    a, d = s["add"], s["drop"]
    return (
        f'<div class="swap">'
        f'<div class="swaphead"><span class="gain">+{s["gain"]:.1f}</span>'
        f'<span class="swaplabel">{label}</span></div>'
        f'<div class="leg add"><span class="tag">ADD</span>'
        f'<span class="pn">{a["name"]}</span>'
        f'<span class="cl">{a["pos"]} &middot; {a["team"]}</span>'
        f'<span class="pv">{a[key]:.1f}</span></div>'
        f'{fixture_chips(a["long_fix"])}'
        f'<div class="leg drop"><span class="tag">DROP</span>'
        f'<span class="pn">{d["name"]}</span>'
        f'<span class="cl">{d["pos"]} &middot; {d["team"]}</span>'
        f'<span class="pv">{d[key]:.1f}</span></div>'
        + (f'<div class="note-inline warn">{a["notes"]}</div>' if a["notes"] else "")
        + f'</div>'
    )


def swaps_pane(swaps, key, blurb):
    if not swaps:
        return (f'<section class="card"><div class="head"><h2>No moves suggested</h2>'
                f'<p>{blurb}</p></div>'
                f'<div class="empty">Nothing in the free agent pool clears '
                f'the {SWAP_THRESHOLD} pts/gw threshold right now.</div>'
                f'</section>')
    return (f'<section class="card"><div class="head"><h2>Suggested moves</h2>'
            f'<p>{blurb}</p></div>'
            f'{"".join(swap_card(s, key, "projected gain") for s in swaps)}'
            f'</section>')


roster_by_pos = defaultdict(list)
for p in roster:
    roster_by_pos[p["pos_id"]].append(p)

roster_html = ""
for pos_id in (1, 2, 3, 4):
    group = sorted(roster_by_pos.get(pos_id, []), key=lambda p: -p["long"])
    if not group:
        continue
    roster_html += (f'<section class="card"><div class="head">'
                    f'<h2>{POS[pos_id]}</h2><p>{len(group)} on roster</p></div>'
                    f'{"".join(player_row(p, "long") for p in group)}</section>')
if not roster_html:
    roster_html = ('<section class="card"><div class="head">'
                   '<h2>Roster not found</h2>'
                   '<p>Check the MY_TEAM setting</p></div></section>')

free_html = ""
for pos_id in (1, 2, 3, 4):
    group = sorted([p for p in free if p["pos_id"] == pos_id],
                   key=lambda p: -p["long"])[:10]
    if not group:
        continue
    free_html += (f'<section class="card"><div class="head">'
                  f'<h2>{POS[pos_id]}</h2><p>Top {len(group)} available</p></div>'
                  f'{"".join(player_row(p, "long") for p in group)}</section>')

updated = datetime.now(TZ).strftime("%a %d %b, %I:%M %p").replace(" 0", " ")
gw_label = f"GW{next_gw}" if next_gw else "season end"

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
  .reload {
    flex: none; width: 30px; height: 30px; border-radius: 9px;
    background: #141922; border: 1px solid #252c3a; color: #838ea3;
    font-size: 16px; line-height: 1; font-family: inherit;
  }
  .tabs {
    display: flex; gap: 4px; background: #141922; border: 1px solid #252c3a;
    border-radius: 12px; padding: 4px; margin-bottom: 15px;
    overflow-x: auto; scrollbar-width: none;
  }
  .tabs::-webkit-scrollbar { display: none; }
  .tab {
    flex: 1 0 auto; padding: 9px 12px; border-radius: 9px; white-space: nowrap;
    font-size: 12.5px; font-weight: 600; color: #838ea3;
    border: none; background: none; font-family: inherit;
  }
  .tab.on { background: #2f7fe0; color: #fff; }
  .pane { display: none; }
  .pane.on { display: block; }
  .card {
    background: #141922; border: 1px solid #252c3a;
    border-radius: 15px; overflow: hidden; margin-bottom: 14px;
  }
  .head { padding: 14px 16px 11px; border-bottom: 1px solid #252c3a; }
  .head h2 { margin: 0; font-size: 17px; font-weight: 700; }
  .head p { margin: 3px 0 0; font-size: 12px; color: #838ea3; }
  .row {
    display: flex; align-items: center; gap: 10px;
    padding: 11px 14px; border-bottom: 1px solid #1c2029;
  }
  .row:last-of-type { border-bottom: none; }
  .row.lowconf { opacity: .72; }
  .pos {
    flex: none; width: 32px; text-align: center; font-size: 10px;
    font-weight: 800; padding: 3px 0; border-radius: 6px; letter-spacing: .04em;
  }
  .p1 { background: #3b2f00; color: #f0c14b; }
  .p2 { background: #10304a; color: #6cc0ff; }
  .p3 { background: #113626; color: #4fd490; }
  .p4 { background: #3c1420; color: #ff7a92; }
  .who { flex: 1; min-width: 0; }
  .pname { font-size: 14px; font-weight: 600; }
  .club { font-size: 10.5px; color: #838ea3; margin-left: 6px; font-weight: 500; }
  .meta { font-size: 10.5px; color: #838ea3; margin-top: 2px; }
  .note-inline { font-size: 10.5px; color: #f0b429; margin-top: 2px; }
  .note-inline.warn { padding: 0 14px 11px; }
  .fixes { display: flex; gap: 2px; flex: none; }
  .fixes i {
    display: block; width: 15px; height: 15px; border-radius: 3px;
    font-size: 9px; font-weight: 700; line-height: 15px; text-align: center;
    font-style: normal;
  }
  .d1 { background: #1b5e35; color: #9ff5c0; }
  .d2 { background: #23603a; color: #a8f0c2; }
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
  .gain {
    background: #1e5f3a; color: #8bf0b5; font-size: 12px; font-weight: 800;
    padding: 3px 9px; border-radius: 20px;
  }
  .swaplabel { font-size: 10.5px; color: #838ea3; text-transform: uppercase;
    letter-spacing: .07em; }
  .leg { display: flex; align-items: center; gap: 8px; padding: 5px 0; }
  .tag {
    font-size: 9px; font-weight: 800; padding: 3px 6px; border-radius: 5px;
    letter-spacing: .05em; flex: none; width: 42px; text-align: center;
  }
  .add .tag { background: #14432c; color: #6ee7a6; }
  .drop .tag { background: #45202a; color: #ff9aad; }
  .pn { font-size: 14px; font-weight: 600; flex: 1; min-width: 0; }
  .cl { font-size: 10.5px; color: #838ea3; flex: none; }
  .pv { font-size: 14px; font-weight: 700; flex: none; min-width: 34px;
    text-align: right; font-variant-numeric: tabular-nums; }
  .empty { padding: 30px 18px; text-align: center; font-size: 13px; color: #838ea3; }
  .note { padding: 11px 16px; font-size: 11.5px; color: #838ea3; line-height: 1.45;
    border-top: 1px solid #1c2029; }
  .note b { color: #f2f5fa; }
  footer { text-align: center; font-size: 11px; color: #4a5468; margin-top: 24px; }
"""

SCRIPT = """
<script>
(function () {
  var tabs = document.querySelectorAll('.tab');
  var panes = document.querySelectorAll('.pane');
  var order = ['moves', 'roster', 'free'];
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

horizon_label = (f"GW{horizon[0]}&ndash;{horizon[-1]}"
                 if len(horizon) > 1 else gw_label)

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
  </div>

  <div class="pane on" data-pane="moves">
    {swaps_pane(swaps_short, "short", f"Best single-week move for {gw_label}")}
    {swaps_pane(swaps_long, "long", f"Best move across {horizon_label}")}
    <section class="card"><div class="head"><h2>How this is worked out</h2>
      <p>So you can argue with it</p></div>
      <div class="note">
      Each player gets a projected points-per-gameweek from three parts:
      <b>{int(W_HISTORY*100)}%</b> season scoring rate scaled by minutes,
      <b>{int(W_FORM*100)}%</b> recent form, and
      <b>{int(W_UNDERLYING*100)}%</b> underlying numbers &mdash; expected goals
      and assists, clean sheet odds from expected goals conceded, save rate for
      keepers, and defensive contributions. That's multiplied by injury
      availability, then by fixture difficulty over the window.
      Faded rows and "small sample" flags mean there isn't enough football
      played yet to trust the number. Only same-position swaps are suggested,
      since draft squads must stay legal at 2/5/5/3.
      </div></section>
  </div>

  <div class="pane" data-pane="roster">{roster_html}</div>
  <div class="pane" data-pane="free">{free_html}</div>

  <footer>Fixture colours: green easy, red hard &middot; rebuilt daily</footer>
</div>
{SCRIPT}
</body>
</html>
"""

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html)

print(f"Roster: {len(roster)} players | free agents considered: {len(free)}")
print(f"Suggestions: {len(swaps_short)} short, {len(swaps_long)} long")
print(f"Built index.html for {league_name}, next {gw_label}")
