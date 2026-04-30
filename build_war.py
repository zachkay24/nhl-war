#!/usr/bin/env python3
"""
NHL WAR Model - Player Card Generator
======================================
Reads CSV files (skaters.csv, goalies.csv) and generates
an interactive HTML player card lookup tool.

To update: replace the CSV files with new data and re-run this script.

WAR Components:
  - EV Offense WAR  : Individual offensive production at 5-on-5 vs replacement
  - EV Defense WAR  : On-ice xGoals-against suppression at 5-on-5 vs league avg
  - PP WAR          : Power play offensive contributions vs average PP player
  - PK WAR          : Penalty kill defensive impact vs league average PK xGA rate
  - Finishing       : Goals above xGoals (shooting talent signal, not in WAR total)
  - Total WAR       : Sum of EV Off + EV Def + PP + PK components

Scaling: 1 WAR = 6 Goals Above Replacement (standard hockey analytics)
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime

# ─── Configuration ────────────────────────────────────────────────────────────
GOALS_PER_WIN       = 6.0
MIN_EV_ICETIME_SEC  = 300
MIN_PP_ICETIME_SEC  = 60
MIN_PK_ICETIME_SEC  = 60
REPL_LEVEL_PCT      = 0.15

SIT_EV  = "5on5"
SIT_PP  = "5on4"
SIT_PK  = "4on5"
SIT_ALL = "all"

# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_data(folder):
    sk = pd.read_csv(os.path.join(folder, "skaters.csv"))
    go = pd.read_csv(os.path.join(folder, "goalies.csv"))
    return sk, go


def load_old_season(old_folder, year):
    """Load one season's worth of rows from the combined historical CSVs."""
    year = int(year)
    sk = pd.read_csv(os.path.join(old_folder, "skaters_2008_to_2024.csv"))
    go = pd.read_csv(os.path.join(old_folder, "goalies_2008_to_2024.csv"))
    return sk[sk["season"] == year].copy(), go[go["season"] == year].copy()


def load_old_lines(old_folder, year):
    year = int(year)
    li = pd.read_csv(os.path.join(old_folder, "lines_2008_to_2024.csv"))
    return li[li["season"] == year].copy()


def load_old_teams(old_folder, year):
    year = int(year)
    tm = pd.read_csv(os.path.join(old_folder, "teams_2008_to_2024.csv"))
    return tm[tm["season"] == year].copy()


# ─── Season / Nav Helpers ──────────────────────────────────────────────────────

def season_label(year):
    """2025 -> '2025-26'"""
    y = int(year)
    return f"{y}-{str(y + 1)[-2:]}"


def page_filename(page_type, year, latest_year):
    """Return the HTML filename for a given page type and season year."""
    suffix = "" if int(year) == int(latest_year) else f"_{season_label(year)}"
    return {
        "players": f"NHL_WAR_Cards{suffix}.html",
        "teams":   f"NHL_WAR_Teams{suffix}.html",
        "lines":   f"NHL_WAR_Lines{suffix}.html",
    }[page_type]


def make_nav(page_type, current_year, all_years, is_in_subdir=False):
    """Generate site nav HTML with page links and season dropdown.

    is_in_subdir: True when the file being generated lives in 'Old Seasons/'
                  so that links to current-season files use '../' prefix and
                  the About link resolves correctly.
    """
    latest = max(int(y) for y in all_years)
    cur = int(current_year)

    def href_for(pt, y):
        """Relative href from this file's location to the target page."""
        fname = page_filename(pt, y, latest)
        if int(y) == latest:
            # target is a current-season file sitting in the root folder
            return f"../{fname}" if is_in_subdir else fname
        else:
            # target is an old-season file sitting in Old Seasons/
            return fname if is_in_subdir else f"Old Seasons/{fname}"

    about_href = "../NHL_WAR_About.html" if is_in_subdir else "NHL_WAR_About.html"

    pages = [("players", "Players"), ("teams", "Teams"), ("lines", "Lines")]
    link_parts = []
    for pt, lbl in pages:
        active_attr = ' class="active"' if pt == page_type else ""
        href = href_for(pt, cur)
        link_parts.append(f'<a href="{href}"{active_attr}>{lbl}</a>')
    link_parts.append(f'<a href="{about_href}">About</a>')
    links = "\n  ".join(link_parts)

    opts = "\n    ".join(
        f'<option value="{href_for(page_type, y)}"'
        f'{" selected" if int(y) == cur else ""}>{season_label(y)}</option>'
        for y in sorted(all_years, reverse=True)
    )
    sel = (
        f'<select onchange="location.href=this.value" style="'
        f'padding:6px 14px;border-radius:20px;border:1px solid var(--border);'
        f'background:var(--surf2);color:var(--text);font-size:0.82rem;'
        f'cursor:pointer;outline:none;font-weight:600;">\n    {opts}\n  </select>'
    )
    return f'<nav class="site-nav">\n  {links}\n  {sel}\n</nav>'


# ─── Helpers ──────────────────────────────────────────────────────────────────

def safe_div(num, den, default=0.0):
    return np.where(den > 0, num / den, default)

def pos_group(position):
    return "F" if position in ["L", "R", "C"] else "D"

def percentile_rank(value, series, higher_better=True):
    arr = np.array(series.dropna())
    if len(arr) == 0:
        return 50.0
    return float(np.mean(arr < value) * 100) if higher_better else float(np.mean(arr > value) * 100)


# ─── Skater WAR ───────────────────────────────────────────────────────────────

def compute_skater_war(sk, season):
    ev      = sk[sk["situation"] == SIT_EV].copy()
    pp      = sk[sk["situation"] == SIT_PP].copy()
    pk      = sk[sk["situation"] == SIT_PK].copy()
    all_sit = sk[sk["situation"] == SIT_ALL].copy()

    for df in [ev, pp, pk, all_sit]:
        df["ice_hr"]   = df["icetime"] / 3600
        df["pos_group"] = df["position"].apply(pos_group)

    # EV Offense
    ev["iOff_total"] = (ev["I_F_xGoals"] + 0.75*ev["I_F_primaryAssists"]
                        + 0.5*ev["I_F_secondaryAssists"])
    ev["iOff60"] = safe_div(ev["iOff_total"], ev["ice_hr"])
    ev["repl_off60"] = np.nan
    for pg in ["F", "D"]:
        mask_q = (ev["pos_group"] == pg) & (ev["icetime"] >= MIN_EV_ICETIME_SEC)
        if mask_q.sum() > 5:
            ev.loc[ev["pos_group"] == pg, "repl_off60"] = np.percentile(
                ev.loc[mask_q, "iOff60"], REPL_LEVEL_PCT * 100)
    ev["repl_off60"] = ev["repl_off60"].fillna(ev["iOff60"].quantile(REPL_LEVEL_PCT))
    ev["EV_Off_GAR"] = (ev["iOff60"] - ev["repl_off60"]) * ev["ice_hr"]
    ev["EV_Off_WAR"] = ev["EV_Off_GAR"] / GOALS_PER_WIN
    ev["EV_xGF60"]   = safe_div(ev["OnIce_F_xGoals"], ev["ice_hr"])
    ev["EV_HDxG60"]  = safe_div(ev["I_F_highDangerxGoals"], ev["ice_hr"])

    # EV Defense
    total_on  = ev["OnIce_F_xGoals"]  + ev["OnIce_A_xGoals"]
    total_off = ev["OffIce_F_xGoals"] + ev["OffIce_A_xGoals"]
    ev["onIce_xGA_share"]  = safe_div(ev["OnIce_A_xGoals"],  total_on,  0.5)
    ev["offIce_xGA_share"] = safe_div(ev["OffIce_A_xGoals"], total_off, 0.5)
    ev["def_relative"]     = ev["offIce_xGA_share"] - ev["onIce_xGA_share"]
    ev["EV_xGA60"]         = safe_div(ev["OnIce_A_xGoals"], ev["ice_hr"])
    ev["EV_Def_GAR"] = ev["def_relative"] * safe_div(total_on, ev["ice_hr"]) * ev["ice_hr"]
    ev["EV_Def_WAR"] = ev["EV_Def_GAR"] / GOALS_PER_WIN

    # PP
    pp["pp_iScore"]  = (pp["I_F_xGoals"] + 0.75*pp["I_F_primaryAssists"]
                        + 0.5*pp["I_F_secondaryAssists"])
    pp["pp_iOff60"]  = safe_div(pp["pp_iScore"], pp["ice_hr"])
    pp_q = pp[pp["icetime"] >= MIN_PP_ICETIME_SEC]
    repl_pp60 = np.average(pp_q["pp_iOff60"], weights=pp_q["icetime"]) * 0.40 if len(pp_q) > 5 else 0.0
    pp["PP_GAR"] = (pp["pp_iOff60"] - repl_pp60) * pp["ice_hr"]
    pp["PP_WAR"] = pp["PP_GAR"] / GOALS_PER_WIN

    # PK
    pk["pk_xGA60"] = safe_div(pk["OnIce_A_xGoals"], pk["ice_hr"])
    pk["pk_xGF60"] = safe_div(pk["OnIce_F_xGoals"], pk["ice_hr"])
    pk_q = pk[pk["icetime"] >= MIN_PK_ICETIME_SEC]
    avg_pk_xGA60 = np.average(pk_q["pk_xGA60"], weights=pk_q["icetime"]) if len(pk_q) > 5 else 6.0
    pk["PK_GAR"] = (avg_pk_xGA60 - pk["pk_xGA60"]) * pk["ice_hr"]
    pk["PK_WAR"] = pk["PK_GAR"] / GOALS_PER_WIN

    # All-sit
    all_sit["Finishing"] = all_sit["I_F_goals"] - all_sit["I_F_xGoals"]
    all_sit["TOI_min"]   = all_sit["icetime"] / 60

    # Merge
    results = {}

    def merge(pid, base, upd):
        if pid not in results:
            results[pid] = base
        results[pid].update({k: v for k, v in upd.items() if v is not None})

    def sv(val, nd=2, i=False):
        if pd.isna(val): return 0
        return int(val) if i else round(float(val), nd)

    for _, r in ev.iterrows():
        pid = r["playerId"]
        merge(pid,
            dict(playerId=pid, name=r["name"], team=r["team"],
                 position=r["position"], pos_group=r["pos_group"],
                 season=season, games_played=sv(r.get("games_played",0),i=True)),
            dict(EV_icetime_min=sv(r["icetime"]/60,1), EV_Off_GAR=sv(r["EV_Off_GAR"]),
                 EV_Off_WAR=sv(r["EV_Off_WAR"]), EV_Def_GAR=sv(r["EV_Def_GAR"]),
                 EV_Def_WAR=sv(r["EV_Def_WAR"]),
                 xGF_pct=sv(r["onIce_xGoalsPercentage"]*100,1),
                 Corsi_pct=sv(r["onIce_corsiPercentage"]*100,1),
                 iOff60=sv(r["iOff60"]), EV_xGF60=sv(r["EV_xGF60"]),
                 EV_xGA60=sv(r["EV_xGA60"]), EV_HDxG60=sv(r["EV_HDxG60"])))

    for _, r in pp.iterrows():
        pid = r["playerId"]
        merge(pid,
            dict(playerId=pid, name=r["name"], team=r["team"],
                 position=r["position"], pos_group=r["pos_group"],
                 season=season, games_played=sv(r.get("games_played",0),i=True)),
            dict(PP_icetime_min=sv(r["icetime"]/60,1), PP_GAR=sv(r["PP_GAR"]),
                 PP_WAR=sv(r["PP_WAR"]), PP_xG=sv(r["I_F_xGoals"]),
                 PP_goals=sv(r.get("I_F_goals",0),i=True),
                 PP_points=sv(r.get("I_F_points",0),i=True),
                 PP_primaryAssists=sv(r.get("I_F_primaryAssists",0),i=True),
                 PP_secondaryAssists=sv(r.get("I_F_secondaryAssists",0),i=True),
                 PP_per60=sv(r["pp_iOff60"])))

    for _, r in pk.iterrows():
        pid = r["playerId"]
        merge(pid,
            dict(playerId=pid, name=r["name"], team=r["team"],
                 position=r["position"], pos_group=r["pos_group"],
                 season=season, games_played=sv(r.get("games_played",0),i=True)),
            dict(PK_icetime_min=sv(r["icetime"]/60,1), PK_GAR=sv(r["PK_GAR"]),
                 PK_WAR=sv(r["PK_WAR"]), PK_xGA60=sv(r["pk_xGA60"]), PK_xGF60=sv(r["pk_xGF60"]),
                 PK_points=sv(r.get("I_F_points",0),i=True),
                 PK_primaryAssists=sv(r.get("I_F_primaryAssists",0),i=True)))

    for _, r in all_sit.iterrows():
        pid = r["playerId"]
        merge(pid,
            dict(playerId=pid, name=r["name"], team=r["team"],
                 position=r["position"], pos_group=pos_group(r["position"]),
                 season=season, games_played=sv(r.get("games_played",0),i=True)),
            dict(total_icetime_min=sv(r["TOI_min"],1),
                 goals=sv(r.get("I_F_goals",0),i=True),
                 primaryAssists=sv(r.get("I_F_primaryAssists",0),i=True),
                 secondaryAssists=sv(r.get("I_F_secondaryAssists",0),i=True),
                 points=sv(r.get("I_F_points",0),i=True), xGoals=sv(r.get("I_F_xGoals",0)),
                 Finishing=sv(r["Finishing"]),
                 faceoffsWon=sv(r.get("faceoffsWon",0),i=True),
                 faceoffsLost=sv(r.get("faceoffsLost",0),i=True),
                 faceoff_pct=sv(
                     float(r.get("faceoffsWon",0)) /
                     max(float(r.get("faceoffsWon",0))+float(r.get("faceoffsLost",0)),1)*100, 1),
                 hits=sv(r.get("I_F_hits",0),i=True),
                 takeaways=sv(r.get("I_F_takeaways",0),i=True),
                 giveaways=sv(r.get("I_F_giveaways",0),i=True),
                 shots=sv(r.get("I_F_shotsOnGoal",0),i=True),
                 HDxG=sv(r.get("I_F_highDangerxGoals",0)), gameScore=sv(r.get("gameScore",0)),
                 penaltiesDrawn=sv(r.get("penaltiesDrawn",0),i=True),
                 penalties=sv(r.get("penalties",0),i=True)))

    players = list(results.values())
    for p in players:
        p["Total_WAR"] = round(p.get("EV_Off_WAR",0)+p.get("EV_Def_WAR",0)
                               +p.get("PP_WAR",0)+p.get("PK_WAR",0), 2)
        p["Total_GAR"] = round(p["Total_WAR"] * GOALS_PER_WIN, 2)
        gp = p.get("games_played", 0) or 1
        p["WAR_per82"] = round(p["Total_WAR"] / gp * 82, 2)
        p["EV_Off_WAR_per82"] = round(p.get("EV_Off_WAR",0) / gp * 82, 2)
        p["EV_Def_WAR_per82"] = round(p.get("EV_Def_WAR",0) / gp * 82, 2)
        p["PP_WAR_per82"]     = round(p.get("PP_WAR",0)     / gp * 82, 2)
        p["PK_WAR_per82"]     = round(p.get("PK_WAR",0)     / gp * 82, 2)
        for key in ["EV_Off_WAR","EV_Def_WAR","PP_WAR","PK_WAR","EV_Off_GAR",
                    "EV_Def_GAR","PP_GAR","PK_GAR","EV_icetime_min","PP_icetime_min",
                    "PK_icetime_min","Finishing","xGF_pct","Corsi_pct"]:
            p.setdefault(key, 0)
    return players


# ─── Goalie WAR ───────────────────────────────────────────────────────────────

def compute_goalie_war(go, season):
    g    = go[go["situation"] == SIT_ALL].copy()
    ev_g = go[go["situation"] == SIT_EV].copy()
    g["ice_hr"]     = g["icetime"] / 3600
    g["GSAE"]       = g["xGoals"] - g["goals"]
    g["SavePct"]    = safe_div(g["ongoal"] - g["goals"], g["ongoal"])
    g["xSavePct"]   = safe_div(g["ongoal"] - g["xGoals"], g["ongoal"])
    g["GSAEper60"]  = safe_div(g["GSAE"], g["ice_hr"])
    g["ShotFaced60"]= safe_div(g["ongoal"], g["ice_hr"])
    g["HD_GSAE"]    = g["highDangerxGoals"] - g["highDangerGoals"]
    g["Goalie_WAR"] = g["GSAE"] / GOALS_PER_WIN

    results = []
    for _, r in g.iterrows():
        pid = r["playerId"]
        ev_row = ev_g[ev_g["playerId"] == pid]
        ev_gsae = float((ev_row["xGoals"] - ev_row["goals"]).values[0]) if len(ev_row) else 0
        gp   = int(r.get("games_played", 0)) or 1
        gwar = round(float(r["Goalie_WAR"]), 2)
        results.append(dict(
            playerId=pid, name=r["name"], team=r["team"],
            position="G", pos_group="G",
            season=season, games_played=int(r.get("games_played",0)),
            total_icetime_min=round(r["icetime"]/60,1),
            GSAE=round(float(r["GSAE"]),2), GSAE_per60=round(float(r["GSAEper60"]),2),
            EV_GSAE=round(ev_gsae,2), SavePct=round(float(r["SavePct"]),4),
            xSavePct=round(float(r["xSavePct"]),4), HD_GSAE=round(float(r["HD_GSAE"]),2),
            shots_faced=int(r["ongoal"]), goals_allowed=int(r["goals"]),
            xGoals_against=round(float(r["xGoals"]),2),
            ShotFaced60=round(float(r["ShotFaced60"]),2),
            Goalie_WAR=gwar, Total_WAR=gwar,
            WAR_per82=round(gwar / gp * 82, 2),
        ))
    return results


# ─── Percentile Computation ───────────────────────────────────────────────────

def add_percentiles(players, goalies):
    def make_series(lst, key):
        return pd.Series([p.get(key, 0) for p in lst])

    skater_keys = [
        ("Total_WAR",    True), ("WAR_per82",    True),
        ("EV_Off_WAR",   True), ("EV_Def_WAR",   True),
        ("PP_WAR",       True), ("PK_WAR",        True),
        ("Finishing",    True), ("xGF_pct",       True),
        ("Corsi_pct",    True), ("iOff60",        True),
        ("EV_xGA60",     False),
    ]
    fwds = [p for p in players if p.get("pos_group") == "F"]
    dmen = [p for p in players if p.get("pos_group") == "D"]

    for key, hb in skater_keys:
        all_v = make_series(players, key)
        fv    = make_series(fwds, key)
        dv    = make_series(dmen, key)
        for p in players:
            val = p.get(key, 0)
            pv  = fv if p.get("pos_group") == "F" else dv
            p[f"{key}_pct_all"] = round(percentile_rank(val, all_v, hb), 1)
            p[f"{key}_pct_pos"] = round(percentile_rank(val, pv, hb), 1)

    goalie_keys = [("Goalie_WAR",True),("WAR_per82",True),("GSAE",True),
                   ("GSAE_per60",True),("SavePct",True),("HD_GSAE",True)]
    for key, hb in goalie_keys:
        vals = make_series(goalies, key)
        for g in goalies:
            g[f"{key}_pct"] = round(percentile_rank(g.get(key, 0), vals, hb), 1)

    return players, goalies


# ─── HTML Template ────────────────────────────────────────────────────────────

def build_html(all_records, season, timestamp, nav_html="", player_history_json="{}", player_index_json="{}", in_subdir=False):
    data_json  = json.dumps(all_records, ensure_ascii=False)
    max_gp     = max((p.get("games_played", 0) for p in all_records), default=82)
    player_hist_json = player_history_json  # already a JSON string
    in_subdir_js = "true" if in_subdir else "false"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NHL WAR Player Cards</title>
<style>
  :root {{
    --bg:      #0a0e1a;
    --surf:    #111827;
    --surf2:   #1e2a3a;
    --border:  #2a3a50;
    --accent:  #00b4d8;
    --accent2: #0077b6;
    --gold:    #f4a261;
    --red:     #e63946;
    --green:   #2dc653;
    --muted:   #64748b;
    --text:    #e2e8f0;
    --text2:   #94a3b8;
    --r:       12px;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg); color: var(--text);
    font-family: 'Segoe UI', system-ui, sans-serif;
    min-height: 100vh; padding: 24px 16px;
  }}

  /* ── Header ── */
  /* ── Nav ── */
  .site-nav {{
    display: flex; justify-content: center; gap: 6px;
    margin: 0 auto 28px; max-width: 480px;
  }}
  .site-nav a {{
    padding: 8px 22px; border-radius: 50px;
    border: 1px solid var(--border); background: var(--surf);
    color: var(--text2); font-size: 0.85rem; font-weight: 600;
    text-decoration: none; transition: all 0.15s;
  }}
  .site-nav a:hover {{ border-color: var(--accent); color: var(--accent); }}
  .site-nav a.active {{ background: var(--accent2); border-color: var(--accent); color: #fff; }}

  .header {{ text-align: center; margin-bottom: 28px; }}
  .header h1 {{
    font-size: 2.2rem; font-weight: 800; letter-spacing: -0.5px;
    background: linear-gradient(135deg, var(--accent), var(--gold));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
  }}
  .header p {{ color: var(--text2); margin-top: 6px; font-size: 0.9rem; }}

  /* ── Search ── */
  .search-wrap {{
    max-width: 680px; margin: 0 auto 20px; position: relative;
  }}
  #search {{
    width: 100%; padding: 13px 20px; border-radius: 50px;
    border: 2px solid var(--border); background: var(--surf);
    color: var(--text); font-size: 1.05rem; outline: none;
    transition: border-color 0.2s;
  }}
  #search:focus {{ border-color: var(--accent); }}
  #search::placeholder {{ color: var(--muted); }}
  .dropdown {{
    position: absolute; top: calc(100% + 6px); left: 0; right: 0;
    background: var(--surf); border: 1px solid var(--border);
    border-radius: var(--r); max-height: 240px; overflow-y: auto;
    z-index: 200; display: none;
  }}
  .dropdown.show {{ display: block; }}
  .dd-item {{
    padding: 9px 18px; cursor: pointer;
    display: flex; justify-content: space-between; align-items: center;
    font-size: 0.93rem; transition: background 0.12s;
  }}
  .dd-item:hover {{ background: var(--surf2); }}
  .dd-item .pos-b {{
    font-size: 0.7rem; font-weight: 700; padding: 2px 6px;
    border-radius: 4px; background: var(--surf2); color: var(--accent);
    white-space: nowrap;
  }}
  .dd-item.dd-hist {{ opacity: 0.75; }}
  .dd-item.dd-hist .pos-b {{ color: var(--text2); }}
  .dd-sep {{
    padding: 4px 18px; font-size: 0.68rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 1px; color: var(--muted);
    background: var(--surf2); border-top: 1px solid var(--border);
  }}
  .dd-item .team-t {{ color: var(--text2); font-size: 0.82rem; margin-left: 6px; }}

  /* ── Controls / Filters ── */
  .controls {{
    max-width: 1200px; margin: 0 auto 16px;
    display: flex; gap: 8px; flex-wrap: wrap; align-items: center;
  }}
  .ctrl-row {{
    display: flex; gap: 8px; flex-wrap: wrap; align-items: center; width: 100%;
  }}
  .filter-tabs {{ display: flex; gap: 4px; flex-wrap: wrap; }}
  .tab {{
    padding: 6px 14px; border-radius: 20px; border: 1px solid var(--border);
    background: var(--surf); color: var(--text2); font-size: 0.82rem;
    font-weight: 600; cursor: pointer; transition: all 0.15s; white-space: nowrap;
  }}
  .tab:hover {{ border-color: var(--accent); color: var(--accent); }}
  .tab.active {{ background: var(--accent2); border-color: var(--accent); color: #fff; }}
  .ctrl-select {{
    padding: 6px 12px; border-radius: 20px;
    border: 1px solid var(--border); background: var(--surf);
    color: var(--text); font-size: 0.82rem; cursor: pointer; outline: none;
  }}
  .ctrl-select:focus {{ border-color: var(--accent); }}
  .ctrl-label {{
    font-size: 0.78rem; color: var(--text2); white-space: nowrap;
    display: flex; align-items: center; gap: 6px;
  }}
  .ctrl-input {{
    width: 64px; padding: 5px 8px; border-radius: 8px;
    border: 1px solid var(--border); background: var(--surf);
    color: var(--text); font-size: 0.82rem; outline: none; text-align: center;
  }}
  .ctrl-input:focus {{ border-color: var(--accent); }}
  .ctrl-divider {{
    width: 1px; height: 24px; background: var(--border); flex-shrink: 0;
  }}
  .controls-right {{ margin-left: auto; color: var(--muted); font-size: 0.8rem; }}

  /* ── Leaderboard Table ── */
  #leaderboard {{ margin: 0 auto; }}
  .lb-wrap {{
    background: var(--surf); border: 1px solid var(--border);
    border-radius: var(--r); overflow-x: auto;
  }}
  .lb-table {{
    border-collapse: collapse; font-size: 0.84rem; white-space: nowrap;
  }}
  .lb-table thead tr {{
    background: var(--surf2); border-bottom: 2px solid var(--border);
  }}
  .lb-table th {{
    padding: 9px 10px; text-align: left; font-size: 0.68rem;
    font-weight: 700; letter-spacing: 0.8px; text-transform: uppercase;
    color: var(--text2); white-space: nowrap; cursor: pointer;
    user-select: none; transition: color 0.15s;
  }}
  .lb-table th:hover {{ color: var(--accent); }}
  .lb-table th.sort-asc::after  {{ content: ' ↑'; color: var(--accent); }}
  .lb-table th.sort-desc::after {{ content: ' ↓'; color: var(--accent); }}
  .lb-table th.num {{ text-align: right; }}
  .lb-table td {{
    padding: 8px 10px; border-bottom: 1px solid rgba(42,58,80,0.5);
  }}
  .lb-table td.num {{ text-align: right; }}
  .lb-table tbody tr {{ cursor: pointer; transition: background 0.12s; }}
  .lb-table tbody tr:hover {{ background: rgba(0,180,216,0.07); }}
  .lb-table tbody tr:last-child td {{ border-bottom: none; }}

  /* Sticky identity columns */
  .sc1,.sc2,.sc3,.sc4 {{ position: sticky; z-index: 2; background: var(--surf); }}
  .lb-table thead .sc1,
  .lb-table thead .sc2,
  .lb-table thead .sc3,
  .lb-table thead .sc4 {{ background: var(--surf2); z-index: 3; }}
  .lb-table tbody tr:hover .sc1,
  .lb-table tbody tr:hover .sc2,
  .lb-table tbody tr:hover .sc3,
  .lb-table tbody tr:hover .sc4 {{ background: #0e1d30; }}
  .sc1 {{ left: 0;    min-width: 36px;  max-width: 36px; }}
  .sc2 {{ left: 36px; min-width: 150px; max-width: 150px; }}
  .sc3 {{ left: 186px;min-width: 50px;  max-width: 50px; }}
  .sc4 {{ left: 236px;min-width: 50px;  max-width: 50px;
          box-shadow: 3px 0 8px rgba(0,0,0,0.4); }}

  /* Column group separators */
  .grp-l {{ border-left: 2px solid var(--border) !important; }}

  .rank-cell {{ color: var(--muted); font-size: 0.78rem; }}
  .name-cell {{ font-weight: 600; max-width: 150px; overflow: hidden; text-overflow: ellipsis; }}
  .name-cell .team-sub {{ color: var(--text2); font-size: 0.75rem; font-weight: 400; }}
  .pos-pill {{
    display: inline-block; padding: 2px 6px; border-radius: 4px;
    font-size: 0.68rem; font-weight: 700; background: var(--surf2);
    color: var(--accent); border: 1px solid var(--border);
  }}
  .pos-pill.G {{ color: #9b59b6; }}

  .war-cell {{ font-weight: 700; font-size: 0.9rem; }}
  .c-gold {{ color: var(--gold); }}
  .c-green {{ color: var(--green); }}
  .c-red {{ color: var(--red); }}
  .c-muted {{ color: var(--text2); }}

  .lb-footer {{
    padding: 10px 16px; background: var(--surf2); border-top: 1px solid var(--border);
    font-size: 0.75rem; color: var(--muted); text-align: right;
  }}

  /* ── Card Area ── */
  #card-area {{ max-width: 920px; margin: 0 auto; display: none; }}
  .back-btn {{
    display: inline-flex; align-items: center; gap: 6px;
    margin-bottom: 16px; padding: 8px 16px;
    background: var(--surf); border: 1px solid var(--border);
    border-radius: 20px; color: var(--text2); font-size: 0.85rem;
    cursor: pointer; transition: all 0.15s;
  }}
  .back-btn:hover {{ border-color: var(--accent); color: var(--accent); }}

  /* ── Player Card ── */
  .player-card {{
    background: var(--surf); border: 1px solid var(--border);
    border-radius: var(--r); overflow: hidden;
  }}
  .card-header {{
    background: linear-gradient(135deg, var(--surf2), #0d1b2a);
    padding: 22px 26px; display: flex; align-items: center;
    gap: 18px; border-bottom: 1px solid var(--border);
  }}
  .avatar {{
    width: 68px; height: 68px; border-radius: 50%;
    background: linear-gradient(135deg, var(--accent2), var(--accent));
    display: flex; align-items: center; justify-content: center;
    font-size: 1.7rem; font-weight: 900; color: #fff; flex-shrink: 0;
  }}
  .avatar.goalie {{ background: linear-gradient(135deg, #6a0dad, #9b59b6); }}
  .player-info h2 {{ font-size: 1.55rem; font-weight: 800; line-height: 1.1; }}
  .player-meta {{ display: flex; gap: 8px; margin-top: 6px; flex-wrap: wrap; align-items: center; }}
  .meta-b {{
    font-size: 0.77rem; font-weight: 600; padding: 3px 10px;
    border-radius: 20px; background: var(--surf); border: 1px solid var(--border); color: var(--text2);
  }}
  .meta-b.pos {{ background: var(--accent2); border-color: var(--accent); color: #fff; }}
  .meta-b.pos.G {{ background: #6a0dad; border-color: #9b59b6; }}

  /* WAR summary */
  .war-summary {{
    padding: 18px 26px; display: flex; gap: 12px; flex-wrap: wrap;
    border-bottom: 1px solid var(--border);
    background: linear-gradient(to right, rgba(0,180,216,0.05), transparent);
  }}
  .war-big {{
    flex: 1; min-width: 130px; text-align: center;
    padding: 12px 14px; background: var(--surf2);
    border-radius: 10px; border: 1px solid var(--border);
  }}
  .war-big.total {{ border-color: var(--accent); background: rgba(0,180,216,0.08); }}
  .war-big .lbl {{
    font-size: 0.68rem; font-weight: 700; letter-spacing: 1px;
    color: var(--text2); text-transform: uppercase; margin-bottom: 4px;
  }}
  .war-big .val {{ font-size: 1.9rem; font-weight: 900; line-height: 1; }}
  .war-big .pct-lbl {{ font-size: 0.74rem; color: var(--text2); margin-top: 4px; }}

  /* Components */
  .card-body {{ padding: 22px 26px; display: flex; flex-direction: column; gap: 22px; }}
  .section-title {{
    font-size: 0.72rem; font-weight: 700; letter-spacing: 1.5px;
    text-transform: uppercase; color: var(--muted); margin-bottom: 12px;
    display: flex; align-items: center; gap: 8px;
  }}
  .section-title::after {{ content: ''; flex: 1; height: 1px; background: var(--border); }}
  .war-comp {{ display: flex; flex-direction: column; gap: 9px; }}
  .comp-row {{
    display: grid; grid-template-columns: 128px 1fr 110px;
    align-items: center; gap: 10px;
  }}
  .comp-lbl {{ font-size: 0.86rem; font-weight: 600; color: var(--text2); white-space: nowrap; }}
  .bar-track {{
    height: 9px; background: var(--surf2); border-radius: 4px;
    overflow: hidden; position: relative;
  }}
  .bar-track::before {{
    content: ''; position: absolute; left: 50%; top: 0;
    width: 2px; height: 100%; background: var(--border); z-index: 1;
  }}
  .bar-fill {{
    height: 100%; border-radius: 4px; transition: width 0.5s ease;
    position: relative; z-index: 2;
  }}
  .comp-vals {{ text-align: right; }}
  .comp-vals .cv {{ font-weight: 700; font-size: 0.9rem; }}
  .comp-vals .cp {{ color: var(--text2); font-size: 0.73rem; margin-top: 1px; }}

  .stats-grid {{
    display: grid; grid-template-columns: repeat(auto-fill, minmax(115px, 1fr)); gap: 8px;
  }}
  .stat-box {{
    background: var(--surf2); border: 1px solid var(--border);
    border-radius: 8px; padding: 10px 12px; text-align: center;
  }}
  .stat-box .sv {{ font-size: 1.15rem; font-weight: 800; }}
  .stat-box .sl {{
    font-size: 0.68rem; color: var(--text2);
    text-transform: uppercase; letter-spacing: 0.5px; margin-top: 2px;
  }}
  .ice-row {{ display: flex; gap: 8px; flex-wrap: wrap; }}
  .ice-b {{
    background: var(--surf2); border: 1px solid var(--border);
    border-radius: 6px; padding: 6px 12px; font-size: 0.8rem;
    display: flex; align-items: center; gap: 6px;
  }}
  .ice-b .sit {{ color: var(--muted); font-size: 0.72rem; }}
  .methodology {{
    padding: 12px 26px; border-top: 1px solid var(--border);
    background: rgba(0,0,0,0.2); font-size: 0.72rem;
    color: var(--muted); line-height: 1.6;
  }}

  /* Scrollbar */
  ::-webkit-scrollbar {{ width: 6px; }}
  ::-webkit-scrollbar-track {{ background: var(--surf); }}
  ::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}

  .footer {{ text-align: center; font-size: 0.72rem; color: var(--muted); margin-top: 28px; }}

  @media (max-width: 640px) {{
    .comp-row {{ grid-template-columns: 100px 1fr 90px; }}
    .war-big .val {{ font-size: 1.5rem; }}
    .card-header {{ flex-direction: column; align-items: flex-start; }}
  }}
</style>
</head>
<body>

<div class="header">
  <h1>⛸ NHL WAR Player Cards</h1>
  <p>Wins Above Replacement · Season {season}</p>
  <button id="update-btn" onclick="document.getElementById('update-modal').style.display='flex'" style="margin-top:14px;padding:8px 22px;border-radius:50px;border:1px solid var(--border);background:var(--surf);color:var(--text2);font-size:0.82rem;cursor:pointer;" onmouseover="this.style.borderColor='var(--accent)';this.style.color='var(--accent)'" onmouseout="this.style.borderColor='var(--border)';this.style.color='var(--text2)'">⬆ Update Data</button>
</div>

{nav_html}

<div id="update-modal" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,0.8);z-index:500;align-items:center;justify-content:center;padding:16px;" onclick="if(event.target===this)this.style.display='none'">
  <div style="background:var(--surf);border:1px solid var(--border);border-radius:16px;max-width:480px;width:100%;padding:32px;position:relative;">
    <button onclick="document.getElementById('update-modal').style.display='none'" style="position:absolute;top:16px;right:16px;background:none;border:none;color:var(--text2);font-size:1.3rem;cursor:pointer;">✕</button>
    <div style="font-size:1.5rem;margin-bottom:12px;">🏒 How to Update Your Data</div>
    <p style="color:var(--text2);font-size:0.9rem;line-height:1.6;margin-bottom:20px;">When you have new CSV files, updating is simple:</p>
    <ol style="color:var(--text);font-size:0.9rem;line-height:2;padding-left:20px;margin-bottom:20px;">
      <li>Replace the CSV files in your <strong>NHL WAR</strong> folder</li>
      <li>Double-click <strong style="color:var(--accent)">Update NHL WAR.command</strong> in that same folder</li>
      <li>Wait a few seconds — all pages open automatically when done</li>
    </ol>
    <div style="background:var(--surf2);border-radius:10px;padding:14px 16px;font-size:0.82rem;color:var(--text2);line-height:1.6;">💡 <strong style="color:var(--text)">Tip:</strong> Keep <em>Update NHL WAR.command</em> in your Dock for quick access.</div>
  </div>
</div>

<div class="search-wrap">
  <input id="search" type="text" placeholder="Search any player (e.g. McDavid, Makar, Shesterkin)…" autocomplete="off">
  <div class="dropdown" id="dropdown"></div>
</div>

<!-- Filters -->
<div class="controls" id="controls">
  <div class="ctrl-row">
    <div class="filter-tabs" id="pos-tabs">
      <div class="tab active" data-pos="ALL">All</div>
      <div class="tab" data-pos="F">Forwards</div>
      <div class="tab" data-pos="C">Centers</div>
      <div class="tab" data-pos="L">Left Wing</div>
      <div class="tab" data-pos="R">Right Wing</div>
      <div class="tab" data-pos="D">Defense</div>
      <div class="tab" data-pos="G">Goalies</div>
    </div>
    <div class="ctrl-divider"></div>
    <select class="ctrl-select" id="team-select">
      <option value="ALL">All Teams</option>
    </select>
    <div class="ctrl-divider"></div>
    <label class="ctrl-label">
      Min GP
      <input class="ctrl-input" id="min-gp" type="number" min="1" max="{max_gp}" value="1" placeholder="1">
    </label>
    <span class="controls-right" id="row-count"></span>
  </div>
</div>

<!-- Leaderboard -->
<div id="leaderboard">
  <div class="lb-wrap">
    <table class="lb-table" id="lb-table">
      <thead>
        <tr>
          <!-- Identity (sticky) -->
          <th class="rank-cell sc1">#</th>
          <th class="sc2" data-col="name">Player</th>
          <th class="sc3" data-col="position">Pos</th>
          <th class="sc4" data-col="team">Team</th>
          <!-- Basic -->
          <th class="num grp-l" data-col="games_played">GP</th>
          <th class="num" data-col="total_icetime_min">TOI</th>
          <!-- WAR -->
          <th class="num grp-l sort-desc" data-col="Total_WAR">WAR</th>
          <th class="num" data-col="WAR_per82">WAR/82</th>
          <th class="num" data-col="EV_Off_WAR">EV Off</th>
          <th class="num" data-col="EV_Def_WAR">EV Def</th>
          <th class="num" data-col="PP_WAR">PP WAR</th>
          <th class="num" data-col="PK_WAR">PK WAR</th>
          <!-- 5-on-5 -->
          <th class="num grp-l" data-col="xGF_pct">xGF%</th>
          <th class="num" data-col="Corsi_pct">CF%</th>
          <th class="num" data-col="EV_xGF60">xGF/60</th>
          <th class="num" data-col="EV_xGA60">xGA/60</th>
          <th class="num" data-col="EV_HDxG60">HDxG/60</th>
          <th class="num" data-col="EV_icetime_min">EV TOI</th>
          <!-- Individual -->
          <th class="num grp-l" data-col="points">Pts</th>
          <th class="num" data-col="goals">G</th>
          <th class="num" data-col="primaryAssists">A1</th>
          <th class="num" data-col="secondaryAssists">A2</th>
          <th class="num" data-col="xGoals">xG</th>
          <th class="num" data-col="Finishing">Finishing</th>
          <th class="num" data-col="iOff60">iOff/60</th>
          <th class="num" data-col="shots">Shots</th>
          <th class="num" data-col="HDxG">HDxG</th>
          <th class="num" data-col="gameScore">GS</th>
          <!-- Power Play -->
          <th class="num grp-l" data-col="PP_icetime_min">PP TOI</th>
          <th class="num" data-col="PP_points">PP Pts</th>
          <th class="num" data-col="PP_goals">PP G</th>
          <th class="num" data-col="PP_xG">PP xG</th>
          <!-- Penalty Kill -->
          <th class="num grp-l" data-col="PK_icetime_min">PK TOI</th>
          <th class="num" data-col="PK_xGA60">PK xGA/60</th>
          <!-- Goalie -->
          <th class="num grp-l" data-col="GSAE">GSAE</th>
          <th class="num" data-col="GSAE_per60">GSAE/60</th>
          <th class="num" data-col="EV_GSAE">EV GSAE</th>
          <th class="num" data-col="HD_GSAE">HD GSAE</th>
          <th class="num" data-col="SavePct">Sv%</th>
          <th class="num" data-col="xSavePct">xSv%</th>
          <th class="num" data-col="shots_faced">SF</th>
          <th class="num" data-col="goals_allowed">GA</th>
          <th class="num" data-col="xGoals_against">xGA</th>
          <th class="num" data-col="ShotFaced60">SF/60</th>
          <!-- Misc -->
          <th class="num grp-l" data-col="hits">Hits</th>
          <th class="num" data-col="takeaways">TK</th>
          <th class="num" data-col="giveaways">GV</th>
          <th class="num" data-col="faceoff_pct">FO%</th>
        </tr>
      </thead>
      <tbody id="lb-body"></tbody>
    </table>
    <div class="lb-footer" id="lb-footer"></div>
  </div>
</div>

<!-- Player Card (hidden until selected) -->
<div id="card-area">
  <div class="back-btn" id="back-btn">← Back to Rankings</div>
  <div id="card-inner"></div>
</div>

<div class="footer">Generated {timestamp} · Data: skaters.csv + goalies.csv</div>

<script>
const PLAYERS = {data_json};
const GPW = {GOALS_PER_WIN};
const PLAYER_HIST = {player_hist_json};
const PLAYER_INDEX = {player_index_json};
const CUR_SEASON = '{season}';
const IN_SUBDIR = {in_subdir_js};

const byId = {{}};
PLAYERS.forEach(p => {{ byId[p.playerId] = p; }});

// Populate team dropdown
const allTeams = [...new Set(PLAYERS.map(p => p.team))].sort();
const teamSel  = document.getElementById('team-select');
allTeams.forEach(t => {{
  const o = document.createElement('option');
  o.value = t; o.textContent = t; teamSel.appendChild(o);
}});

// ── State ──────────────────────────────────────────────────────────────────
let sortCol   = 'Total_WAR';
let sortAsc   = false;
let posFilter = 'ALL';
let teamFilter= 'ALL';
let minGP     = 1;

// ── Filtering & Sorting ────────────────────────────────────────────────────
function filteredPlayers() {{
  return PLAYERS.filter(p => {{
    const posOk  = posFilter === 'ALL' ? true
                 : posFilter === 'F'   ? ['C','L','R'].includes(p.position)
                 : p.position === posFilter || p.pos_group === posFilter;
    const teamOk = teamFilter === 'ALL' || p.team === teamFilter;
    const gpOk   = (p.games_played ?? 0) >= minGP;
    return posOk && teamOk && gpOk;
  }});
}}

const GOALIE_ONLY = new Set(['GSAE','GSAE_per60','EV_GSAE','HD_GSAE','SavePct','xSavePct','shots_faced','goals_allowed','xGoals_against','ShotFaced60']);
const SKATER_ONLY = new Set(['xGF_pct','Corsi_pct','EV_xGF60','EV_xGA60','EV_HDxG60','EV_icetime_min','PP_icetime_min','PP_goals','PP_points','PP_xG','PP_per60','PK_icetime_min','PK_xGA60','Finishing','iOff60','goals','points','primaryAssists','secondaryAssists','xGoals','shots','HDxG','gameScore','hits','takeaways','giveaways','faceoff_pct']);
function sortVal(p, col) {{
  if (GOALIE_ONLY.has(col) && p.pos_group !== 'G') return -9999;
  if (SKATER_ONLY.has(col) && p.pos_group === 'G') return -9999;
  return p[col] ?? -9999;
}}

function sortedPlayers() {{
  return filteredPlayers().slice().sort((a, b) => {{
    let av = sortVal(a, sortCol);
    let bv = sortVal(b, sortCol);
    if (typeof av === 'string') av = av.toLowerCase();
    if (typeof bv === 'string') bv = bv.toLowerCase();
    return sortAsc ? (av < bv ? -1 : av > bv ? 1 : 0)
                   : (av > bv ? -1 : av < bv ? 1 : 0);
  }});
}}

// ── Helpers ────────────────────────────────────────────────────────────────
function warColor(v) {{
  return v > 3 ? 'c-gold' : v > 1 ? 'c-green' : v > -1 ? 'c-muted' : 'c-red';
}}
function warHex(v) {{
  return v > 3 ? '#f4a261' : v > 1 ? '#2dc653' : v > -1 ? '#e2e8f0' : '#e63946';
}}
function barColor(pct) {{
  return pct >= 80 ? 'linear-gradient(90deg,#f4a261,#e9c46a)'
       : pct >= 60 ? 'linear-gradient(90deg,#2dc653,#52b788)'
       : pct >= 40 ? 'linear-gradient(90deg,#4a90d9,#00b4d8)'
       :             'linear-gradient(90deg,#e63946,#c1121f)';
}}
function fmt(v, d=2) {{
  if (v == null) return '—';
  const n = parseFloat(v);
  return isNaN(n) ? '—' : (n >= 0 ? '+' : '') + n.toFixed(d);
}}
function fmtN(v, d=2) {{
  if (v == null) return '—';
  const n = parseFloat(v);
  return isNaN(n) ? '—' : n.toFixed(d);
}}

// Percentile label: rounded whole number, format "(X% among Forwards)"
function pctLabel(pct, group) {{
  const p = Math.round(parseFloat(pct) || 0);
  return group ? `(${{p}}% among ${{group}})` : `${{p}}%`;
}}

function posGroupLabel(pos_group) {{
  return pos_group === 'F' ? 'Forwards' : pos_group === 'D' ? 'Defensemen' : 'Goalies';
}}

function initials(name) {{
  return name.split(' ').map(w => w[0]).join('').slice(0,2).toUpperCase();
}}

// ── Leaderboard ────────────────────────────────────────────────────────────
function renderLeaderboard() {{
  const rows  = sortedPlayers();
  const tbody = document.getElementById('lb-body');

  tbody.innerHTML = rows.map((p, i) => {{
    const isG = p.pos_group === 'G';
    const tw  = p.Total_WAR ?? 0;
    const t82 = p.WAR_per82 ?? 0;
    // Helpers scoped to this row
    const n  = (v, d=1) => v != null ? fmtN(v, d) : '—';
    const sk = (k, d=1) => isG ? '—' : n(p[k], d);   // skater-only
    const gk = (k, d=2) => isG ? n(p[k], d) : '—';   // goalie-only
    const sw = (k)      => isG ? '—' : fmt(p[k] ?? 0); // skater signed
    const xc = v => v >= 52 ? 'c-green' : v <= 48 ? 'c-red' : 'c-muted';
    const pill = isG ? 'pos-pill G' : 'pos-pill';
    const foStr = (!isG && ((p.faceoffsWon||0)+(p.faceoffsLost||0)) > 0)
                  ? n(p.faceoff_pct, 1) + '%' : '—';

    return `<tr data-id="${{p.playerId}}">
      <td class="rank-cell num sc1">${{i+1}}</td>
      <td class="name-cell sc2">${{p.name}}<div class="team-sub">${{p.team}}</div></td>
      <td class="sc3"><span class="${{pill}}">${{p.position}}</span></td>
      <td class="sc4">${{p.team}}</td>
      <td class="num grp-l">${{p.games_played ?? '—'}}</td>
      <td class="num c-muted">${{n(p.total_icetime_min, 1)}}</td>
      <td class="num grp-l war-cell ${{warColor(tw)}}">${{fmt(tw)}}</td>
      <td class="num ${{warColor(t82)}}">${{fmt(t82)}}</td>
      <td class="num ${{isG ? 'c-muted' : warColor(p.EV_Off_WAR??0)}}">${{sw('EV_Off_WAR')}}</td>
      <td class="num ${{isG ? 'c-muted' : warColor(p.EV_Def_WAR??0)}}">${{sw('EV_Def_WAR')}}</td>
      <td class="num ${{isG ? 'c-muted' : warColor(p.PP_WAR??0)}}">${{sw('PP_WAR')}}</td>
      <td class="num ${{isG ? 'c-muted' : warColor(p.PK_WAR??0)}}">${{sw('PK_WAR')}}</td>
      <td class="num grp-l ${{isG ? 'c-muted' : xc(p.xGF_pct??50)}}">${{isG ? '—' : n(p.xGF_pct,1)+'%'}}</td>
      <td class="num c-muted">${{isG ? '—' : n(p.Corsi_pct,1)+'%'}}</td>
      <td class="num c-muted">${{sk('EV_xGF60', 2)}}</td>
      <td class="num c-muted">${{sk('EV_xGA60', 2)}}</td>
      <td class="num c-muted">${{sk('EV_HDxG60', 2)}}</td>
      <td class="num c-muted">${{sk('EV_icetime_min', 1)}}</td>
      <td class="num grp-l">${{sk('points', 0)}}</td>
      <td class="num">${{sk('goals', 0)}}</td>
      <td class="num c-muted">${{sk('primaryAssists', 0)}}</td>
      <td class="num c-muted">${{sk('secondaryAssists', 0)}}</td>
      <td class="num c-muted">${{sk('xGoals', 2)}}</td>
      <td class="num ${{isG ? 'c-muted' : warColor(p.Finishing??0)}}">${{sw('Finishing')}}</td>
      <td class="num c-muted">${{sk('iOff60', 2)}}</td>
      <td class="num c-muted">${{sk('shots', 0)}}</td>
      <td class="num c-muted">${{sk('HDxG', 2)}}</td>
      <td class="num c-muted">${{sk('gameScore', 2)}}</td>
      <td class="num grp-l c-muted">${{sk('PP_icetime_min', 1)}}</td>
      <td class="num c-muted">${{sk('PP_points', 0)}}</td>
      <td class="num c-muted">${{sk('PP_goals', 0)}}</td>
      <td class="num c-muted">${{sk('PP_xG', 2)}}</td>
      <td class="num grp-l c-muted">${{sk('PK_icetime_min', 1)}}</td>
      <td class="num c-muted">${{sk('PK_xGA60', 2)}}</td>
      <td class="num grp-l ${{isG ? warColor((p.GSAE??0)/3) : 'c-muted'}}">${{gk('GSAE', 2)}}</td>
      <td class="num c-muted">${{gk('GSAE_per60', 2)}}</td>
      <td class="num c-muted">${{gk('EV_GSAE', 2)}}</td>
      <td class="num c-muted">${{gk('HD_GSAE', 2)}}</td>
      <td class="num c-muted">${{isG ? fmtN((p.SavePct??0)*100, 2)+'%' : '—'}}</td>
      <td class="num c-muted">${{isG ? fmtN((p.xSavePct??0)*100, 2)+'%' : '—'}}</td>
      <td class="num c-muted">${{gk('shots_faced', 0)}}</td>
      <td class="num c-muted">${{gk('goals_allowed', 0)}}</td>
      <td class="num c-muted">${{gk('xGoals_against', 2)}}</td>
      <td class="num c-muted">${{gk('ShotFaced60', 1)}}</td>
      <td class="num grp-l c-muted">${{sk('hits', 0)}}</td>
      <td class="num c-muted">${{sk('takeaways', 0)}}</td>
      <td class="num c-muted">${{sk('giveaways', 0)}}</td>
      <td class="num c-muted">${{foStr}}</td>
    </tr>`;
  }}).join('');

  document.getElementById('row-count').textContent =
    `${{rows.length}} player${{rows.length !== 1 ? 's' : ''}}`;
  document.getElementById('lb-footer').textContent =
    'Scroll right for more stats. WAR/82 = prorated. Skaters: Finishing = Goals − xGoals. Goalies: GSAE = xGoals Faced − Goals Allowed.';

  tbody.querySelectorAll('tr').forEach(tr => {{
    tr.addEventListener('click', () => openCard(byId[parseInt(tr.dataset.id)]));
  }});
}}

// ── Sort ───────────────────────────────────────────────────────────────────
document.querySelectorAll('#lb-table th[data-col]').forEach(th => {{
  th.addEventListener('click', () => {{
    const col = th.dataset.col;
    sortAsc = (sortCol === col) ? !sortAsc : (col === 'name' || col === 'team' || col === 'position');
    sortCol = col;
    document.querySelectorAll('#lb-table th').forEach(h => h.classList.remove('sort-asc','sort-desc'));
    th.classList.add(sortAsc ? 'sort-asc' : 'sort-desc');
    renderLeaderboard();
  }});
}});

// ── Position Tabs ──────────────────────────────────────────────────────────
document.getElementById('pos-tabs').addEventListener('click', e => {{
  const tab = e.target.closest('.tab');
  if (!tab) return;
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  tab.classList.add('active');
  posFilter = tab.dataset.pos;
  renderLeaderboard();
}});

// ── Team Filter ────────────────────────────────────────────────────────────
document.getElementById('team-select').addEventListener('change', e => {{
  teamFilter = e.target.value; renderLeaderboard();
}});

// ── Min GP & Min WAR inputs ────────────────────────────────────────────────
function debounce(fn, ms) {{
  let t; return (...args) => {{ clearTimeout(t); t = setTimeout(() => fn(...args), ms); }};
}}
const gpInput  = document.getElementById('min-gp');
gpInput.addEventListener('input', debounce(() => {{
  minGP = parseInt(gpInput.value) || 1; renderLeaderboard();
}}, 300));

// ── Search ─────────────────────────────────────────────────────────────────
const searchEl   = document.getElementById('search');
const dropdownEl = document.getElementById('dropdown');
const normalize  = s => s.toLowerCase().replace(/[^a-z0-9 ]/g,'');

searchEl.addEventListener('input', () => {{
  const q = normalize(searchEl.value.trim());
  if (q.length < 2) {{ dropdownEl.classList.remove('show'); return; }}

  // Current season matches
  const curHits = PLAYERS.filter(p => normalize(p.name).includes(q))
    .sort((a,b) => (b.Total_WAR||0)-(a.Total_WAR||0)).slice(0, 8);
  const curIds = new Set(curHits.map(p => String(p.playerId)));

  // Historical matches (players not in current season)
  const histEntries = Object.entries(PLAYER_INDEX)
    .filter(([pid, p]) => !curIds.has(pid) && normalize(p.n).includes(q))
    .sort((a,b) => a[1].n.localeCompare(b[1].n))
    .slice(0, 6);

  if (!curHits.length && !histEntries.length) {{ dropdownEl.classList.remove('show'); return; }}

  let html = '';
  if (curHits.length) {{
    html += curHits.map(p => `
      <div class="dd-item" data-id="${{p.playerId}}" data-cur="1">
        <span>${{p.name}}<span class="team-t"> ${{p.team}}</span></span>
        <span class="pos-b">${{p.position}}</span>
      </div>`).join('');
  }}
  if (histEntries.length) {{
    if (curHits.length) html += `<div class="dd-sep">Other Seasons</div>`;
    html += histEntries.map(([pid, p]) => `
      <div class="dd-item dd-hist" data-id="${{pid}}" data-cur="0">
        <span>${{p.n}}<span class="team-t"> ${{p.t}}</span></span>
        <span class="pos-b">${{p.pos}} · ${{p.s}}</span>
      </div>`).join('');
  }}
  dropdownEl.innerHTML = html;
  dropdownEl.classList.add('show');
}});

dropdownEl.addEventListener('click', e => {{
  const item = e.target.closest('.dd-item');
  if (!item) return;
  const pid = item.dataset.id;
  if (item.dataset.cur === '1') {{
    const p = byId[parseInt(pid)];
    searchEl.value = p.name;
    dropdownEl.classList.remove('show');
    openCard(p);
  }} else {{
    const pi = PLAYER_INDEX[pid];
    if (!pi) return;
    searchEl.value = pi.n;
    dropdownEl.classList.remove('show');
    // Try opening directly if the player happens to be in this season
    const pDirect = byId[parseInt(pid)];
    if (pDirect) {{ openCard(pDirect); return; }}
    // Navigate to the right page depending on where we are
    const isCurSeason = pi.s === CUR_SEASON;
    let href;
    if (IN_SUBDIR) {{
      href = isCurSeason
        ? `../NHL_WAR_Cards.html#p=${{pid}}`
        : `NHL_WAR_Cards_${{pi.s}}.html#p=${{pid}}`;
    }} else {{
      href = isCurSeason
        ? `NHL_WAR_Cards.html#p=${{pid}}`
        : `Old Seasons/NHL_WAR_Cards_${{pi.s}}.html#p=${{pid}}`;
    }}
    location.href = href;
  }}
}});
document.addEventListener('click', e => {{
  if (!e.target.closest('.search-wrap')) dropdownEl.classList.remove('show');
}});
searchEl.addEventListener('keydown', e => {{
  if (e.key === 'Enter') {{ const f = dropdownEl.querySelector('.dd-item'); if (f) f.click(); }}
}});

// ── Season-aware player navigation ─────────────────────────────────────────
let currentPlayer = null;
const navSel = document.querySelector('.site-nav select');
let origOptions = [];
if (navSel) {{
  origOptions = [...navSel.options].map(o => ({{v: o.value, t: o.text, s: o.selected}}));
  // Replace inline onchange so we can intercept it when a card is open
  navSel.onchange = function() {{ location.href = this.value; }};
}}

function updateNavForPlayer(player) {{
  if (!navSel || !origOptions.length) return;
  const hist = PLAYER_HIST[player.playerId] || [];
  const playerSeasons = new Set(hist.map(e => e.s));
  navSel.innerHTML = '';
  origOptions.forEach(opt => {{
    if (!playerSeasons.has(opt.t)) return;   // skip seasons they didn't play
    const o = document.createElement('option');
    o.value = opt.v + '#p=' + player.playerId;
    o.text  = opt.t;
    o.selected = opt.t === player.season;
    navSel.appendChild(o);
  }});
}}

function restoreNav() {{
  if (!navSel || !origOptions.length) return;
  navSel.innerHTML = '';
  origOptions.forEach(opt => {{
    const o = document.createElement('option');
    o.value = opt.v; o.text = opt.t; o.selected = opt.s;
    navSel.appendChild(o);
  }});
}}

// ── Open / Close Card ──────────────────────────────────────────────────────
function openCard(player) {{
  currentPlayer = player;
  history.replaceState(null, '', '#p=' + player.playerId);
  updateNavForPlayer(player);
  document.getElementById('leaderboard').style.display = 'none';
  document.getElementById('controls').style.display    = 'none';
  const ca = document.getElementById('card-area');
  ca.style.display = 'block';
  document.getElementById('card-inner').innerHTML =
    player.pos_group === 'G' ? renderGoalieCard(player) : renderSkaterCard(player);
  window.scrollTo({{top:0, behavior:'smooth'}});
}}
document.getElementById('back-btn').addEventListener('click', () => {{
  currentPlayer = null;
  history.replaceState(null, '', window.location.pathname + window.location.search);
  restoreNav();
  document.getElementById('card-area').style.display   = 'none';
  document.getElementById('leaderboard').style.display = 'block';
  document.getElementById('controls').style.display    = 'flex';
  searchEl.value = '';
}});

// ── Card Builders ──────────────────────────────────────────────────────────
function compRow(label, warVal, pct, group) {{
  const w   = Math.max(2, Math.min(100, pct));
  return `
    <div class="comp-row">
      <div class="comp-lbl">${{label}}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${{w}}%;background:${{barColor(pct)}}"></div></div>
      <div class="comp-vals">
        <div class="cv" style="color:${{warHex(warVal)}}">${{fmt(warVal)}}</div>
        <div class="cp">${{pctLabel(pct, group)}}</div>
      </div>
    </div>`;
}}

function statBox(label, value) {{
  return `<div class="stat-box"><div class="sv">${{value}}</div><div class="sl">${{label}}</div></div>`;
}}

function renderWARChart(playerId, currentSeason) {{
  const hist = PLAYER_HIST[playerId];
  if (!hist || hist.length < 2) return '';
  const W=480,H=110,PL=34,PR=8,PT=12,PB=28;
  const plotW=W-PL-PR, plotH=H-PT-PB;
  const vals=hist.map(e=>e.w);
  const maxV=Math.max(...vals,1), minV=Math.min(...vals,0);
  const range=maxV-minV||1;
  const step=plotW/hist.length;
  const barW=Math.max(4,Math.floor(step)-2);
  const zeroY=PT+plotH*(maxV/range);
  const cid='wtt_'+String(playerId).replace(/\W/g,'_');
  let bars='',labels='';
  hist.forEach((e,i)=>{{
    const cx=PL+i*step+step/2;
    const isCur=e.s===currentSeason;
    const bh=Math.abs(e.w/range*plotH);
    const y=e.w>=0?zeroY-bh:zeroY;
    const col=isCur?'#00b4d8':e.w>=0?'#2dc653':'#e63946';
    const teamStr=e.t?' · '+e.t:'';
    const tip=e.s+': '+(e.w>=0?'+':'')+e.w.toFixed(2)+' WAR'+teamStr;
    bars+=`<rect x="${{(cx-step/2+1).toFixed(1)}}" y="${{PT}}" width="${{(step-2).toFixed(1)}}" height="${{plotH}}" fill="transparent" style="cursor:default" onmouseover="(function(){{var t=document.getElementById('${{cid}}');if(t){{t.style.display='block';t.textContent='${{tip}}'}}}})()" onmouseout="(function(){{var t=document.getElementById('${{cid}}');if(t)t.style.display='none'}})()"/>`;
    bars+=`<rect x="${{(cx-barW/2).toFixed(1)}}" y="${{y.toFixed(1)}}" width="${{barW}}" height="${{Math.max(1,bh).toFixed(1)}}" fill="${{col}}" rx="1" opacity="${{isCur?1:0.65}}" pointer-events="none"/>`;
    if(isCur||i%Math.max(1,Math.round(hist.length/6))===0){{
      const yr=e.s.slice(2,4)+'-'+e.s.slice(7);
      labels+=`<text x="${{cx.toFixed(1)}}" y="${{(H-4).toFixed(1)}}" text-anchor="middle" font-size="8" fill="${{isCur?'#00b4d8':'#64748b'}}" pointer-events="none">${{yr}}</text>`;
    }}
  }});
  const zLine=`<line x1="${{PL}}" y1="${{zeroY.toFixed(1)}}" x2="${{W-PR}}" y2="${{zeroY.toFixed(1)}}" stroke="#2a3a50" stroke-width="1" pointer-events="none"/>`;
  const maxLbl=`<text x="${{(PL-3)}}" y="${{(PT+6)}}" text-anchor="end" font-size="8" fill="#64748b" pointer-events="none">${{maxV.toFixed(1)}}</text>`;
  const zLbl=`<text x="${{(PL-3)}}" y="${{(zeroY+4).toFixed(1)}}" text-anchor="end" font-size="8" fill="#64748b" pointer-events="none">0</text>`;
  return `<div class="section-title" style="margin-top:18px">WAR History</div>
<div style="position:relative;margin-top:6px">
  <div id="${{cid}}" style="display:none;position:absolute;top:0;left:50%;transform:translateX(-50%);background:#1e2a3a;border:1px solid #2a3a50;border-radius:6px;padding:3px 10px;font-size:0.76rem;color:#e2e8f0;white-space:nowrap;z-index:10;pointer-events:none"></div>
  <div style="overflow-x:auto"><svg width="100%" viewBox="0 0 ${{W}} ${{H}}" style="max-width:${{W}}px;display:block">${{zLine}}${{bars}}${{labels}}${{maxLbl}}${{zLbl}}</svg></div>
</div>`;
}}

function renderSkaterCard(p) {{
  const pg = posGroupLabel(p.pos_group);

  const sumItems = [
    {{label:'Total WAR',    val:p.Total_WAR,   pct:p.Total_WAR_pct_pos,   cls:'total'}},
    {{label:'EV Offense',  val:p.EV_Off_WAR,  pct:p.EV_Off_WAR_pct_pos}},
    {{label:'EV Defense',  val:p.EV_Def_WAR,  pct:p.EV_Def_WAR_pct_pos}},
    {{label:'Power Play',  val:p.PP_WAR,       pct:p.PP_WAR_pct_pos}},
    {{label:'Penalty Kill',val:p.PK_WAR,       pct:p.PK_WAR_pct_pos}},
  ];

  const sumHtml = sumItems.map(it => `
    <div class="war-big ${{it.cls||''}}">
      <div class="lbl">${{it.label}}</div>
      <div class="val" style="color:${{warHex(it.val||0)}}">${{fmt(it.val||0)}}</div>
      <div class="pct-lbl">${{pctLabel(it.pct||0, pg)}}</div>
    </div>`).join('');

  return `
    <div class="player-card">
      <div class="card-header">
        <div class="avatar">${{initials(p.name)}}</div>
        <div class="player-info">
          <h2>${{p.name}}</h2>
          <div class="player-meta">
            <span class="meta-b pos">${{p.position}}</span>
            <span class="meta-b">${{p.team}}</span>
            <span class="meta-b">${{p.games_played}} GP · Season ${{p.season}}</span>
          </div>
        </div>
      </div>

      <div class="war-summary">${{sumHtml}}</div>

      <div class="card-body">
        <div>
          <div class="section-title">WAR Breakdown — Season Total vs Per 82</div>
          <div class="war-comp">
            ${{compRow('EV Offense',   p.EV_Off_WAR,  p.EV_Off_WAR_pct_pos||0,  pg)}}
            ${{compRow('EV Defense',   p.EV_Def_WAR,  p.EV_Def_WAR_pct_pos||0,  pg)}}
            ${{compRow('Power Play',   p.PP_WAR,       p.PP_WAR_pct_pos||0,       pg)}}
            ${{compRow('Penalty Kill', p.PK_WAR,       p.PK_WAR_pct_pos||0,       pg)}}
            ${{compRow('Finishing',    p.Finishing,    p.Finishing_pct_pos||0,    pg)}}
          </div>
          <div class="ice-row" style="margin-top:12px">
            <div class="ice-b"><span class="sit">Total WAR</span>${{fmt(p.Total_WAR)}}</div>
            <div class="ice-b" style="border-color:var(--accent)"><span class="sit">WAR/82</span>${{fmt(p.WAR_per82)}}</div>
            <div class="ice-b"><span class="sit">EV Off/82</span>${{fmt(p.EV_Off_WAR_per82)}}</div>
            <div class="ice-b"><span class="sit">EV Def/82</span>${{fmt(p.EV_Def_WAR_per82)}}</div>
            <div class="ice-b"><span class="sit">PP/82</span>${{fmt(p.PP_WAR_per82)}}</div>
            <div class="ice-b"><span class="sit">PK/82</span>${{fmt(p.PK_WAR_per82)}}</div>
          </div>
        </div>
        <div>
          <div class="section-title">Scoring</div>
          <div class="stats-grid">
            ${{statBox('Goals', p.goals??'—')}}
            ${{statBox('Points', p.points??'—')}}
            ${{statBox('1° Assists', p.primaryAssists??'—')}}
            ${{statBox('2° Assists', p.secondaryAssists??'—')}}
            ${{statBox('PP Points', p.PP_points??'—')}}
            ${{statBox('PP 1° A', p.PP_primaryAssists??'—')}}
            ${{statBox('PP 2° A', p.PP_secondaryAssists??'—')}}
            ${{statBox('PK Points', p.PK_points??'—')}}
            ${{statBox('Shots', p.shots??'—')}}
            ${{statBox('xGoals', fmtN(p.xGoals,1))}}
            ${{statBox('Finishing', fmt(p.Finishing,1))}}
            ${{(p.faceoffsWon||0)+(p.faceoffsLost||0) > 0 ? statBox('FO%', fmtN(p.faceoff_pct,1)+'%') : statBox('FO%', '—')}}
          </div>
        </div>
        <div>
          <div class="section-title">On-Ice &amp; Defensive</div>
          <div class="stats-grid">
            ${{statBox('xGF%', fmtN(p.xGF_pct,1)+'%')}}
            ${{statBox('Corsi%', fmtN(p.Corsi_pct,1)+'%')}}
            ${{statBox('iOff/60', fmtN(p.iOff60,2))}}
            ${{statBox('xGF/60', fmtN(p.EV_xGF60,2))}}
            ${{statBox('xGA/60', fmtN(p.EV_xGA60,2))}}
            ${{statBox('HDxG', fmtN(p.HDxG,2))}}
            ${{statBox('Hits', p.hits??'—')}}
            ${{statBox('Takeaways', p.takeaways??'—')}}
            ${{statBox('Giveaways', p.giveaways??'—')}}
          </div>
        </div>
        <div>
          <div class="section-title">Ice Time</div>
          <div class="ice-row">
            <div class="ice-b"><span class="sit">TOI</span>${{fmtN(p.total_icetime_min,0)}} min</div>
            <div class="ice-b"><span class="sit">5v5</span>${{fmtN(p.EV_icetime_min,0)}} min</div>
            <div class="ice-b"><span class="sit">PP</span>${{fmtN(p.PP_icetime_min,0)}} min</div>
            <div class="ice-b"><span class="sit">PK</span>${{fmtN(p.PK_icetime_min,0)}} min</div>
          </div>
        </div>
      </div>

      ${{renderWARChart(p.playerId, p.season)}}

      <div class="methodology">
        <strong>WAR:</strong>
        EV Offense = individual xGoals + weighted assists (0.75× primary, 0.5× secondary) per 60 vs positional replacement level (15th %ile).
        EV Defense = on-ice vs off-ice xGoals-against share differential.
        PP = scoring rate on 5-on-4 vs 40% of avg PP production.
        PK = xGA suppression on 4-on-5 vs league avg.
        WAR/82 = total WAR ÷ GP × 82 (pace-adjusted).
        1 WAR = ${{GPW}} Goals Above Replacement. Percentiles vs same position (${{pg}}).
      </div>
    </div>`;
}}

function renderGoalieCard(p) {{
  const sumHtml = `
    <div class="war-big total">
      <div class="lbl">Goalie WAR</div>
      <div class="val" style="color:${{warHex(p.Goalie_WAR||0)}}">${{fmt(p.Goalie_WAR||0)}}</div>
      <div class="pct-lbl">${{pctLabel(p.Goalie_WAR_pct||0, 'Goalies')}}</div>
    </div>
    <div class="war-big">
      <div class="lbl">WAR/82</div>
      <div class="val" style="color:${{warHex(p.WAR_per82||0)}}">${{fmt(p.WAR_per82||0)}}</div>
      <div class="pct-lbl">${{pctLabel(p.WAR_per82_pct||0, 'Goalies')}}</div>
    </div>
    <div class="war-big">
      <div class="lbl">GSAE</div>
      <div class="val" style="color:${{warHex((p.GSAE||0)/3)}}">${{fmt(p.GSAE||0,1)}}</div>
      <div class="pct-lbl">${{pctLabel(p.GSAE_pct||0, 'Goalies')}}</div>
    </div>
    <div class="war-big">
      <div class="lbl">Save %</div>
      <div class="val">${{((p.SavePct||0)*100).toFixed(2)}}%</div>
      <div class="pct-lbl">${{pctLabel(p.SavePct_pct||0, 'Goalies')}}</div>
    </div>`;

  return `
    <div class="player-card">
      <div class="card-header">
        <div class="avatar goalie">${{initials(p.name)}}</div>
        <div class="player-info">
          <h2>${{p.name}}</h2>
          <div class="player-meta">
            <span class="meta-b pos G">G</span>
            <span class="meta-b">${{p.team}}</span>
            <span class="meta-b">${{p.games_played}} GP · Season ${{p.season}}</span>
          </div>
        </div>
      </div>
      <div class="war-summary">${{sumHtml}}</div>
      <div class="card-body">
        <div>
          <div class="section-title">WAR Components</div>
          <div class="war-comp">
            ${{compRow('Goalie WAR',  p.Goalie_WAR||0,  p.Goalie_WAR_pct||0,  'Goalies')}}
            ${{compRow('WAR/82',      p.WAR_per82||0,   p.WAR_per82_pct||0,   'Goalies')}}
            ${{compRow('GSAE Total',  p.GSAE||0,         p.GSAE_pct||0,         'Goalies')}}
            ${{compRow('GSAE per 60', p.GSAE_per60||0,   p.GSAE_per60_pct||0,   'Goalies')}}
            ${{compRow('HD GSAE',     p.HD_GSAE||0,      p.HD_GSAE_pct||0,      'Goalies')}}
          </div>
        </div>
        <div>
          <div class="section-title">Save Performance</div>
          <div class="stats-grid">
            ${{statBox('Save %', ((p.SavePct||0)*100).toFixed(2)+'%')}}
            ${{statBox('xSave %', ((p.xSavePct||0)*100).toFixed(2)+'%')}}
            ${{statBox('Sv% – xSv%', fmt(((p.SavePct||0)-(p.xSavePct||0))*100,2)+'%')}}
            ${{statBox('GSAE', fmt(p.GSAE,1))}}
            ${{statBox('EV GSAE', fmt(p.EV_GSAE,1))}}
            ${{statBox('HD GSAE', fmt(p.HD_GSAE,1))}}
            ${{statBox('GSAE/60', fmt(p.GSAE_per60,2))}}
            ${{statBox('Shots Faced', p.shots_faced??'—')}}
            ${{statBox('Shots/60', fmtN(p.ShotFaced60,1))}}
            ${{statBox('Goals Allow', p.goals_allowed??'—')}}
            ${{statBox('xGA', fmtN(p.xGoals_against,1))}}
          </div>
        </div>
      </div>
      ${{renderWARChart(p.playerId, p.season)}}

      <div class="methodology">
        <strong>Goalie WAR:</strong> GSAE = xGoals Faced − Goals Allowed. 1 WAR = ${{GPW}} GSAE.
        WAR/82 = pace-adjusted to a full 82-game season. Percentiles vs all Goalies.
      </div>
    </div>`;
}}

// ── Init ───────────────────────────────────────────────────────────────────
renderLeaderboard();

// Auto-open player card from URL hash (e.g. when switching seasons from a card)
(function() {{
  const hash = location.hash;
  if (!hash.startsWith('#p=')) return;
  const pid = hash.slice(3);
  const player = PLAYERS.find(p => String(p.playerId) === pid);
  if (player) openCard(player);
}})();
</script>
</body>
</html>"""


# ─── Main Builder ─────────────────────────────────────────────────────────────


# ─── Shared Page Chrome ────────────────────────────────────────────────────────

PAGE_STYLE = """
<style>
  :root {
    --bg:     #0a0e1a; --surf:   #111827; --surf2:  #1e2a3a;
    --border: #2a3a50; --accent: #00b4d8; --accent2:#0077b6;
    --gold:   #f4a261; --red:    #e63946; --green:  #2dc653;
    --muted:  #64748b; --text:   #e2e8f0; --text2:  #94a3b8; --r: 12px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg); color: var(--text);
    font-family: 'Segoe UI', system-ui, sans-serif;
    min-height: 100vh; padding: 24px 16px;
  }
  .site-nav {
    display: flex; justify-content: center; gap: 6px;
    margin: 0 auto 28px; max-width: 480px;
  }
  .site-nav a {
    padding: 8px 22px; border-radius: 50px;
    border: 1px solid var(--border); background: var(--surf);
    color: var(--text2); font-size: 0.85rem; font-weight: 600;
    text-decoration: none; transition: all 0.15s;
  }
  .site-nav a:hover { border-color: var(--accent); color: var(--accent); }
  .site-nav a.active { background: var(--accent2); border-color: var(--accent); color: #fff; }
  .header { text-align: center; margin-bottom: 28px; }
  .header h1 {
    font-size: 2.2rem; font-weight: 800; letter-spacing: -0.5px;
    background: linear-gradient(135deg, var(--accent), var(--gold));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
  }
  .header p { color: var(--text2); margin-top: 6px; font-size: 0.9rem; }
  .controls {
    max-width: 1300px; margin: 0 auto 16px;
    display: flex; gap: 8px; flex-wrap: wrap; align-items: center;
  }
  .tab {
    padding: 6px 14px; border-radius: 20px; border: 1px solid var(--border);
    background: var(--surf); color: var(--text2); font-size: 0.82rem;
    font-weight: 600; cursor: pointer; transition: all 0.15s; white-space: nowrap;
  }
  .tab:hover { border-color: var(--accent); color: var(--accent); }
  .tab.active { background: var(--accent2); border-color: var(--accent); color: #fff; }
  .ctrl-select {
    padding: 6px 12px; border-radius: 20px; border: 1px solid var(--border);
    background: var(--surf); color: var(--text); font-size: 0.82rem;
    cursor: pointer; outline: none;
  }
  .ctrl-label { font-size: 0.78rem; color: var(--text2); display: flex; align-items: center; gap: 6px; }
  .ctrl-input {
    width: 72px; padding: 5px 8px; border-radius: 8px;
    border: 1px solid var(--border); background: var(--surf);
    color: var(--text); font-size: 0.82rem; outline: none; text-align: center;
  }
  .ctrl-right { margin-left: auto; color: var(--muted); font-size: 0.8rem; }
  .wrap { max-width: 1300px; margin: 0 auto; }
  .lb-wrap { background: var(--surf); border: 1px solid var(--border); border-radius: var(--r); overflow: hidden; }
  table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  thead { background: var(--surf2); border-bottom: 2px solid var(--border); }
  th {
    padding: 11px 12px; text-align: left; font-size: 0.72rem; font-weight: 700;
    letter-spacing: 0.8px; text-transform: uppercase; color: var(--text2);
    white-space: nowrap; cursor: pointer; user-select: none;
  }
  th.num { text-align: right; }
  th:hover { color: var(--accent); }
  th.sort-asc::after  { content: ' ↑'; color: var(--accent); }
  th.sort-desc::after { content: ' ↓'; color: var(--accent); }
  td { padding: 10px 12px; border-bottom: 1px solid rgba(42,58,80,0.5); white-space: nowrap; }
  td.num { text-align: right; }
  tbody tr { cursor: default; transition: background 0.12s; }
  tbody tr:hover { background: rgba(0,180,216,0.06); }
  tbody tr:last-child td { border-bottom: none; }
  .rank-c { color: var(--muted); font-size: 0.8rem; }
  .name-c { font-weight: 700; }
  .sub-c  { color: var(--text2); font-size: 0.8rem; font-weight: 400; }
  .c-green { color: var(--green); font-weight: 700; }
  .c-red   { color: var(--red);   font-weight: 700; }
  .c-gold  { color: var(--gold);  font-weight: 700; }
  .c-muted { color: var(--text2); }
  .badge {
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 0.72rem; font-weight: 700;
  }
  .badge-line    { background: rgba(0,180,216,0.15); color: var(--accent); }
  .badge-pairing { background: rgba(244,162,97,0.15); color: var(--gold); }
  .lb-footer {
    padding: 10px 16px; background: var(--surf2); border-top: 1px solid var(--border);
    font-size: 0.75rem; color: var(--muted); text-align: right;
  }
  .section-hdr {
    font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px;
    color: var(--text2); padding: 6px 12px; background: rgba(42,58,80,0.3);
    border-bottom: 1px solid var(--border); font-weight: 700;
  }
  .col-group-hdr {
    text-align: center; font-size: 0.65rem; text-transform: uppercase;
    letter-spacing: 0.8px; color: var(--accent); border-bottom: 1px solid var(--border);
    padding: 5px 0; background: rgba(0,180,216,0.04);
  }
</style>
"""

def w(v, nd=2):
    """Round float, return 0 for NaN."""
    try:
        f = float(v)
        return round(f, nd) if not np.isnan(f) else 0
    except Exception:
        return 0

def pct_color(val, lo=45, hi=55):
    """Return CSS class for a percentage: red below lo, green above hi, else muted."""
    if val >= hi:   return "c-green"
    if val <= lo:   return "c-red"
    return "c-muted"

def gf_color(gf, ga):
    if gf > ga:  return "c-green"
    if gf < ga:  return "c-red"
    return "c-muted"


# ─── Teams HTML ───────────────────────────────────────────────────────────────

def _OLD_build_teams_html(teams_csv, season):
    df = pd.read_csv(teams_csv)

    ev   = df[df["situation"] == "5on5"].copy()
    pp   = df[df["situation"] == "5on4"].copy()
    pk   = df[df["situation"] == "4on5"].copy()
    all_ = df[df["situation"] == "all"].copy()

    records = []
    for team in sorted(all_["team"].unique()):
        def row(df_, team=team): return df_[df_["team"] == team]
        ra = row(all_)
        re = row(ev)
        rp = row(pp)
        rk = row(pk)
        if ra.empty: continue
        ra = ra.iloc[0]; re = re.iloc[0] if not re.empty else None
        rp = rp.iloc[0] if not rp.empty else None
        rk = rk.iloc[0] if not rk.empty else None

        gp = w(ra["games_played"], 0)
        gf = w(ra["goalsFor"],     0)
        ga = w(ra["goalsAgainst"], 0)
        xgf_a = w(ra["xGoalsFor"])
        xga_a = w(ra["xGoalsAgainst"])
        sog_f = w(ra["shotsOnGoalFor"],     0)
        sog_a = w(ra["shotsOnGoalAgainst"], 0)
        fo_w  = w(ra.get("faceOffsWonFor",   0), 0)
        fo_l  = w(ra.get("faceOffsWonAgainst", 0), 0)
        fo_pct = round(fo_w / max(fo_w + fo_l, 1) * 100, 1)

        ev_xgf_pct = round(w(re["xGoalsPercentage"]) * 100, 1) if re is not None else 0
        ev_cf_pct  = round(w(re["corsiPercentage"])  * 100, 1) if re is not None else 0
        ev_ff_pct  = round(w(re["fenwickPercentage"]) * 100, 1) if re is not None else 0
        ev_xgf     = w(re["xGoalsFor"])     if re is not None else 0
        ev_xga     = w(re["xGoalsAgainst"]) if re is not None else 0
        ev_gf      = w(re["goalsFor"],  0)  if re is not None else 0
        ev_ga      = w(re["goalsAgainst"], 0) if re is not None else 0
        ev_hdxgf   = w(re["highDangerxGoalsFor"])     if re is not None else 0
        ev_hdxga   = w(re["highDangerxGoalsAgainst"]) if re is not None else 0
        ev_hdgf    = w(re["highDangerGoalsFor"],   0) if re is not None else 0
        ev_hdga    = w(re["highDangerGoalsAgainst"],0) if re is not None else 0

        pp_gf    = w(rp["goalsFor"], 0)    if rp is not None else 0
        pp_xgf   = w(rp["xGoalsFor"])      if rp is not None else 0
        pp_toi   = round(w(rp.get("iceTime", 0)) / 60, 1) if rp is not None else 0
        pk_ga    = w(rk["goalsAgainst"], 0) if rk is not None else 0
        pk_xga   = w(rk["xGoalsAgainst"])  if rk is not None else 0
        pk_toi   = round(w(rk.get("iceTime", 0)) / 60, 1) if rk is not None else 0

        records.append(dict(
            team=str(team), gp=int(gp),
            gf=int(gf), ga=int(ga), xgf_a=xgf_a, xga_a=xga_a,
            sog_f=int(sog_f), sog_a=int(sog_a), fo_pct=fo_pct,
            ev_xgf_pct=ev_xgf_pct, ev_cf_pct=ev_cf_pct, ev_ff_pct=ev_ff_pct,
            ev_xgf=ev_xgf, ev_xga=ev_xga, ev_gf=int(ev_gf), ev_ga=int(ev_ga),
            ev_hdxgf=ev_hdxgf, ev_hdxga=ev_hdxga, ev_hdgf=int(ev_hdgf), ev_hdga=int(ev_hdga),
            pp_gf=int(pp_gf), pp_xgf=pp_xgf, pp_toi=pp_toi,
            pk_ga=int(pk_ga), pk_xga=pk_xga, pk_toi=pk_toi,
        ))

    data_json = json.dumps(records)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NHL WAR – Teams {season}</title>
{PAGE_STYLE}
</head>
<body>
<div class="header">
  <h1>🏒 NHL Team Stats</h1>
  <p>{season} Season &nbsp;·&nbsp; All Situations &amp; 5-on-5</p>
</div>
<nav class="site-nav">
  <a href="NHL_WAR_Cards.html">Players</a>
  <a href="NHL_WAR_Teams.html" class="active">Teams</a>
  <a href="NHL_WAR_Lines.html">Lines</a>
</nav>

<div class="controls">
  <div class="tab active" data-view="ev"  onclick="switchView('ev')">5-on-5</div>
  <div class="tab"        data-view="all" onclick="switchView('all')">Overall</div>
  <div class="tab"        data-view="pp"  onclick="switchView('pp')">Power Play</div>
  <div class="tab"        data-view="pk"  onclick="switchView('pk')">Penalty Kill</div>
  <span class="ctrl-right" id="row-count"></span>
</div>

<div class="wrap">
  <div class="lb-wrap">
    <table id="lb">
      <thead id="thead"></thead>
      <tbody id="tbody"></tbody>
    </table>
    <div class="lb-footer">Data: Natural Stat Trick · Built {datetime.now().strftime("%b %d, %Y")}</div>
  </div>
</div>

<script>
const TEAMS = {data_json};
let SORT_COL = 'ev_xgf_pct', SORT_DIR = -1, VIEW = 'ev';

const VIEWS = {{
  ev: {{
    cols: [
      {{k:'rank',  label:'#',       cl:'',     fmt: (_,i)=>i+1,              td:'rank-c'}},
      {{k:'team',  label:'Team',    cl:'',     fmt: r=>r.team,               td:'name-c'}},
      {{k:'gp',    label:'GP',      cl:'num',  fmt: r=>r.gp}},
      {{k:'ev_xgf_pct', label:'xGF%', cl:'num', fmt: r=>r.ev_xgf_pct.toFixed(1)+'%', color: r=>pctCls(r.ev_xgf_pct)}},
      {{k:'ev_cf_pct',  label:'CF%',  cl:'num', fmt: r=>r.ev_cf_pct.toFixed(1)+'%',  color: r=>pctCls(r.ev_cf_pct)}},
      {{k:'ev_ff_pct',  label:'FF%',  cl:'num', fmt: r=>r.ev_ff_pct.toFixed(1)+'%',  color: r=>pctCls(r.ev_ff_pct)}},
      {{k:'ev_gf',  label:'GF',  cl:'num', fmt: r=>r.ev_gf,  color: r=>gfCls(r.ev_gf, r.ev_ga)}},
      {{k:'ev_ga',  label:'GA',  cl:'num', fmt: r=>r.ev_ga,  color: r=>gaCls(r.ev_gf, r.ev_ga)}},
      {{k:'ev_xgf', label:'xGF', cl:'num', fmt: r=>r.ev_xgf.toFixed(1)}},
      {{k:'ev_xga', label:'xGA', cl:'num', fmt: r=>r.ev_xga.toFixed(1)}},
      {{k:'ev_hdxgf', label:'HDxGF', cl:'num', fmt: r=>r.ev_hdxgf.toFixed(1)}},
      {{k:'ev_hdxga', label:'HDxGA', cl:'num', fmt: r=>r.ev_hdxga.toFixed(1)}},
      {{k:'ev_hdgf', label:'HDGF', cl:'num', fmt: r=>r.ev_hdgf}},
      {{k:'ev_hdga', label:'HDGA', cl:'num', fmt: r=>r.ev_hdga}},
    ],
    defaultSort: 'ev_xgf_pct'
  }},
  all: {{
    cols: [
      {{k:'rank', label:'#',    cl:'',    fmt: (_,i)=>i+1, td:'rank-c'}},
      {{k:'team', label:'Team', cl:'',    fmt: r=>r.team,  td:'name-c'}},
      {{k:'gp',   label:'GP',  cl:'num', fmt: r=>r.gp}},
      {{k:'gf',   label:'GF',  cl:'num', fmt: r=>r.gf,  color: r=>gfCls(r.gf, r.ga)}},
      {{k:'ga',   label:'GA',  cl:'num', fmt: r=>r.ga,  color: r=>gaCls(r.gf, r.ga)}},
      {{k:'xgf_a', label:'xGF', cl:'num', fmt: r=>r.xgf_a.toFixed(1)}},
      {{k:'xga_a', label:'xGA', cl:'num', fmt: r=>r.xga_a.toFixed(1)}},
      {{k:'sog_f', label:'SOG For',  cl:'num', fmt: r=>r.sog_f}},
      {{k:'sog_a', label:'SOG Agnst',cl:'num', fmt: r=>r.sog_a}},
      {{k:'fo_pct', label:'FO%',  cl:'num', fmt: r=>r.fo_pct.toFixed(1)+'%', color: r=>pctCls(r.fo_pct)}},
    ],
    defaultSort: 'gf'
  }},
  pp: {{
    cols: [
      {{k:'rank',   label:'#',      cl:'',    fmt: (_,i)=>i+1, td:'rank-c'}},
      {{k:'team',   label:'Team',   cl:'',    fmt: r=>r.team,  td:'name-c'}},
      {{k:'gp',     label:'GP',     cl:'num', fmt: r=>r.gp}},
      {{k:'pp_gf',  label:'PP GF',  cl:'num', fmt: r=>r.pp_gf,  color: r=>r.pp_gf >= 50 ? 'c-green' : r.pp_gf < 30 ? 'c-red' : 'c-muted'}},
      {{k:'pp_xgf', label:'PP xGF', cl:'num', fmt: r=>r.pp_xgf.toFixed(1)}},
      {{k:'pp_toi', label:'PP TOI (min)', cl:'num', fmt: r=>r.pp_toi.toFixed(0)}},
    ],
    defaultSort: 'pp_gf'
  }},
  pk: {{
    cols: [
      {{k:'rank',   label:'#',      cl:'',    fmt: (_,i)=>i+1, td:'rank-c'}},
      {{k:'team',   label:'Team',   cl:'',    fmt: r=>r.team,  td:'name-c'}},
      {{k:'gp',     label:'GP',     cl:'num', fmt: r=>r.gp}},
      {{k:'pk_ga',  label:'PK GA',  cl:'num', fmt: r=>r.pk_ga,  color: r=>r.pk_ga <= 25 ? 'c-green' : r.pk_ga >= 45 ? 'c-red' : 'c-muted'}},
      {{k:'pk_xga', label:'PK xGA', cl:'num', fmt: r=>r.pk_xga.toFixed(1)}},
      {{k:'pk_toi', label:'PK TOI (min)', cl:'num', fmt: r=>r.pk_toi.toFixed(0)}},
    ],
    defaultSort: 'pk_ga'
  }},
}};

function pctCls(v) {{ return v >= 52 ? 'c-green' : v <= 48 ? 'c-red' : 'c-muted'; }}
function gfCls(gf, ga) {{ return gf > ga ? 'c-green' : gf < ga ? 'c-red' : 'c-muted'; }}
function gaCls(gf, ga) {{ return ga < gf ? 'c-green' : ga > gf ? 'c-red' : 'c-muted'; }}

function switchView(v) {{
  VIEW = v;
  SORT_COL = VIEWS[v].defaultSort;
  SORT_DIR = v === 'pk' ? 1 : -1;
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.view === v));
  render();
}}

function render() {{
  const view = VIEWS[VIEW];
  const cols = view.cols;
  const data = [...TEAMS].sort((a, b) => SORT_DIR * ((b[SORT_COL]||0) - (a[SORT_COL]||0)));

  const thead = document.getElementById('thead');
  thead.innerHTML = '<tr>' + cols.map(c =>
    `<th class="${{c.cl||''}}${{SORT_COL===c.k ? (SORT_DIR===-1?' sort-desc':' sort-asc') : ''}}" data-k="${{c.k}}">${{c.label}}</th>`
  ).join('') + '</tr>';
  thead.querySelectorAll('th').forEach(th => {{
    th.addEventListener('click', () => {{
      if (SORT_COL === th.dataset.k) SORT_DIR *= -1;
      else {{ SORT_COL = th.dataset.k; SORT_DIR = -1; }}
      render();
    }});
  }});

  const tbody = document.getElementById('tbody');
  tbody.innerHTML = data.map((r, i) =>
    '<tr>' + cols.map(c => {{
      const val = c.fmt(r, i);
      const cls = [c.cl||'', c.td||'', c.color ? c.color(r) : ''].filter(Boolean).join(' ');
      return `<td class="${{cls}}">${{val}}</td>`;
    }}).join('') + '</tr>'
  ).join('');

  document.getElementById('row-count').textContent = data.length + ' teams';
}}

render();
</script>
</body>
</html>"""


# ─── Lines HTML ───────────────────────────────────────────────────────────────

def _OLD_build_lines_html(lines_csv, season):
    df = pd.read_csv(lines_csv)
    df = df[df["situation"] == "5on5"].copy()

    records = []
    for _, r in df.iterrows():
        toi_min = w(r.get("icetime", 0)) / 60
        xgf = w(r.get("xGoalsFor", 0))
        xga = w(r.get("xGoalsAgainst", 0))
        xgf_pct = round(xgf / max(xgf + xga, 0.01) * 100, 1)
        cf  = w(r.get("shotAttemptsFor",     0), 0)
        ca  = w(r.get("shotAttemptsAgainst", 0), 0)
        cf_pct = round(cf / max(cf + ca, 1) * 100, 1)
        gf  = int(w(r.get("goalsFor",     0), 0))
        ga  = int(w(r.get("goalsAgainst", 0), 0))
        hdxgf = w(r.get("highDangerxGoalsFor",     0))
        hdxga = w(r.get("highDangerxGoalsAgainst", 0))
        hdxgf_pct = round(hdxgf / max(hdxgf + hdxga, 0.01) * 100, 1)
        sog_f = int(w(r.get("shotsOnGoalFor",     0), 0))
        sog_a = int(w(r.get("shotsOnGoalAgainst", 0), 0))
        records.append(dict(
            name=str(r.get("name", "")),
            team=str(r.get("team", "")),
            pos=str(r.get("position", "")),
            gp=int(w(r.get("games_played", 0), 0)),
            toi_min=round(toi_min, 1),
            xgf_pct=xgf_pct, cf_pct=cf_pct,
            gf=gf, ga=ga,
            xgf=round(xgf, 2), xga=round(xga, 2),
            hdxgf=round(hdxgf, 2), hdxga=round(hdxga, 2),
            hdxgf_pct=hdxgf_pct,
            sog_f=sog_f, sog_a=sog_a,
        ))

    data_json = json.dumps(records)
    teams = sorted(set(r["team"] for r in records))
    team_opts = '<option value="">All Teams</option>' + "".join(
        f'<option value="{t}">{t}</option>' for t in teams)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NHL WAR – Lines {season}</title>
{PAGE_STYLE}
</head>
<body>
<div class="header">
  <h1>⛸ Lines &amp; Pairings</h1>
  <p>{season} Season &nbsp;·&nbsp; 5-on-5</p>
</div>
<nav class="site-nav">
  <a href="NHL_WAR_Cards.html">Players</a>
  <a href="NHL_WAR_Teams.html">Teams</a>
  <a href="NHL_WAR_Lines.html" class="active">Lines</a>
</nav>

<div class="controls">
  <div class="tab active" data-pos="all"     onclick="setPos('all')">All</div>
  <div class="tab"        data-pos="line"    onclick="setPos('line')">Lines</div>
  <div class="tab"        data-pos="pairing" onclick="setPos('pairing')">Pairings</div>
  <select class="ctrl-select" id="team-sel" onchange="render()">
    {team_opts}
  </select>
  <label class="ctrl-label">
    Min TOI (min)
    <input class="ctrl-input" id="min-toi" type="number" value="100" min="0">
  </label>
  <span class="ctrl-right" id="row-count"></span>
</div>

<div class="wrap">
  <div class="lb-wrap">
    <table id="lb">
      <thead>
        <tr>
          <th data-k="rank"     >  #</th>
          <th data-k="name"     >  Line / Pairing</th>
          <th data-k="team"     >  Team</th>
          <th data-k="pos"      >  Type</th>
          <th data-k="gp"  class="num">GP</th>
          <th data-k="toi_min" class="num">TOI (min)</th>
          <th data-k="xgf_pct" class="num">xGF%</th>
          <th data-k="cf_pct"  class="num">CF%</th>
          <th data-k="gf"  class="num">GF</th>
          <th data-k="ga"  class="num">GA</th>
          <th data-k="xgf" class="num">xGF</th>
          <th data-k="xga" class="num">xGA</th>
          <th data-k="hdxgf_pct" class="num">HDxGF%</th>
          <th data-k="hdxgf"  class="num">HDxGF</th>
          <th data-k="hdxga"  class="num">HDxGA</th>
        </tr>
      </thead>
      <tbody id="tbody"></tbody>
    </table>
    <div class="lb-footer">Data: Natural Stat Trick &nbsp;·&nbsp; 5-on-5 only &nbsp;·&nbsp; Built {datetime.now().strftime("%b %d, %Y")}</div>
  </div>
</div>

<script>
const ALL_LINES = {data_json};
let SORT_COL = 'xgf_pct', SORT_DIR = -1, POS_FILTER = 'all';

function pctCls(v) {{ return v >= 52 ? 'c-green' : v <= 48 ? 'c-red' : 'c-muted'; }}
function gfCls(gf, ga) {{ return gf > ga ? 'c-green' : gf < ga ? 'c-red' : 'c-muted'; }}

function setPos(p) {{
  POS_FILTER = p;
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.pos === p));
  render();
}}

function render() {{
  const team   = document.getElementById('team-sel').value;
  const minToi = parseFloat(document.getElementById('min-toi').value) || 0;

  let data = ALL_LINES.filter(r => {{
    if (POS_FILTER !== 'all' && r.pos !== POS_FILTER) return false;
    if (team && r.team !== team) return false;
    if (r.toi_min < minToi) return false;
    return true;
  }});

  data.sort((a, b) => SORT_DIR * ((b[SORT_COL]||0) - (a[SORT_COL]||0)));
  document.getElementById('row-count').textContent = data.length + ' combos';

  const tbody = document.getElementById('tbody');
  tbody.innerHTML = data.map((r, i) => `
    <tr>
      <td class="rank-c">${{i+1}}</td>
      <td class="name-c">${{esc(r.name)}}</td>
      <td>${{esc(r.team)}}</td>
      <td><span class="badge badge-${{r.pos}}">${{r.pos === 'line' ? 'Line' : 'Pairing'}}</span></td>
      <td class="num">${{r.gp}}</td>
      <td class="num">${{r.toi_min.toFixed(0)}}</td>
      <td class="num ${{pctCls(r.xgf_pct)}}">${{r.xgf_pct.toFixed(1)}}%</td>
      <td class="num ${{pctCls(r.cf_pct)}}">${{r.cf_pct.toFixed(1)}}%</td>
      <td class="num ${{gfCls(r.gf, r.ga)}}">${{r.gf}}</td>
      <td class="num ${{r.ga <= r.gf ? 'c-green' : r.ga > r.gf ? 'c-red' : 'c-muted'}}">${{r.ga}}</td>
      <td class="num">${{r.xgf.toFixed(2)}}</td>
      <td class="num">${{r.xga.toFixed(2)}}</td>
      <td class="num ${{pctCls(r.hdxgf_pct)}}">${{r.hdxgf_pct.toFixed(1)}}%</td>
      <td class="num">${{r.hdxgf.toFixed(2)}}</td>
      <td class="num">${{r.hdxga.toFixed(2)}}</td>
    </tr>
  `).join('');

  // header sort indicators
  document.querySelectorAll('#lb th').forEach(th => {{
    th.classList.remove('sort-asc','sort-desc');
    if (th.dataset.k === SORT_COL)
      th.classList.add(SORT_DIR === -1 ? 'sort-desc' : 'sort-asc');
  }});
}}

document.querySelectorAll('#lb th').forEach(th => {{
  th.addEventListener('click', () => {{
    const k = th.dataset.k;
    if (!k || k === 'rank' || k === 'name' || k === 'team' || k === 'pos') return;
    if (SORT_COL === k) SORT_DIR *= -1;
    else {{ SORT_COL = k; SORT_DIR = -1; }}
    render();
  }});
}});

let toiTimer;
document.getElementById('min-toi').addEventListener('input', () => {{
  clearTimeout(toiTimer); toiTimer = setTimeout(render, 300);
}});

function esc(s) {{
  return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}}

render();
</script>
</body>
</html>"""



# ─── Teams HTML (with search + card) ─────────────────────────────────────────

def build_teams_html(teams_csv, season, nav_html="", team_history_json="{}"):
    df = pd.read_csv(teams_csv)
    ev   = df[df["situation"] == "5on5"].copy()
    pp   = df[df["situation"] == "5on4"].copy()
    pk   = df[df["situation"] == "4on5"].copy()
    all_ = df[df["situation"] == "all"].copy()

    records = []
    for team in sorted(all_["team"].unique()):
        def row(d, t=team): return d[d["team"] == t]
        ra = row(all_); re = row(ev); rp = row(pp); rk = row(pk)
        if ra.empty: continue
        ra=ra.iloc[0]; re=re.iloc[0] if not re.empty else None
        rp=rp.iloc[0] if not rp.empty else None; rk=rk.iloc[0] if not rk.empty else None

        gp    = w(ra["games_played"],0)
        gf    = w(ra["goalsFor"],0);         ga    = w(ra["goalsAgainst"],0)
        xgf_a = w(ra["xGoalsFor"]);          xga_a = w(ra["xGoalsAgainst"])
        sog_f = w(ra["shotsOnGoalFor"],0);   sog_a = w(ra["shotsOnGoalAgainst"],0)
        fo_w  = w(ra.get("faceOffsWonFor",0),0); fo_l = w(ra.get("faceOffsWonAgainst",0),0)
        fo_pct= round(fo_w/max(fo_w+fo_l,1)*100,1)
        hits_f= w(ra.get("hitsFor",0),0);   hits_a= w(ra.get("hitsAgainst",0),0)
        pk_f  = w(ra.get("penaltiesFor",0),0); pk_a= w(ra.get("penaltiesAgainst",0),0)

        ev_xgf_pct = round(w(re["xGoalsPercentage"])*100,1) if re is not None else 0
        ev_cf_pct  = round(w(re["corsiPercentage"])*100,1)  if re is not None else 0
        ev_ff_pct  = round(w(re["fenwickPercentage"])*100,1) if re is not None else 0
        ev_xgf = w(re["xGoalsFor"])     if re is not None else 0
        ev_xga = w(re["xGoalsAgainst"]) if re is not None else 0
        ev_gf  = int(w(re["goalsFor"],0))  if re is not None else 0
        ev_ga  = int(w(re["goalsAgainst"],0)) if re is not None else 0
        ev_hdxgf = w(re["highDangerxGoalsFor"])     if re is not None else 0
        ev_hdxga = w(re["highDangerxGoalsAgainst"]) if re is not None else 0
        ev_hdgf  = int(w(re["highDangerGoalsFor"],0))    if re is not None else 0
        ev_hdga  = int(w(re["highDangerGoalsAgainst"],0)) if re is not None else 0
        ev_sog_f = int(w(re["shotsOnGoalFor"],0)) if re is not None else 0
        ev_sog_a = int(w(re["shotsOnGoalAgainst"],0)) if re is not None else 0
        ev_toi_min = round(w(re.get("iceTime",0))/60,1) if re is not None else 0

        pp_gf  = int(w(rp["goalsFor"],0))   if rp is not None else 0
        pp_xgf = w(rp["xGoalsFor"])         if rp is not None else 0
        pp_toi = round(w(rp.get("iceTime",0))/60,1) if rp is not None else 0
        pp_hdxgf = w(rp["highDangerxGoalsFor"]) if rp is not None else 0
        pp_sog_f = int(w(rp["shotsOnGoalFor"],0)) if rp is not None else 0

        pk_ga  = int(w(rk["goalsAgainst"],0)) if rk is not None else 0
        pk_xga = w(rk["xGoalsAgainst"])       if rk is not None else 0
        pk_toi = round(w(rk.get("iceTime",0))/60,1) if rk is not None else 0
        pk_hdxga = w(rk["highDangerxGoalsAgainst"]) if rk is not None else 0
        pk_sog_a = int(w(rk["shotsOnGoalAgainst"],0)) if rk is not None else 0

        records.append(dict(
            team=str(team), gp=int(gp),
            gf=int(gf), ga=int(ga), xgf_a=xgf_a, xga_a=xga_a,
            sog_f=int(sog_f), sog_a=int(sog_a), fo_pct=fo_pct,
            hits_f=int(hits_f), hits_a=int(hits_a),
            pk_drawn=int(pk_f), pk_taken=int(pk_a),
            ev_xgf_pct=ev_xgf_pct, ev_cf_pct=ev_cf_pct, ev_ff_pct=ev_ff_pct,
            ev_xgf=ev_xgf, ev_xga=ev_xga, ev_gf=ev_gf, ev_ga=ev_ga,
            ev_hdxgf=ev_hdxgf, ev_hdxga=ev_hdxga, ev_hdgf=ev_hdgf, ev_hdga=ev_hdga,
            ev_sog_f=ev_sog_f, ev_sog_a=ev_sog_a, ev_toi_min=ev_toi_min,
            pp_gf=pp_gf, pp_xgf=pp_xgf, pp_toi=pp_toi,
            pp_hdxgf=pp_hdxgf, pp_sog_f=pp_sog_f,
            pk_ga=pk_ga, pk_xga=pk_xga, pk_toi=pk_toi,
            pk_hdxga=pk_hdxga, pk_sog_a=pk_sog_a,
        ))

    data_json = json.dumps(records)
    team_hist_js = team_history_json  # already a JSON string
    ts = datetime.now().strftime("%b %d, %Y")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NHL WAR – Teams {season}</title>
{PAGE_STYLE}
<style>
  /* ── Search ── */
  .search-wrap {{ max-width:520px; margin:0 auto 20px; position:relative; }}
  .search-wrap input {{
    width:100%; padding:12px 20px; border-radius:50px;
    border:2px solid var(--border); background:var(--surf);
    color:var(--text); font-size:1rem; outline:none; transition:border-color 0.2s;
  }}
  .search-wrap input:focus {{ border-color:var(--accent); }}
  .ac-box {{
    position:absolute; top:calc(100% + 6px); left:0; right:0;
    background:var(--surf); border:1px solid var(--border); border-radius:var(--r);
    z-index:100; box-shadow:0 8px 32px rgba(0,0,0,0.5); display:none;
  }}
  .ac-item {{ padding:10px 18px; cursor:pointer; font-size:0.9rem; font-weight:600; transition:background 0.12s; }}
  .ac-item:hover {{ background:var(--surf2); color:var(--accent); }}

  /* ── Leaderboard scrollable table ── */
  .lb-wrap {{ overflow-x:auto !important; }}
  .lb-table {{ border-collapse:collapse; font-size:0.84rem; white-space:nowrap; width:100%; }}
  .lb-table th {{ background:var(--surf2); position:sticky; top:0; z-index:1;
    font-size:0.75rem; font-weight:700; letter-spacing:0.5px; text-transform:uppercase;
    color:var(--text2); cursor:pointer; user-select:none; padding:8px 10px;
    border-bottom:2px solid var(--border); }}
  .lb-table th:hover {{ color:var(--text); }}
  .lb-table td {{ padding:7px 10px; border-bottom:1px solid var(--border); }}
  .lb-table tbody tr:hover td {{ background:#0e1d30; }}
  .sc1 {{ position:sticky !important; z-index:2 !important; background:var(--surf); left:0; min-width:36px; max-width:36px; }}
  .sc2 {{ position:sticky !important; z-index:2 !important; background:var(--surf); left:36px; min-width:70px; max-width:70px; box-shadow:3px 0 8px rgba(0,0,0,0.4); }}
  .lb-table thead .sc1, .lb-table thead .sc2 {{ background:var(--surf2) !important; z-index:3 !important; }}
  .lb-table tbody tr:hover .sc1, .lb-table tbody tr:hover .sc2 {{ background:#0e1d30 !important; }}
  .grp-l {{ border-left:2px solid var(--border) !important; }}

  /* ── Card Area ── */
  #card-area {{ max-width:920px; margin:0 auto; display:none; }}
  .back-btn {{
    display:inline-flex; align-items:center; gap:6px;
    margin-bottom:16px; padding:8px 16px;
    background:var(--surf); border:1px solid var(--border);
    border-radius:20px; color:var(--text2); font-size:0.85rem;
    cursor:pointer; transition:all 0.15s;
  }}
  .back-btn:hover {{ border-color:var(--accent); color:var(--accent); }}

  /* ── Team Card ── */
  .player-card {{
    background:var(--surf); border:1px solid var(--border);
    border-radius:var(--r); overflow:hidden;
  }}
  .card-header {{
    background:linear-gradient(135deg,var(--surf2),#0d1b2a);
    padding:22px 26px; display:flex; align-items:center;
    gap:18px; border-bottom:1px solid var(--border);
  }}
  .avatar {{
    width:68px; height:68px; border-radius:50%;
    background:linear-gradient(135deg,#1a3a5c,#00b4d8);
    display:flex; align-items:center; justify-content:center;
    font-size:1.3rem; font-weight:900; color:#fff; flex-shrink:0; letter-spacing:1px;
  }}
  .player-info h2 {{ font-size:1.55rem; font-weight:800; line-height:1.1; }}
  .player-meta {{ display:flex; gap:8px; margin-top:6px; flex-wrap:wrap; align-items:center; }}
  .meta-b {{
    font-size:0.77rem; font-weight:600; padding:3px 10px;
    border-radius:20px; background:var(--surf); border:1px solid var(--border); color:var(--text2);
  }}
  .war-summary {{
    padding:18px 26px; display:flex; gap:12px; flex-wrap:wrap;
    border-bottom:1px solid var(--border);
    background:linear-gradient(to right,rgba(0,180,216,0.05),transparent);
  }}
  .war-big {{
    flex:1; min-width:130px; text-align:center;
    padding:12px 14px; background:var(--surf2);
    border-radius:10px; border:1px solid var(--border);
  }}
  .war-big.highlight {{ border-color:var(--accent); background:rgba(0,180,216,0.08); }}
  .war-big .lbl {{
    font-size:0.68rem; font-weight:700; letter-spacing:1px;
    color:var(--text2); text-transform:uppercase; margin-bottom:4px;
  }}
  .war-big .val {{ font-size:1.9rem; font-weight:900; line-height:1; }}
  .war-big .sub {{ font-size:0.74rem; color:var(--text2); margin-top:4px; }}
  .card-body {{ padding:22px 26px; display:flex; flex-direction:column; gap:22px; }}
  .section-title {{
    font-size:0.72rem; font-weight:700; letter-spacing:1.5px;
    text-transform:uppercase; color:var(--muted); margin-bottom:12px;
    display:flex; align-items:center; gap:8px;
  }}
  .section-title::after {{ content:''; flex:1; height:1px; background:var(--border); }}
  .stat-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(110px,1fr)); gap:10px; }}
  .stat-box {{
    background:var(--surf2); border:1px solid var(--border);
    border-radius:8px; padding:10px 12px; text-align:center;
  }}
  .stat-box .sv {{ font-size:1.1rem; font-weight:800; }}
  .stat-box .sl {{ font-size:0.68rem; color:var(--text2); text-transform:uppercase; letter-spacing:0.5px; margin-top:2px; }}
  .bar-row {{ margin-bottom:12px; }}
  .bar-top {{ display:flex; justify-content:space-between; align-items:baseline; margin-bottom:4px; font-size:0.84rem; }}
  .bar-lbl {{ font-weight:600; }}
  .bar-val {{ font-weight:700; }}
  .bar-track {{ height:7px; background:var(--surf2); border-radius:4px; overflow:hidden; }}
  .bar-fill  {{ height:100%; border-radius:4px; }}
  .bar-blue  {{ background:linear-gradient(90deg,#0077b6,#00b4d8); }}
  .bar-green {{ background:linear-gradient(90deg,#2dc653,#00b4d8); }}
  .bar-red   {{ background:linear-gradient(90deg,#e63946,#c1121f); }}

  @media (max-width:640px) {{
    .card-header {{ flex-direction:column; align-items:flex-start; }}
    .war-big .val {{ font-size:1.5rem; }}
  }}
</style>
</head>
<body>
<div class="header">
  <h1>🏒 NHL Team Stats</h1>
  <p>{season} Season · All Situations &amp; 5-on-5</p>
</div>
{nav_html}

<div id="leaderboard-area">
  <div class="search-wrap">
    <input id="search" type="text" placeholder="Search a team… (e.g. TOR, BOS, EDM)" autocomplete="off">
    <div class="ac-box" id="ac-box"></div>
  </div>

  <div class="controls">
    <span class="ctrl-right" id="row-count"></span>
  </div>

  <div class="wrap">
    <div class="lb-wrap">
      <table id="lb" class="lb-table">
        <thead>
          <tr>
            <th class="sc1" data-k="rank">#</th>
            <th class="sc2" data-k="team">Team</th>
            <th class="num grp-l" data-k="gp">GP</th>
            <th class="num grp-l" data-k="ev_xgf_pct" title="5-on-5 Expected Goals For %">xGF%</th>
            <th class="num" data-k="ev_cf_pct" title="5-on-5 Corsi For %">CF%</th>
            <th class="num" data-k="ev_ff_pct" title="5-on-5 Fenwick For %">FF%</th>
            <th class="num" data-k="ev_gf" title="5-on-5 Goals For">EV&nbsp;GF</th>
            <th class="num" data-k="ev_ga" title="5-on-5 Goals Against">EV&nbsp;GA</th>
            <th class="num" data-k="ev_xgf" title="5-on-5 Expected Goals For">EV&nbsp;xGF</th>
            <th class="num" data-k="ev_xga" title="5-on-5 Expected Goals Against">EV&nbsp;xGA</th>
            <th class="num" data-k="ev_hdxgf" title="5-on-5 High Danger xGoals For">HDxGF</th>
            <th class="num" data-k="ev_hdxga" title="5-on-5 High Danger xGoals Against">HDxGA</th>
            <th class="num" data-k="ev_hdgf" title="5-on-5 High Danger Goals For">HDGF</th>
            <th class="num" data-k="ev_hdga" title="5-on-5 High Danger Goals Against">HDGA</th>
            <th class="num" data-k="ev_toi_min" title="5-on-5 Time on Ice (minutes)">EV&nbsp;TOI</th>
            <th class="num grp-l" data-k="pp_gf" title="Power Play Goals For">PP&nbsp;GF</th>
            <th class="num" data-k="pp_xgf" title="Power Play Expected Goals For">PP&nbsp;xGF</th>
            <th class="num" data-k="pp_hdxgf" title="Power Play High Danger xGoals For">PP&nbsp;HDxGF</th>
            <th class="num" data-k="pp_sog_f" title="Power Play Shots on Goal For">PP&nbsp;SOG</th>
            <th class="num" data-k="pp_toi" title="Power Play Time on Ice (minutes)">PP&nbsp;TOI</th>
            <th class="num grp-l" data-k="pk_ga" title="Penalty Kill Goals Against">PK&nbsp;GA</th>
            <th class="num" data-k="pk_xga" title="Penalty Kill Expected Goals Against">PK&nbsp;xGA</th>
            <th class="num" data-k="pk_hdxga" title="Penalty Kill High Danger xGoals Against">PK&nbsp;HDxGA</th>
            <th class="num" data-k="pk_sog_a" title="Penalty Kill Shots on Goal Against">PK&nbsp;SOG</th>
            <th class="num" data-k="pk_toi" title="Penalty Kill Time on Ice (minutes)">PK&nbsp;TOI</th>
            <th class="num grp-l" data-k="gf" title="All-situations Goals For">GF</th>
            <th class="num" data-k="ga" title="All-situations Goals Against">GA</th>
            <th class="num" data-k="xgf_a" title="All-situations Expected Goals For">xGF</th>
            <th class="num" data-k="xga_a" title="All-situations Expected Goals Against">xGA</th>
            <th class="num" data-k="sog_f" title="All-situations Shots on Goal For">SOG&nbsp;For</th>
            <th class="num" data-k="sog_a" title="All-situations Shots on Goal Against">SOG&nbsp;Agnst</th>
            <th class="num" data-k="fo_pct" title="Faceoff Win %">FO%</th>
            <th class="num" data-k="hits_f" title="Hits For">Hits</th>
            <th class="num" data-k="pk_drawn" title="Penalties Drawn">Pens&nbsp;Drawn</th>
            <th class="num" data-k="pk_taken" title="Penalties Taken">Pens&nbsp;Taken</th>
          </tr>
        </thead>
        <tbody id="tbody"></tbody>
      </table>
      <div class="lb-footer">Data: Natural Stat Trick · Built {ts}</div>
    </div>
  </div>
</div>

<!-- Team Card (full-page, hidden until selected) -->
<div id="card-area">
  <div class="back-btn" id="back-btn">← Back to Teams</div>
  <div id="card-inner"></div>
</div>

<script>
const TEAMS = {data_json};
const TEAM_HIST = {team_hist_js};
const SEASON = '{season}';
let SORT_COL = 'ev_xgf_pct', SORT_DIR = 1;
let currentTeam = null;

function pCls(v)       {{ return v>=52?'c-green':v<=48?'c-red':'c-muted'; }}
function gCls(gf,ga)   {{ return gf>ga?'c-green':gf<ga?'c-red':'c-muted'; }}
function bCls(gf,ga)   {{ return ga<gf?'c-green':ga>gf?'c-red':'c-muted'; }}
function pBar(v,lo=45,hi=55) {{ return v>=hi?'bar-green':v<=lo?'bar-red':'bar-blue'; }}
function xgfHex(v)     {{ return v>=52?'#2dc653':v<=48?'#e63946':'#e2e8f0'; }}
function diffHex(d)    {{ return d>0?'#2dc653':d<0?'#e63946':'#e2e8f0'; }}

function render() {{
  const data = [...TEAMS].sort((a,b) => {{
    const av = a[SORT_COL] ?? -Infinity;
    const bv = b[SORT_COL] ?? -Infinity;
    return SORT_DIR * (bv - av);
  }});
  document.getElementById('row-count').textContent = data.length + ' teams';
  document.querySelectorAll('#lb th').forEach(th => {{
    th.classList.remove('sort-asc','sort-desc');
    if (th.dataset.k === SORT_COL) th.classList.add(SORT_DIR === -1 ? 'sort-desc' : 'sort-asc');
  }});
  const n  = (v, d=1) => v != null ? (+v).toFixed(d) : '—';
  const tbody = document.getElementById('tbody');
  tbody.innerHTML = data.map((r,i) => `<tr>
    <td class="sc1 rank-c">${{i+1}}</td>
    <td class="sc2 name-c">${{r.team}}</td>
    <td class="num grp-l">${{r.gp}}</td>
    <td class="num grp-l ${{pCls(r.ev_xgf_pct)}}">${{n(r.ev_xgf_pct)}}%</td>
    <td class="num ${{pCls(r.ev_cf_pct)}}">${{n(r.ev_cf_pct)}}%</td>
    <td class="num ${{pCls(r.ev_ff_pct)}}">${{n(r.ev_ff_pct)}}%</td>
    <td class="num ${{gCls(r.ev_gf,r.ev_ga)}}">${{r.ev_gf}}</td>
    <td class="num ${{bCls(r.ev_gf,r.ev_ga)}}">${{r.ev_ga}}</td>
    <td class="num">${{n(r.ev_xgf)}}</td>
    <td class="num">${{n(r.ev_xga)}}</td>
    <td class="num">${{n(r.ev_hdxgf)}}</td>
    <td class="num">${{n(r.ev_hdxga)}}</td>
    <td class="num">${{r.ev_hdgf}}</td>
    <td class="num">${{r.ev_hdga}}</td>
    <td class="num">${{n(r.ev_toi_min, 0)}} min</td>
    <td class="num grp-l ${{r.pp_gf>=50?'c-green':r.pp_gf<30?'c-red':'c-muted'}}">${{r.pp_gf}}</td>
    <td class="num">${{n(r.pp_xgf)}}</td>
    <td class="num">${{n(r.pp_hdxgf)}}</td>
    <td class="num">${{r.pp_sog_f}}</td>
    <td class="num">${{n(r.pp_toi, 0)}} min</td>
    <td class="num grp-l ${{r.pk_ga<=25?'c-green':r.pk_ga>=45?'c-red':'c-muted'}}">${{r.pk_ga}}</td>
    <td class="num">${{n(r.pk_xga)}}</td>
    <td class="num">${{n(r.pk_hdxga)}}</td>
    <td class="num">${{r.pk_sog_a}}</td>
    <td class="num">${{n(r.pk_toi, 0)}} min</td>
    <td class="num grp-l ${{gCls(r.gf,r.ga)}}">${{r.gf}}</td>
    <td class="num ${{bCls(r.gf,r.ga)}}">${{r.ga}}</td>
    <td class="num">${{n(r.xgf_a)}}</td>
    <td class="num">${{n(r.xga_a)}}</td>
    <td class="num">${{r.sog_f}}</td>
    <td class="num">${{r.sog_a}}</td>
    <td class="num ${{pCls(r.fo_pct)}}">${{n(r.fo_pct)}}%</td>
    <td class="num">${{r.hits_f}}</td>
    <td class="num">${{r.pk_drawn}}</td>
    <td class="num">${{r.pk_taken}}</td>
  </tr>`).join('');
  tbody.querySelectorAll('tr').forEach((tr,i) => {{
    tr.style.cursor = 'pointer';
    tr.addEventListener('click', () => showCard(data[i]));
  }});
}}

document.querySelectorAll('#lb th').forEach(th => {{
  th.addEventListener('click', () => {{
    const k = th.dataset.k;
    if (!k || k === 'rank') return;
    if (SORT_COL === k) SORT_DIR *= -1; else {{ SORT_COL = k; SORT_DIR = -1; }}
    render();
  }});
}});

/* ── Nav filtering ── */
const navSel = document.querySelector('.site-nav select');
let origOptions = [];
if (navSel) {{
  origOptions = [...navSel.options].map(o => ({{v:o.value, t:o.text, s:o.selected}}));
  navSel.onchange = function() {{ location.href = this.value; }};
}}
function updateNavForTeam(r) {{
  if (!navSel || !origOptions.length) return;
  const hist = TEAM_HIST[r.team] || [];
  const teamSeasons = new Set(hist.map(e => e.s));
  navSel.innerHTML = '';
  origOptions.forEach(opt => {{
    if (!teamSeasons.has(opt.t)) return;
    const o = document.createElement('option');
    o.value = opt.v + '#t=' + encodeURIComponent(r.team);
    o.text = opt.t; o.selected = opt.t === SEASON;
    navSel.appendChild(o);
  }});
}}
function restoreNav() {{
  if (!navSel || !origOptions.length) return;
  navSel.innerHTML = '';
  origOptions.forEach(opt => {{
    const o = document.createElement('option');
    o.value = opt.v; o.text = opt.t; o.selected = opt.s;
    navSel.appendChild(o);
  }});
}}

/* ── Card helpers ── */
function sb(lbl,val,cls='') {{
  return `<div class="stat-box"><div class="sv ${{cls}}">${{val}}</div><div class="sl">${{lbl}}</div></div>`;
}}
function barRow(lbl,val,pct,barCls) {{
  pct=Math.max(0,Math.min(100,pct));
  return `<div class="bar-row"><div class="bar-top"><span class="bar-lbl">${{lbl}}</span><span class="bar-val">${{val}}</span></div><div class="bar-track"><div class="bar-fill ${{barCls}}" style="width:${{pct}}%"></div></div></div>`;
}}

/* ── Team Card renderer ── */
function renderTeamCard(r) {{
  const diff=r.gf-r.ga; const dStr=(diff>=0?'+':'')+diff;
  const evDiff=r.ev_gf-r.ev_ga; const evDStr=(evDiff>=0?'+':'')+evDiff;
  return `
  <div class="player-card">
    <div class="card-header">
      <div class="avatar">${{r.team}}</div>
      <div class="player-info">
        <h2>${{r.team}}</h2>
        <div class="player-meta">
          <span class="meta-b">${{SEASON}}</span>
          <span class="meta-b">${{r.gp}} GP</span>
          <span class="meta-b">${{r.gf}} GF – ${{r.ga}} GA &nbsp;<span style="color:${{diffHex(diff)}}">${{dStr}}</span></span>
        </div>
      </div>
    </div>

    <div class="war-summary">
      <div class="war-big highlight">
        <div class="lbl">xGF%</div>
        <div class="val" style="color:${{xgfHex(r.ev_xgf_pct)}}">${{r.ev_xgf_pct.toFixed(1)}}%</div>
        <div class="sub">5-on-5</div>
      </div>
      <div class="war-big">
        <div class="lbl">EV Goal Diff</div>
        <div class="val" style="color:${{diffHex(evDiff)}}">${{evDStr}}</div>
        <div class="sub">${{r.ev_gf}} GF – ${{r.ev_ga}} GA</div>
      </div>
      <div class="war-big">
        <div class="lbl">PP GF</div>
        <div class="val" style="color:${{r.pp_gf>=50?'#2dc653':r.pp_gf<30?'#e63946':'#e2e8f0'}}">${{r.pp_gf}}</div>
        <div class="sub">Power Play</div>
      </div>
      <div class="war-big">
        <div class="lbl">PK GA</div>
        <div class="val" style="color:${{r.pk_ga<=25?'#2dc653':r.pk_ga>=45?'#e63946':'#e2e8f0'}}">${{r.pk_ga}}</div>
        <div class="sub">Penalty Kill</div>
      </div>
    </div>

    <div class="card-body">
      <div>
        <div class="section-title">5-on-5 (Even Strength)</div>
        ${{barRow('xGF%',r.ev_xgf_pct.toFixed(1)+'%',r.ev_xgf_pct*2,pBar(r.ev_xgf_pct))}}
        ${{barRow('CF%', r.ev_cf_pct.toFixed(1)+'%', r.ev_cf_pct*2, pBar(r.ev_cf_pct))}}
        ${{barRow('FF%', r.ev_ff_pct.toFixed(1)+'%', r.ev_ff_pct*2, pBar(r.ev_ff_pct))}}
        <div class="stat-grid" style="margin-top:14px">
          ${{sb('EV GF',  r.ev_gf,  r.ev_gf>r.ev_ga?'c-green':r.ev_gf<r.ev_ga?'c-red':'')}}
          ${{sb('EV GA',  r.ev_ga,  r.ev_ga<r.ev_gf?'c-green':r.ev_ga>r.ev_gf?'c-red':'')}}
          ${{sb('xGF',   r.ev_xgf.toFixed(1))}}
          ${{sb('xGA',   r.ev_xga.toFixed(1))}}
          ${{sb('HDxGF', r.ev_hdxgf.toFixed(1))}}
          ${{sb('HDxGA', r.ev_hdxga.toFixed(1))}}
          ${{sb('HDGF',  r.ev_hdgf)}}
          ${{sb('HDGA',  r.ev_hdga)}}
          ${{sb('TOI',   r.ev_toi_min.toFixed(0)+' min')}}
        </div>
      </div>
      <div>
        <div class="section-title">Power Play</div>
        <div class="stat-grid">
          ${{sb('PP GF',    r.pp_gf,  r.pp_gf>=50?'c-green':r.pp_gf<30?'c-red':'')}}
          ${{sb('PP xGF',   r.pp_xgf.toFixed(1))}}
          ${{sb('PP HDxGF', r.pp_hdxgf.toFixed(1))}}
          ${{sb('PP SOG',   r.pp_sog_f)}}
          ${{sb('PP TOI',   r.pp_toi.toFixed(0)+' min')}}
        </div>
      </div>
      <div>
        <div class="section-title">Penalty Kill</div>
        <div class="stat-grid">
          ${{sb('PK GA',    r.pk_ga,  r.pk_ga<=25?'c-green':r.pk_ga>=45?'c-red':'')}}
          ${{sb('PK xGA',   r.pk_xga.toFixed(1))}}
          ${{sb('PK HDxGA', r.pk_hdxga.toFixed(1))}}
          ${{sb('PK SOG',   r.pk_sog_a)}}
          ${{sb('PK TOI',   r.pk_toi.toFixed(0)+' min')}}
        </div>
      </div>
      <div>
        <div class="section-title">Overall</div>
        <div class="stat-grid">
          ${{sb('GF',         r.gf,  r.gf>r.ga?'c-green':r.gf<r.ga?'c-red':'')}}
          ${{sb('GA',         r.ga,  r.ga<r.gf?'c-green':r.ga>r.gf?'c-red':'')}}
          ${{sb('xGF',        r.xgf_a.toFixed(1))}}
          ${{sb('xGA',        r.xga_a.toFixed(1))}}
          ${{sb('SOG For',    r.sog_f)}}
          ${{sb('SOG Agnst',  r.sog_a)}}
          ${{sb('FO%',        r.fo_pct.toFixed(1)+'%', pCls(r.fo_pct))}}
          ${{sb('Hits For',   r.hits_f)}}
          ${{sb('Pens Drawn', r.pk_drawn)}}
          ${{sb('Pens Taken', r.pk_taken)}}
        </div>
      </div>
      ${{renderXGFChart(r.team)}}
    </div>
  </div>`;
}}

function showCard(r) {{
  currentTeam = r;
  history.replaceState(null, '', '#t=' + encodeURIComponent(r.team));
  updateNavForTeam(r);
  document.getElementById('leaderboard-area').style.display = 'none';
  document.getElementById('card-area').style.display = 'block';
  document.getElementById('card-inner').innerHTML = renderTeamCard(r);
  window.scrollTo({{top:0, behavior:'smooth'}});
}}

document.getElementById('back-btn').addEventListener('click', () => {{
  currentTeam = null;
  history.replaceState(null, '', window.location.pathname + window.location.search);
  restoreNav();
  document.getElementById('card-area').style.display = 'none';
  document.getElementById('leaderboard-area').style.display = 'block';
  searchEl.value = ''; acBox.style.display = 'none';
}});

function renderXGFChart(teamName) {{
  const hist = TEAM_HIST[teamName];
  if (!hist || hist.length < 2) return '';
  const W=480,H=110,PL=38,PR=8,PT=12,PB=28;
  const plotW=W-PL-PR, plotH=H-PT-PB;
  const vals=hist.map(e=>e.x);
  const minV=Math.min(...vals,44), maxV=Math.max(...vals,56);
  const range=maxV-minV||1;
  const step=plotW/hist.length;
  const barW=Math.max(4,Math.floor(step)-2);
  const fiftyY=PT+plotH*((maxV-50)/range);
  const cid='xtt_'+teamName.replace(/\W/g,'_');
  let bars='',labels='';
  hist.forEach((e,i)=>{{
    const cx=PL+i*step+step/2;
    const isCur=e.s===SEASON;
    const valH=(e.x-50)/range*plotH;
    const bh=Math.abs(valH);
    const y=valH>=0?fiftyY-valH:fiftyY;
    const col=isCur?'#00b4d8':e.x>=50?'#2dc653':'#e63946';
    const tip=e.s+': '+e.x.toFixed(1)+'% xGF';
    bars+=`<rect x="${{(cx-step/2+1).toFixed(1)}}" y="${{PT}}" width="${{(step-2).toFixed(1)}}" height="${{plotH}}" fill="transparent" style="cursor:default" onmouseover="(function(){{var t=document.getElementById('${{cid}}');if(t){{t.style.display='block';t.textContent='${{tip}}'}}}})()" onmouseout="(function(){{var t=document.getElementById('${{cid}}');if(t)t.style.display='none'}})()"/>`;
    bars+=`<rect x="${{(cx-barW/2).toFixed(1)}}" y="${{y.toFixed(1)}}" width="${{barW}}" height="${{Math.max(1,bh).toFixed(1)}}" fill="${{col}}" rx="1" opacity="${{isCur?1:0.65}}" pointer-events="none"/>`;
    if(isCur||i%Math.max(1,Math.round(hist.length/6))===0){{
      const yr=e.s.slice(2,4)+'-'+e.s.slice(7);
      labels+=`<text x="${{cx.toFixed(1)}}" y="${{(H-4).toFixed(1)}}" text-anchor="middle" font-size="8" fill="${{isCur?'#00b4d8':'#64748b'}}" pointer-events="none">${{yr}}</text>`;
    }}
  }});
  const zLine=`<line x1="${{PL}}" y1="${{fiftyY.toFixed(1)}}" x2="${{W-PR}}" y2="${{fiftyY.toFixed(1)}}" stroke="#2a3a50" stroke-width="1" stroke-dasharray="3,3" pointer-events="none"/>`;
  const lbl50=`<text x="${{(PL-3)}}" y="${{(fiftyY+3).toFixed(1)}}" text-anchor="end" font-size="8" fill="#64748b" pointer-events="none">50%</text>`;
  const lblMax=`<text x="${{(PL-3)}}" y="${{(PT+8)}}" text-anchor="end" font-size="8" fill="#64748b" pointer-events="none">${{maxV.toFixed(0)}}%</text>`;
  return `<div class="section-title" style="margin-top:4px">xGF% History</div>
<div style="position:relative;margin-top:8px">
  <div id="${{cid}}" style="display:none;position:absolute;top:0;left:50%;transform:translateX(-50%);background:#1e2a3a;border:1px solid #2a3a50;border-radius:6px;padding:3px 10px;font-size:0.76rem;color:#e2e8f0;white-space:nowrap;z-index:10;pointer-events:none"></div>
  <div style="overflow-x:auto"><svg width="100%" viewBox="0 0 ${{W}} ${{H}}" style="max-width:${{W}}px;display:block">${{zLine}}${{lbl50}}${{lblMax}}${{bars}}${{labels}}</svg></div>
</div>`;
}}

/* ── Search ── */
const searchEl=document.getElementById('search');
const acBox=document.getElementById('ac-box');
searchEl.addEventListener('input',()=>{{
  const q=searchEl.value.trim().toUpperCase();
  if(!q){{acBox.style.display='none';return;}}
  const hits=TEAMS.filter(r=>r.team.toUpperCase().includes(q));
  if(!hits.length){{acBox.style.display='none';return;}}
  acBox.innerHTML=hits.map(r=>`<div class="ac-item" data-t="${{r.team}}">${{r.team}}</div>`).join('');
  acBox.style.display='block';
  acBox.querySelectorAll('.ac-item').forEach(el=>el.addEventListener('click',()=>{{
    const t=el.dataset.t; const team=TEAMS.find(r=>r.team===t);
    if(team){{showCard(team); searchEl.value=''; acBox.style.display='none';}}
  }}));
}});
document.addEventListener('click',e=>{{if(!e.target.closest('.search-wrap'))acBox.style.display='none';}});

render();

/* ── Auto-open from URL hash ── */
(function() {{
  const hash = location.hash;
  if (!hash.startsWith('#t=')) return;
  const tname = decodeURIComponent(hash.slice(3));
  const team = TEAMS.find(r => r.team === tname);
  if (team) showCard(team);
}})();
</script>
</body>
</html>"""


# ─── Lines HTML (with search + card) ─────────────────────────────────────────

def build_lines_html(lines_csv, season, nav_html=""):
    df = pd.read_csv(lines_csv)
    df = df[df["situation"] == "5on5"].copy()

    records = []
    for _, r in df.iterrows():
        toi_min = w(r.get("icetime", 0)) / 60
        xgf = w(r.get("xGoalsFor", 0));  xga = w(r.get("xGoalsAgainst", 0))
        xgf_pct = round(xgf / max(xgf + xga, 0.01) * 100, 1)
        cf  = w(r.get("shotAttemptsFor",     0), 0)
        ca  = w(r.get("shotAttemptsAgainst", 0), 0)
        cf_pct = round(cf / max(cf + ca, 1) * 100, 1)
        uf  = w(r.get("unblockedShotAttemptsFor",     0), 0)
        ua  = w(r.get("unblockedShotAttemptsAgainst", 0), 0)
        ff_pct = round(uf / max(uf + ua, 1) * 100, 1)
        gf  = int(w(r.get("goalsFor",     0), 0))
        ga  = int(w(r.get("goalsAgainst", 0), 0))
        hdxgf = w(r.get("highDangerxGoalsFor",     0))
        hdxga = w(r.get("highDangerxGoalsAgainst", 0))
        hdxgf_pct = round(hdxgf / max(hdxgf + hdxga, 0.01) * 100, 1)
        hdgf = int(w(r.get("highDangerGoalsFor",     0), 0))
        hdga = int(w(r.get("highDangerGoalsAgainst", 0), 0))
        sog_f = int(w(r.get("shotsOnGoalFor",     0), 0))
        sog_a = int(w(r.get("shotsOnGoalAgainst", 0), 0))
        rbd_f = int(w(r.get("reboundsFor",     0), 0))
        rbd_a = int(w(r.get("reboundsAgainst", 0), 0))
        fo_w  = int(w(r.get("faceOffsWonFor",     0), 0))
        fo_l  = int(w(r.get("faceOffsWonAgainst", 0), 0))
        fo_pct= round(fo_w / max(fo_w + fo_l, 1) * 100, 1) if (fo_w + fo_l) > 0 else None
        giveaways = int(w(r.get("giveawaysFor", 0), 0))
        takeaways = int(w(r.get("takeawaysFor", 0), 0))
        records.append(dict(
            name=str(r.get("name","")), team=str(r.get("team","")),
            pos=str(r.get("position","")),
            gp=int(w(r.get("games_played",0),0)),
            toi_min=round(toi_min,1),
            xgf_pct=xgf_pct, cf_pct=cf_pct, ff_pct=ff_pct,
            gf=gf, ga=ga, xgf=round(xgf,2), xga=round(xga,2),
            hdxgf=round(hdxgf,2), hdxga=round(hdxga,2),
            hdxgf_pct=hdxgf_pct, hdgf=hdgf, hdga=hdga,
            sog_f=sog_f, sog_a=sog_a, rbd_f=rbd_f, rbd_a=rbd_a,
            fo_pct=fo_pct, giveaways=giveaways, takeaways=takeaways,
        ))

    data_json = json.dumps(records)
    teams = sorted(set(r["team"] for r in records))
    team_opts = '<option value="">All Teams</option>' + "".join(
        f'<option value="{t}">{t}</option>' for t in teams)
    ts = datetime.now().strftime("%b %d, %Y")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NHL WAR – Lines {season}</title>
{PAGE_STYLE}
<style>
  .search-wrap {{ max-width:520px; margin:0 auto 20px; position:relative; }}
  .search-wrap input {{
    width:100%; padding:12px 20px; border-radius:50px;
    border:2px solid var(--border); background:var(--surf);
    color:var(--text); font-size:1rem; outline:none; transition:border-color 0.2s;
  }}
  .search-wrap input:focus {{ border-color:var(--accent); }}
  .ac-box {{
    position:absolute; top:calc(100% + 6px); left:0; right:0;
    background:var(--surf); border:1px solid var(--border); border-radius:var(--r);
    z-index:100; box-shadow:0 8px 32px rgba(0,0,0,0.5); display:none; max-height:240px; overflow-y:auto;
  }}
  .ac-item {{ padding:10px 18px; cursor:pointer; font-size:0.88rem; display:flex; justify-content:space-between; align-items:center; transition:background 0.12s; }}
  .ac-item:hover {{ background:var(--surf2); }}
  .ac-name {{ font-weight:600; }}
  .ac-meta {{ color:var(--text2); font-size:0.8rem; }}
  /* ── Leaderboard scrollable table ── */
  .lb-wrap {{ overflow-x:auto !important; }}
  .lb-table {{ border-collapse:collapse; font-size:0.84rem; white-space:nowrap; width:100%; }}
  .lb-table th {{ background:var(--surf2); position:sticky; top:0; z-index:1;
    font-size:0.75rem; font-weight:700; letter-spacing:0.5px; text-transform:uppercase;
    color:var(--text2); cursor:pointer; user-select:none; padding:8px 10px;
    border-bottom:2px solid var(--border); }}
  .lb-table th:hover {{ color:var(--text); }}
  .lb-table td {{ padding:7px 10px; border-bottom:1px solid var(--border); }}
  .lb-table tbody tr:hover td {{ background:#0e1d30; }}
  .sc1 {{ position:sticky !important; z-index:2 !important; background:var(--surf); left:0; min-width:36px; max-width:36px; }}
  .sc2 {{ position:sticky !important; z-index:2 !important; background:var(--surf); left:36px; min-width:200px; max-width:200px; }}
  .sc3 {{ position:sticky !important; z-index:2 !important; background:var(--surf); left:236px; min-width:55px; max-width:55px; }}
  .sc4 {{ position:sticky !important; z-index:2 !important; background:var(--surf); left:291px; min-width:75px; max-width:75px; box-shadow:3px 0 8px rgba(0,0,0,0.4); }}
  .lb-table thead .sc1,.lb-table thead .sc2,.lb-table thead .sc3,.lb-table thead .sc4 {{ background:var(--surf2) !important; z-index:3 !important; }}
  .lb-table tbody tr:hover .sc1,.lb-table tbody tr:hover .sc2,.lb-table tbody tr:hover .sc3,.lb-table tbody tr:hover .sc4 {{ background:#0e1d30 !important; }}
  .grp-l {{ border-left:2px solid var(--border) !important; }}

  /* ── Card Area ── */
  #card-area {{ max-width:920px; margin:0 auto; display:none; }}
  .back-btn {{
    display:inline-flex; align-items:center; gap:6px;
    margin-bottom:16px; padding:8px 16px;
    background:var(--surf); border:1px solid var(--border);
    border-radius:20px; color:var(--text2); font-size:0.85rem;
    cursor:pointer; transition:all 0.15s;
  }}
  .back-btn:hover {{ border-color:var(--accent); color:var(--accent); }}
  .player-card {{ background:var(--surf); border:1px solid var(--border); border-radius:var(--r); overflow:hidden; }}
  .card-header {{ background:linear-gradient(135deg,var(--surf2),#0d1b2a); padding:22px 26px; display:flex; align-items:center; gap:18px; border-bottom:1px solid var(--border); }}
  .avatar {{ width:68px; height:68px; border-radius:50%; background:linear-gradient(135deg,#1a3a5c,#00b4d8); display:flex; align-items:center; justify-content:center; font-size:1.1rem; font-weight:900; color:#fff; flex-shrink:0; letter-spacing:1px; text-align:center; }}
  .player-info h2 {{ font-size:1.3rem; font-weight:800; line-height:1.2; }}
  .player-meta {{ display:flex; gap:8px; margin-top:6px; flex-wrap:wrap; align-items:center; }}
  .meta-b {{ font-size:0.77rem; font-weight:600; padding:3px 10px; border-radius:20px; background:var(--surf); border:1px solid var(--border); color:var(--text2); }}
  .war-summary {{ padding:18px 26px; display:flex; gap:12px; flex-wrap:wrap; border-bottom:1px solid var(--border); background:linear-gradient(to right,rgba(0,180,216,0.05),transparent); }}
  .war-big {{ flex:1; min-width:120px; text-align:center; padding:12px 14px; background:var(--surf2); border-radius:10px; border:1px solid var(--border); }}
  .war-big.highlight {{ border-color:var(--accent); background:rgba(0,180,216,0.08); }}
  .war-big .lbl {{ font-size:0.68rem; font-weight:700; letter-spacing:1px; color:var(--text2); text-transform:uppercase; margin-bottom:4px; }}
  .war-big .val {{ font-size:1.9rem; font-weight:900; line-height:1; }}
  .war-big .sub {{ font-size:0.74rem; color:var(--text2); margin-top:4px; }}
  .card-body {{ padding:22px 26px; display:flex; flex-direction:column; gap:22px; }}
  .section-title {{ font-size:0.72rem; font-weight:700; letter-spacing:1.5px; text-transform:uppercase; color:var(--muted); margin-bottom:12px; display:flex; align-items:center; gap:8px; }}
  .section-title::after {{ content:''; flex:1; height:1px; background:var(--border); }}
  .stat-grid-lc {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(110px,1fr)); gap:10px; }}
  .stat-box {{ background:var(--surf2); border:1px solid var(--border); border-radius:8px; padding:10px 12px; text-align:center; }}
  .stat-box .sv {{ font-size:1.1rem; font-weight:800; }}
  .stat-box .sl {{ font-size:0.68rem; color:var(--text2); text-transform:uppercase; letter-spacing:0.5px; margin-top:2px; }}
  .bar-row {{ margin-bottom:12px; }}
  .bar-top {{ display:flex; justify-content:space-between; align-items:baseline; margin-bottom:4px; font-size:0.84rem; }}
  .bar-lbl {{ font-weight:600; }}
  .bar-val {{ font-weight:700; }}
  .bar-track {{ height:7px; background:var(--surf2); border-radius:4px; overflow:hidden; }}
  .bar-fill  {{ height:100%; border-radius:4px; }}
  .bar-blue  {{ background:linear-gradient(90deg,#0077b6,#00b4d8); }}
  .bar-green {{ background:linear-gradient(90deg,#2dc653,#00b4d8); }}
  .bar-red   {{ background:linear-gradient(90deg,#e63946,#c1121f); }}
</style>
</head>
<body>
<div class="header">
  <h1>⛸ Lines &amp; Pairings</h1>
  <p>{season} Season · 5-on-5</p>
</div>
{nav_html}

<div id="leaderboard-area">
  <div class="search-wrap">
    <input id="search" type="text" placeholder="Search by player name… (e.g. McDavid, Makar)" autocomplete="off">
    <div class="ac-box" id="ac-box"></div>
  </div>

  <div class="controls">
    <div class="tab active" data-pos="all"     onclick="setPos('all')">All</div>
    <div class="tab"        data-pos="line"    onclick="setPos('line')">Lines</div>
    <div class="tab"        data-pos="pairing" onclick="setPos('pairing')">Pairings</div>
    <select class="ctrl-select" id="team-sel" onchange="render()">
      {team_opts}
    </select>
    <label class="ctrl-label">
      Min TOI (min)
      <input class="ctrl-input" id="min-toi" type="number" value="100" min="0">
    </label>
    <span class="ctrl-right" id="row-count"></span>
  </div>

  <div class="wrap">
    <div class="lb-wrap">
      <table id="lb" class="lb-table">
        <thead>
          <tr>
            <th class="sc1" data-k="rank">#</th>
            <th class="sc2" data-k="name">Line / Pairing</th>
            <th class="sc3" data-k="team">Team</th>
            <th class="sc4" data-k="pos">Type</th>
            <th class="num grp-l" data-k="gp">GP</th>
            <th class="num" data-k="toi_min" title="Time on Ice (minutes)">TOI</th>
            <th class="num grp-l" data-k="xgf_pct" title="Expected Goals For %">xGF%</th>
            <th class="num" data-k="cf_pct" title="Corsi For %">CF%</th>
            <th class="num" data-k="ff_pct" title="Fenwick For %">FF%</th>
            <th class="num grp-l" data-k="gf" title="Goals For">GF</th>
            <th class="num" data-k="ga" title="Goals Against">GA</th>
            <th class="num grp-l" data-k="xgf" title="Expected Goals For">xGF</th>
            <th class="num" data-k="xga" title="Expected Goals Against">xGA</th>
            <th class="num grp-l" data-k="hdxgf_pct" title="High Danger Expected Goals For %">HDxGF%</th>
            <th class="num" data-k="hdxgf" title="High Danger Expected Goals For">HDxGF</th>
            <th class="num" data-k="hdxga" title="High Danger Expected Goals Against">HDxGA</th>
            <th class="num" data-k="hdgf" title="High Danger Goals For">HDGF</th>
            <th class="num" data-k="hdga" title="High Danger Goals Against">HDGA</th>
            <th class="num grp-l" data-k="sog_f" title="Shots on Goal For">SOG&nbsp;For</th>
            <th class="num" data-k="sog_a" title="Shots on Goal Against">SOG&nbsp;Agnst</th>
            <th class="num" data-k="rbd_f" title="Rebounds For">RBD&nbsp;For</th>
            <th class="num" data-k="rbd_a" title="Rebounds Against">RBD&nbsp;Agnst</th>
            <th class="num grp-l" data-k="giveaways" title="Giveaways">GV</th>
            <th class="num" data-k="takeaways" title="Takeaways">TK</th>
            <th class="num" data-k="fo_pct" title="Faceoff Win %">FO%</th>
          </tr>
        </thead>
        <tbody id="tbody"></tbody>
      </table>
      <div class="lb-footer">Data: Natural Stat Trick · 5-on-5 only · Built {ts}</div>
    </div>
  </div>
</div>

<!-- Line Card (full-page, hidden until selected) -->
<div id="card-area">
  <div class="back-btn" id="back-btn">← Back to Lines</div>
  <div id="card-inner"></div>
</div>

<script>
const ALL_LINES = {data_json};
let SORT_COL='xgf_pct', SORT_DIR=1, POS_FILTER='all';

function pCls(v)     {{ return v>=52?'c-green':v<=48?'c-red':'c-muted'; }}
function gCls(gf,ga) {{ return gf>ga?'c-green':gf<ga?'c-red':'c-muted'; }}
function bCls(gf,ga) {{ return ga<gf?'c-green':ga>gf?'c-red':'c-muted'; }}
function pBar(v)     {{ return v>=52?'bar-green':v<=48?'bar-red':'bar-blue'; }}
function esc(s)      {{ return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }}

function setPos(p) {{
  POS_FILTER=p;
  document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('active',t.dataset.pos===p));
  render();
}}

function getFiltered() {{
  const team=document.getElementById('team-sel').value;
  const minToi=parseFloat(document.getElementById('min-toi').value)||0;
  return ALL_LINES.filter(r=>{{
    if(POS_FILTER!=='all'&&r.pos!==POS_FILTER) return false;
    if(team&&r.team!==team) return false;
    if(r.toi_min<minToi) return false;
    return true;
  }});
}}

function render() {{
  const data=[...getFiltered()].sort((a,b)=>{{
    const av=a[SORT_COL]??-Infinity; const bv=b[SORT_COL]??-Infinity;
    return SORT_DIR*(bv-av);
  }});
  document.getElementById('row-count').textContent=data.length+' combos';
  document.querySelectorAll('#lb th').forEach(th=>{{
    th.classList.remove('sort-asc','sort-desc');
    if(th.dataset.k===SORT_COL) th.classList.add(SORT_DIR===-1?'sort-desc':'sort-asc');
  }});
  const tbody=document.getElementById('tbody');
  const n=(v,d=1)=>v!=null?(+v).toFixed(d):'—';
  tbody.innerHTML=data.map((r,i)=>`<tr style="cursor:pointer">
    <td class="sc1 rank-c">${{i+1}}</td>
    <td class="sc2 name-c">${{esc(r.name)}}</td>
    <td class="sc3">${{esc(r.team)}}</td>
    <td class="sc4"><span class="badge badge-${{r.pos}}">${{r.pos==='line'?'Line':'Pairing'}}</span></td>
    <td class="num grp-l">${{r.gp}}</td>
    <td class="num">${{r.toi_min.toFixed(0)}} min</td>
    <td class="num grp-l ${{pCls(r.xgf_pct)}}">${{n(r.xgf_pct)}}%</td>
    <td class="num ${{pCls(r.cf_pct)}}">${{n(r.cf_pct)}}%</td>
    <td class="num ${{pCls(r.ff_pct)}}">${{n(r.ff_pct)}}%</td>
    <td class="num grp-l ${{gCls(r.gf,r.ga)}}">${{r.gf}}</td>
    <td class="num ${{bCls(r.gf,r.ga)}}">${{r.ga}}</td>
    <td class="num grp-l">${{n(r.xgf,2)}}</td>
    <td class="num">${{n(r.xga,2)}}</td>
    <td class="num grp-l ${{pCls(r.hdxgf_pct)}}">${{n(r.hdxgf_pct)}}%</td>
    <td class="num">${{n(r.hdxgf,2)}}</td>
    <td class="num">${{n(r.hdxga,2)}}</td>
    <td class="num">${{r.hdgf}}</td>
    <td class="num">${{r.hdga}}</td>
    <td class="num grp-l">${{r.sog_f}}</td>
    <td class="num">${{r.sog_a}}</td>
    <td class="num">${{r.rbd_f}}</td>
    <td class="num">${{r.rbd_a}}</td>
    <td class="num grp-l">${{r.giveaways}}</td>
    <td class="num">${{r.takeaways}}</td>
    <td class="num">${{r.fo_pct!=null?n(r.fo_pct)+'%':'—'}}</td>
  </tr>`).join('');
  tbody.querySelectorAll('tr').forEach((tr,i)=>tr.addEventListener('click',()=>showCard(data[i])));
}}

document.querySelectorAll('#lb th').forEach(th=>{{
  th.addEventListener('click',()=>{{
    const k=th.dataset.k;
    if(!k||['rank','name','team','pos'].includes(k)) return;
    if(SORT_COL===k) SORT_DIR*=-1; else{{SORT_COL=k;SORT_DIR=-1;}}
    render();
  }});
}});

/* ── Line Card (full-page) ── */
function sb(lbl,val,cls='') {{
  return `<div class="stat-box"><div class="sv ${{cls}}">${{val}}</div><div class="sl">${{lbl}}</div></div>`;
}}
function barRow(lbl,val,pct,barCls) {{
  pct=Math.max(0,Math.min(100,pct));
  return `<div class="bar-row"><div class="bar-top"><span class="bar-lbl">${{lbl}}</span><span class="bar-val">${{val}}</span></div><div class="bar-track"><div class="bar-fill ${{barCls}}" style="width:${{pct}}%"></div></div></div>`;
}}

function showCard(r) {{
  const fo=r.fo_pct!=null?r.fo_pct.toFixed(1)+'%':'—';
  const diff=r.gf-r.ga; const dStr=(diff>=0?'+':'')+diff;
  const xgfColor=r.xgf_pct>=52?'#2dc653':r.xgf_pct<=48?'#e63946':'#e2e8f0';
  const diffColor=diff>0?'#2dc653':diff<0?'#e63946':'#e2e8f0';
  document.getElementById('card-inner').innerHTML=`
  <div class="player-card">
    <div class="card-header">
      <div class="avatar">${{r.pos==='line'?'LN':'PR'}}</div>
      <div class="player-info">
        <h2>${{esc(r.name)}}</h2>
        <div class="player-meta">
          <span class="meta-b"><span class="badge badge-${{r.pos}}">${{r.pos==='line'?'Line':'Pairing'}}</span></span>
          <span class="meta-b">${{esc(r.team)}}</span>
          <span class="meta-b">${{r.gp}} GP</span>
          <span class="meta-b">${{r.toi_min.toFixed(0)}} min TOI</span>
          <span class="meta-b">5-on-5</span>
        </div>
      </div>
    </div>
    <div class="war-summary">
      <div class="war-big highlight">
        <div class="lbl">xGF%</div>
        <div class="val" style="color:${{xgfColor}}">${{r.xgf_pct.toFixed(1)}}%</div>
        <div class="sub">Expected Share</div>
      </div>
      <div class="war-big">
        <div class="lbl">Goal Diff</div>
        <div class="val" style="color:${{diffColor}}">${{dStr}}</div>
        <div class="sub">${{r.gf}} GF – ${{r.ga}} GA</div>
      </div>
      <div class="war-big">
        <div class="lbl">CF%</div>
        <div class="val" style="color:${{r.cf_pct>=52?'#2dc653':r.cf_pct<=48?'#e63946':'#e2e8f0'}}">${{r.cf_pct.toFixed(1)}}%</div>
        <div class="sub">Corsi Share</div>
      </div>
      <div class="war-big">
        <div class="lbl">HDxGF%</div>
        <div class="val" style="color:${{r.hdxgf_pct>=52?'#2dc653':r.hdxgf_pct<=48?'#e63946':'#e2e8f0'}}">${{r.hdxgf_pct.toFixed(1)}}%</div>
        <div class="sub">HD Share</div>
      </div>
    </div>
    <div class="card-body">
      <div>
        <div class="section-title">Possession</div>
        ${{barRow('xGF%',    r.xgf_pct.toFixed(1)+'%',    r.xgf_pct*2,    pBar(r.xgf_pct))}}
        ${{barRow('CF%',     r.cf_pct.toFixed(1)+'%',     r.cf_pct*2,     pBar(r.cf_pct))}}
        ${{barRow('FF%',     r.ff_pct.toFixed(1)+'%',     r.ff_pct*2,     pBar(r.ff_pct))}}
        ${{barRow('HDxGF%',  r.hdxgf_pct.toFixed(1)+'%',  r.hdxgf_pct*2,  pBar(r.hdxgf_pct))}}
      </div>
      <div>
        <div class="section-title">Goals &amp; Expected Goals</div>
        <div class="stat-grid-lc">
          ${{sb('GF',    r.gf,   gCls(r.gf,r.ga))}}
          ${{sb('GA',    r.ga,   bCls(r.gf,r.ga))}}
          ${{sb('xGF',   r.xgf.toFixed(2))}}
          ${{sb('xGA',   r.xga.toFixed(2))}}
          ${{sb('HDxGF', r.hdxgf.toFixed(2))}}
          ${{sb('HDxGA', r.hdxga.toFixed(2))}}
          ${{sb('HDGF',  r.hdgf)}}
          ${{sb('HDGA',  r.hdga)}}
        </div>
      </div>
      <div>
        <div class="section-title">Shots &amp; Other</div>
        <div class="stat-grid-lc">
          ${{sb('SOG For',       r.sog_f)}}
          ${{sb('SOG Agnst',     r.sog_a)}}
          ${{sb('Rebounds For',  r.rbd_f)}}
          ${{sb('Rebounds Agnst',r.rbd_a)}}
          ${{sb('Giveaways',     r.giveaways)}}
          ${{sb('Takeaways',     r.takeaways)}}
          ${{r.fo_pct!=null?sb('FO%', fo, pCls(r.fo_pct)):''}}
        </div>
      </div>
    </div>
  </div>`;
  document.getElementById('leaderboard-area').style.display='none';
  document.getElementById('card-area').style.display='block';
  window.scrollTo({{top:0,behavior:'smooth'}});
}}

document.getElementById('back-btn').addEventListener('click',()=>{{
  document.getElementById('card-area').style.display='none';
  document.getElementById('leaderboard-area').style.display='block';
}});

/* ── Search ── */
const searchEl=document.getElementById('search');
const acBox=document.getElementById('ac-box');
searchEl.addEventListener('input',()=>{{
  const q=searchEl.value.trim().toLowerCase();
  if(!q){{acBox.style.display='none';return;}}
  const minToi=parseFloat(document.getElementById('min-toi').value)||0;
  const hits=ALL_LINES.filter(r=>r.name.toLowerCase().includes(q)&&r.toi_min>=minToi).slice(0,10);
  if(!hits.length){{acBox.style.display='none';return;}}
  acBox.innerHTML=hits.map(r=>`<div class="ac-item"><span class="ac-name">${{esc(r.name)}}</span><span class="ac-meta">${{r.pos==='line'?'Line':'Pairing'}} · ${{r.team}} · ${{r.toi_min.toFixed(0)}} min</span></div>`).join('');
  acBox.style.display='block';
  acBox.querySelectorAll('.ac-item').forEach((el,i)=>el.addEventListener('click',()=>{{
    showCard(hits[i]); searchEl.value=''; acBox.style.display='none';
  }}));
}});
document.addEventListener('click',e=>{{if(!e.target.closest('.search-wrap'))acBox.style.display='none';}});

let toiTimer;
document.getElementById('min-toi').addEventListener('input',()=>{{clearTimeout(toiTimer);toiTimer=setTimeout(render,300);}});

render();
</script>
</body>
</html>"""


def build_one_season(players, goalies, teams_df, lines_df, year, all_years,
                     out_dir, is_latest, player_history_json="{}", team_history_json="{}", player_index_json="{}"):
    """Write all HTML pages for one season using pre-computed WAR data."""
    import tempfile
    lbl = season_label(year)
    suffix = "" if is_latest else f"_{lbl}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    if is_latest:
        html_dir = out_dir
        in_subdir = False
    else:
        html_dir = os.path.join(out_dir, "Old Seasons")
        os.makedirs(html_dir, exist_ok=True)
        in_subdir = True

    print(f"\n{'='*50}")
    print(f"  📅 Season: {lbl}")
    print(f"{'='*50}")

    all_records = players + goalies
    print(f"   {len(players)} skaters · {len(goalies)} goalies")

    top5 = sorted(players, key=lambda p: p["Total_WAR"], reverse=True)[:5]
    print("🏆 Top 5:")
    for p in top5:
        print(f"   {p['name']:25s} {p['Total_WAR']:+.2f} WAR")

    # ── Players ──────────────────────────────────────────────────────────────
    nav = make_nav("players", year, all_years, is_in_subdir=in_subdir)
    html = build_html(all_records, lbl, timestamp, nav, player_history_json, player_index_json, in_subdir)
    path = os.path.join(html_dir, f"NHL_WAR_Cards{suffix}.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ {path} ({os.path.getsize(path)//1024} KB)")

    # ── Teams ─────────────────────────────────────────────────────────────────
    if teams_df is not None:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tf:
            teams_df.to_csv(tf, index=False)
            tmp_teams = tf.name
        nav = make_nav("teams", year, all_years, is_in_subdir=in_subdir)
        t_html = build_teams_html(tmp_teams, lbl, nav, team_history_json)
        os.remove(tmp_teams)
        path = os.path.join(html_dir, f"NHL_WAR_Teams{suffix}.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(t_html)
        print(f"✅ {path} ({os.path.getsize(path)//1024} KB)")

    # ── Lines ─────────────────────────────────────────────────────────────────
    if lines_df is not None:
        li = lines_df[lines_df["icetime"] >= 6000].copy() if not is_latest else lines_df
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tf:
            li.to_csv(tf, index=False)
            tmp_lines = tf.name
        nav = make_nav("lines", year, all_years, is_in_subdir=in_subdir)
        l_html = build_lines_html(tmp_lines, lbl, nav)
        os.remove(tmp_lines)
        path = os.path.join(html_dir, f"NHL_WAR_Lines{suffix}.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(l_html)
        print(f"✅ {path} ({os.path.getsize(path)//1024} KB)")


def build(script_dir):
    from collections import defaultdict
    out_dir    = script_dir
    cur_folder = os.path.join(script_dir, "2025-2026")
    old_folder = os.path.join(script_dir, "Old Seasons")

    # ── Discover seasons ──────────────────────────────────────────────────────
    old_years = []
    if os.path.exists(old_folder):
        sk_path = os.path.join(old_folder, "skaters_2008_to_2024.csv")
        if os.path.exists(sk_path):
            tmp = pd.read_csv(sk_path, usecols=["season"])
            old_years = sorted(tmp["season"].unique().tolist())

    cur_folder_name = os.path.basename(cur_folder)
    try:
        cur_year = int(cur_folder_name.split("-")[0])
    except Exception:
        cur_year = 2025

    all_years = old_years + [cur_year]
    print(f"📂 Found seasons: {[season_label(y) for y in sorted(all_years, reverse=True)]}")

    # ── Load raw CSVs once ────────────────────────────────────────────────────
    print(f"\n📂 Loading data…")
    cur_sk, cur_go = load_data(cur_folder)
    cur_teams_path = os.path.join(cur_folder, "teams.csv")
    cur_lines_path = os.path.join(cur_folder, "lines.csv")
    cur_teams_df = pd.read_csv(cur_teams_path) if os.path.exists(cur_teams_path) else None
    cur_lines_df = pd.read_csv(cur_lines_path) if os.path.exists(cur_lines_path) else None

    sk_all = go_all = tm_all = li_all = None
    if old_years:
        sk_all = pd.read_csv(os.path.join(old_folder, "skaters_2008_to_2024.csv"))
        go_all = pd.read_csv(os.path.join(old_folder, "goalies_2008_to_2024.csv"))
        tm_path = os.path.join(old_folder, "teams_2008_to_2024.csv")
        li_path = os.path.join(old_folder, "lines_2008_to_2024.csv")
        tm_all = pd.read_csv(tm_path) if os.path.exists(tm_path) else None
        li_all = pd.read_csv(li_path) if os.path.exists(li_path) else None

    # ── Pass 1: compute WAR for every season; collect history ─────────────────
    season_computed  = {}           # year -> (players, goalies, teams_df, lines_df)
    player_war_hist  = defaultdict(list)   # playerId -> [{s, w}]
    team_xgf_hist    = defaultdict(list)   # team_abbr -> [{s, x}]

    for year in sorted(all_years):
        lbl = season_label(year)
        is_latest = (int(year) == int(cur_year))

        if is_latest:
            sk, go, teams_df, lines_df = cur_sk, cur_go, cur_teams_df, cur_lines_df
        else:
            sk       = sk_all[sk_all["season"] == year].copy()
            go       = go_all[go_all["season"] == year].copy()
            teams_df = tm_all[tm_all["season"] == year].copy() if tm_all is not None else None
            lines_df = li_all[li_all["season"] == year].copy() if li_all is not None else None

        print(f"⚙️  Computing {lbl}…")
        players = compute_skater_war(sk, lbl)
        goalies = compute_goalie_war(go, lbl)
        players, goalies = add_percentiles(players, goalies)
        season_computed[year] = (players, goalies, teams_df, lines_df)

        # Collect WAR history per player
        for p in players + goalies:
            player_war_hist[p["playerId"]].append({"s": lbl, "w": p["Total_WAR"], "t": p["team"]})

        # Collect xGF% history per team (from 5on5 rows)
        if teams_df is not None:
            ev_teams = teams_df[teams_df["situation"] == "5on5"]
            for _, row in ev_teams.iterrows():
                team = str(row.get("team", ""))
                xgf_pct = round(float(row.get("xGoalsPercentage", 0.5)) * 100, 1)
                team_xgf_hist[team].append({"s": lbl, "x": xgf_pct})

    # Sort histories chronologically (oldest → newest = left → right in chart)
    for pid in player_war_hist:
        player_war_hist[pid].sort(key=lambda e: e["s"])
    for t in team_xgf_hist:
        team_xgf_hist[t].sort(key=lambda e: e["s"])

    player_history_json = json.dumps(dict(player_war_hist))
    team_history_json   = json.dumps(dict(team_xgf_hist))

    # Build cross-season player index (latest season appearance wins)
    player_index = {}
    for year in sorted(all_years):          # ascending → last write = most recent
        players_y, goalies_y, _, _ = season_computed[year]
        lbl = season_label(year)
        for p in players_y + goalies_y:
            player_index[p["playerId"]] = {
                "n":  p["name"],
                "pos": p["position"],
                "pg":  p["pos_group"],
                "t":   p["team"],
                "s":   lbl,
            }
    player_index_json = json.dumps(player_index)

    # ── Pass 2: build HTML for every season ────────────────────────────────────
    for year in sorted(all_years, reverse=True):
        is_latest = (int(year) == int(cur_year))
        players, goalies, teams_df, lines_df = season_computed[year]
        build_one_season(
            players, goalies, teams_df, lines_df,
            year, all_years, out_dir, is_latest,
            player_history_json, team_history_json,
            player_index_json,
        )

    print("\n\n✅ All seasons built successfully.")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    build(script_dir)
