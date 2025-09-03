# app.py — AP Elections API simulator (randomizes every 30s) + MANUAL OVERRIDES
#
# New:
#   - In-memory (optional on-disk) overrides for county vote totals (REP/DEM/IND)
#   - Registry endpoints for UI dropdowns:
#       GET /registry/states                -> ["AL","AK",...]
#       GET /registry/counties?state=XX     -> [{"fips":"06037","name":"Los Angeles County"}, ...]
#   - Override endpoints:
#       GET    /overrides                   -> current overrides
#       POST   /override/county             -> {"fips":"06037","rep":1,"dem":2,"ind":3}
#       DELETE /override/county?fips=06037  -> remove one county override
#       DELETE /overrides                   -> clear all overrides
#
# Behavior:
#   - If a county FIPS is overridden, /v2/elections/... uses that value
#     instead of the randomized epoch value — across epochs.
#
# Notes:
#   - Set SAVE_OVERRIDES=1 to persist overrides to ./overrides.json
#   - This keeps all existing routes and randomization logic intact.
#
# Base app (randomized epochs, endpoints) credit: your original file.
# --------------------------------------------------------------------

import os, time, re, hashlib, json
from typing import Dict, List, Tuple, Optional
import httpx

from fastapi import FastAPI, Query, Request, HTTPException
from fastapi.responses import PlainTextResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="AP Elections API Simulator (30s random epochs + overrides)")

# ---------------------------- Metrics ---------------------------- #

from collections import deque
TOTAL_CALLS = 0
REQUEST_TIMES = deque(maxlen=10000)

@app.middleware("http")
async def _count_calls(request: Request, call_next):
    is_metrics = request.url.path == "/metrics"
    resp = await call_next(request)
    if not is_metrics:
        global TOTAL_CALLS
        TOTAL_CALLS += 1
        REQUEST_TIMES.append(time.time())
    return resp

@app.get("/metrics")
def metrics():
    now = time.time()
    last_min = [t for t in REQUEST_TIMES if now - t <= 60]
    return {"total_calls": TOTAL_CALLS, "calls_per_minute": len(last_min)}

@app.get("/api/ping", response_class=PlainTextResponse)
def ping():
    return "pong"

# ---------------------------- Tunables ---------------------------- #

UPDATE_BUCKETS  = int(os.getenv("UPDATE_BUCKETS", "36"))  # compat
BASELINE_DRIFT  = int(os.getenv("BASELINE_DRIFT", "0"))   # compat
TICK_SECONDS    = int(os.getenv("TICK_SECONDS", "10"))    # compat
EPOCH_SECONDS   = int(os.getenv("EPOCH_SECONDS", "30"))   # << still the epoch cadence
SAVE_OVERRIDES  = os.getenv("SAVE_OVERRIDES", "0") in ("1","true","True","YES","yes")
OVERRIDE_PATH   = os.getenv("OVERRIDE_PATH", "overrides.json")

US_ATLAS_COUNTIES_URL = "https://cdn.jsdelivr.net/npm/us-atlas@3/counties-10m.json"

STATE_FIPS_TO_USPS = {
    "01":"AL","02":"AK","04":"AZ","05":"AR","06":"CA","08":"CO","09":"CT","10":"DE","11":"DC",
    "12":"FL","13":"GA","15":"HI","16":"ID","17":"IL","18":"IN","19":"IA","20":"KS","21":"KY",
    "22":"LA","23":"ME","24":"MD","25":"MA","26":"MI","27":"MN","28":"MS","29":"MO","30":"MT",
    "31":"NE","32":"NV","33":"NH","34":"NJ","35":"NM","36":"NY","37":"NC","38":"ND","39":"OH",
    "40":"OK","41":"OR","42":"PA","44":"RI","45":"SC","46":"SD","47":"TN","48":"TX","49":"UT",
    "50":"VT","51":"VA","53":"WA","54":"WV","55":"WI","56":"WY","72":"PR"
}
USPS_TO_STATE_FIPS = {v:k for k,v in STATE_FIPS_TO_USPS.items()}
PARISH_STATES = {"LA"}
INDEPENDENT_CITY_STATES = {"VA"}

# ----------------------- Helpers / RNG / Names -------------------- #

def seeded_rng_u32(seed: str) -> int:
    import hashlib as _h
    h = _h.blake2b(seed.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(h, "big")

def _epoch_key() -> int:
    return int(time.time() // max(1, EPOCH_SECONDS))

def apize_name(usps: str, canonical: str) -> str:
    n = canonical
    if n.startswith("Saint "): n = "St. " + n[6:]
    n = n.replace("Doña", "Dona")
    n = re.sub(r"\bLa\s+Salle\b", "LaSalle", n)
    n = n.replace("DeKalb", "De Kalb")
    return n

def county_suffix(usps: str, name: str) -> str:
    if name.lower().endswith("city"):
        return name
    if usps in PARISH_STATES:
        return name if name.lower().endswith("parish") else f"{name} Parish"
    return name if name.lower().endswith("county") else f"{name} County"

def ordinal(n: int) -> str:
    return "%d%s" % (n, "th" if 11<=n%100<=13 else {1:"st",2:"nd",3:"rd"}.get(n%10, "th"))

def district_label(n: int) -> str:
    return f"{ordinal(n)} Congressional District"

FIRSTS = ["Alex","Taylor","Jordan","Casey","Riley","Avery","Morgan","Quinn","Hayden","Rowan",
          "Elliot","Jesse","Drew","Parker","Reese"]
LASTS  = ["Smith","Johnson","Brown","Jones","Garcia","Miller","Davis","Martinez","Clark","Lewis",
          "Walker","Young","Allen","King","Wright"]
PARTY_POOL = ["REP","DEM","IND"]

def gen_cd_candidates(did: str, n: int = 3) -> List[Tuple[str,str]]:
    base = seeded_rng_u32(did)
    out, used = [], set()
    for i in range(n):
        f = FIRSTS[(base + i*13) % len(FIRSTS)]
        l = LASTS[(base // 7 + i*17) % len(LASTS)]
        nm = f"{f} {l}"
        if nm in used: nm = f"{nm} Jr."
        used.add(nm)
        party = PARTY_POOL[(base // (i+3)) % len(PARTY_POOL)] if i < 3 else "IND"
        out.append((nm, party))
    names = [x for x in out]
    have = {p for _,p in names[:2]}
    if "REP" not in have: names[0] = (names[0][0], "REP")
    if "DEM" not in have: names[1] = (names[1][0], "DEM")
    return names

def gen_statewide_candidates(usps: str, office: str, n: int = 3) -> List[Tuple[str,str]]:
    office = (office or "P").upper()
    if office == "P":
        return [("Donald Trump","REP"), ("Kamala Harris","DEM"), ("Robert Kennedy","IND")]
    base = seeded_rng_u32(f"{usps}-{office}")
    out, used = [], set()
    for i in range(n):
        f = FIRSTS[(base + i*11) % len(FIRSTS)]
        l = LASTS[(base // 5 + i*7) % len(LASTS)]
        nm = f"{f} {l}"
        if nm in used: nm = f"{nm} II"
        used.add(nm)
        party = PARTY_POOL[(base // (i+2)) % len(PARTY_POOL)] if i < 3 else "IND"
        out.append((nm, party))
    names = [x for x in out]
    have = {p for _,p in names[:2]}
    if "REP" not in have: names[0] = (names[0][0], "REP")
    if "DEM" not in have: names[1] = (names[1][0], "DEM")
    return names

# --------------------- Randomized vote models (30s) -------------------- #

def simulated_votes_30s(fips: str) -> Tuple[int,int,int]:
    epoch = _epoch_key()
    base_seed = seeded_rng_u32(f"{fips}-{epoch}")
    r1 = (base_seed & 0xFFFF)
    r2 = ((base_seed >> 16) & 0xFFFF)

    base_total = 2000 + (base_seed % 250000)
    rep = int(base_total * (0.30 + (r1 % 40) / 100.0))
    dem = int(base_total * (0.20 + (r2 % 55) / 100.0))
    ind = max(0, base_total - rep - dem)

    if (base_seed % 3) == 0 and ind > 0:
        shift = min(ind // 10, 500)
        rep += shift; ind -= shift
    elif (base_seed % 3) == 1 and ind > 0:
        shift = min(ind // 10, 500)
        dem += shift; ind -= shift
    return max(rep,0), max(dem,0), max(ind,0)

def simulated_cd_votes_30s(did: str, k: int = 3) -> List[int]:
    epoch = _epoch_key()
    base_seed = seeded_rng_u32(f"{did}-{epoch}")
    base_total = 150_000 + (base_seed % 750_000)
    parts = []
    rem = base_total
    for i in range(k - 1):
        slice_i = int((0.15 + ((base_seed >> (i*3)) % 60)/100.0) * (rem / (k - i)))
        parts.append(max(1, slice_i))
        rem -= slice_i
    parts.append(max(1, rem))
    if k >= 3 and (base_seed % 2):
        parts[0], parts[1] = parts[1], parts[0]
    return [max(0, x) for x in parts[:k]]

# --------------------- Registries built at startup ---------------- #

STATE_REGISTRY: Dict[str, List[Tuple[str,str,str]]] = {}
STATE_CD_REGISTRY: Dict[str, List[Tuple[str,int,str]]] = {}

@app.on_event("startup")
async def bootstrap():
    # Counties
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(US_ATLAS_COUNTIES_URL); r.raise_for_status()
        topo = r.json()
    geoms = topo.get("objects", {}).get("counties", {}).get("geometries", [])
    for g in geoms:
        fips = str(g.get("id", "")).zfill(5)
        props = g.get("properties", {}) or {}
        name = props.get("name") or props.get("NAMELSAD") or fips
        state_fips = fips[:2]
        usps = STATE_FIPS_TO_USPS.get(state_fips)
        if not usps:
            continue
        canonical = re.sub(r"\s+(County|Parish|city)$", "", name)
        apname = county_suffix(usps, canonical)
        apname = apize_name(usps, apname)
        STATE_REGISTRY.setdefault(usps, []).append((fips, canonical, apname))
    for usps in STATE_REGISTRY:
        STATE_REGISTRY[usps].sort(key=lambda t: t[0])

    # Congressional districts (local TopoJSON file)
    with open("cb_2024_us_cd119_500k.json", "r", encoding="utf-8") as f:
        cd_topo = json.load(f)

    cd_obj = (cd_topo.get("objects", {}).get("districts")
        or cd_topo.get("objects", {}).get("congressional-districts")
        or cd_topo.get("objects", {}).get(next(iter(cd_topo.get("objects", {})), ""), {}))
    geoms_cd = cd_obj.get("geometries", []) if cd_obj else []

    for g in geoms_cd:
        gid = str(g.get("id", "")).strip()
        props = g.get("properties", {}) or {}

        m = re.match(r"^\s*(\d{2})", gid) or re.match(r"^\s*(\d{2})", str(props.get("STATEFP") or ""))
        if not m:
            continue
        state_fips = m.group(1)
        usps = STATE_FIPS_TO_USPS.get(state_fips)
        if not usps:
            continue

        dnum = None
        for key in ("CD119FP","district","DISTRICT","cd","CD","number","NUM"):
            if key in props and str(props[key]).strip().isdigit():
                dnum = int(str(props[key]).strip())
                break
        if dnum is None:
            import re as _re
            tail = _re.findall(r"(\d{1,2})$", gid.replace("-", ""))
            dnum = int(tail[0]) if tail else 1

        label = district_label(dnum)
        STATE_CD_REGISTRY.setdefault(usps, []).append((gid, dnum, label))

    for usps in STATE_CD_REGISTRY:
        STATE_CD_REGISTRY[usps].sort(key=lambda t: (t[1], t[0]))

    # Load any saved overrides at boot
    _load_overrides()

# ---------------------------- Overrides --------------------------- #

# Structure:
# OVERRIDES = {
#   "counties": {
#       "06037": {"REP":123, "DEM":456, "IND":78},
#       ...
#   }
# }
OVERRIDES: Dict[str, Dict[str, Dict[str, int]]] = {"counties": {}}

def _save_overrides():
    if not SAVE_OVERRIDES:
        return
    try:
        with open(OVERRIDE_PATH, "w", encoding="utf-8") as f:
            json.dump(OVERRIDES, f, indent=2, sort_keys=True)
    except Exception as e:
        print("[WARN] Failed to save overrides:", e)

def _load_overrides():
    global OVERRIDES
    if not SAVE_OVERRIDES:
        return
    try:
        if os.path.exists(OVERRIDE_PATH):
            with open(OVERRIDE_PATH, "r", encoding="utf-8") as f:
                OVERRIDES = json.load(f)
        else:
            OVERRIDES = {"counties": {}}
    except Exception as e:
        print("[WARN] Failed to load overrides:", e)
        OVERRIDES = {"counties": {}}

@app.get("/overrides")
def get_overrides():
    return OVERRIDES

@app.post("/override/county")
async def set_county_override(payload: dict):
    fips = str(payload.get("fips") or "").zfill(5)

    # take raw values instead of forcing int()
    rep = payload.get("rep")
    dem = payload.get("dem")
    ind = payload.get("ind")

    if not fips.isdigit() or len(fips) != 5:
        raise HTTPException(status_code=400, detail="invalid fips")

    # no numeric validation — store as-is (could be numbers or strings)
    OVERRIDES.setdefault("counties", {})[fips] = {"REP": rep, "DEM": dem, "IND": ind}
    _save_overrides()
    return {"ok": True, "fips": fips, "override": OVERRIDES["counties"][fips]}


@app.delete("/override/county")
def delete_county_override(fips: str):
    fips = str(fips).zfill(5)
    if fips in OVERRIDES.get("counties", {}):
        OVERRIDES["counties"].pop(fips, None)
        _save_overrides()
        return {"ok": True, "removed": fips}
    return {"ok": True, "removed": None}

@app.delete("/overrides")
def clear_overrides():
    OVERRIDES["counties"].clear()
    _save_overrides()
    return {"ok": True}

# --------------------------- Registries for UI -------------------- #

@app.get("/registry/states")
def list_states():
    # Only states that actually have counties loaded
    out = sorted(STATE_REGISTRY.keys())
    return out

@app.get("/registry/counties")
def list_counties(state: str = Query(..., min_length=2, max_length=2)):
    usps = state.upper()
    rows = STATE_REGISTRY.get(usps, [])
    # Return FIPS + AP name ("Foo County" / "Bar Parish")
    return [{"fips": f, "name": ap} for (f, _canonical, ap) in rows]

# ---------------------------- Counties API ------------------------ #

@app.get("/v2/elections/{date}")
def elections_state_ru(
    request: Request,
    date: str,
    statepostal: str = Query(..., min_length=2, max_length=2),
    raceTypeId: str = Query("G"),
    raceId: str = Query("0"),
    level: str = Query("ru"),
    officeId: str = Query("P", regex="^[PSG]$"),  # President/Senate/Governor
):
    usps = statepostal.upper()
    officeId = (officeId or "P").upper()
    counties = STATE_REGISTRY.get(usps, [])

    if not counties:
        xml = f'<ElectionResults Date="{date}" StatePostal="{usps}" Office="{officeId}"></ElectionResults>'
        return Response(content=xml, media_type="application/xml")

    slate = gen_statewide_candidates(usps, officeId, n=3)
    epoch = _epoch_key()

    parts = [f'<ElectionResults Date="{date}" StatePostal="{usps}" Office="{officeId}" Epoch="{epoch}">']
    for fips, canonical, apname in counties:
        # --- OVERRIDE HOOK ---
        ovr = OVERRIDES.get("counties", {}).get(fips)
        if ovr:
            rep, dem, ind = ovr["REP"], ovr["DEM"], ovr["IND"]
        else:
            rep, dem, ind = simulated_votes_30s(fips)
        # ---------------------

        parts.append(f'  <ReportingUnit Name="{apname}" FIPS="{fips}">')
        for (full, party), vv in zip(slate, (rep, dem, ind)):
            first = full.split(" ", 1)[0]
            last  = full.split(" ", 1)[-1] if " " in full else ""
            parts.append(f'    <Candidate First="{first}" Last="{last}" Party="{party}" VoteCount="{vv}"/>')
        parts.append(  "  </ReportingUnit>")
    parts.append("</ElectionResults>")
    return Response(content="\n".join(parts), media_type="application/xml")

# ----------------------- Congressional Districts API -------------- #
# (House left randomized; you can add a similar override dict for CDs if needed)

@app.get("/v2/districts/{date}")
def districts_state_ru(
    request: Request,
    date: str,
    statepostal: str = Query(..., min_length=2, max_length=2),
    level: str = Query("ru"),
    officeId: str = Query("H", regex="^[H]$"),  # House only
):
    usps = statepostal.upper()
    districts = STATE_CD_REGISTRY.get(usps, [])

    if not districts:
        xml = f'<ElectionResults Date="{date}" StatePostal="{usps}" Office="{officeId}"></ElectionResults>'
        return Response(content=xml, media_type="application/xml")

    epoch = _epoch_key()
    parts = [f'<ElectionResults Date="{date}" StatePostal="{usps}" Office="{officeId}" Epoch="{epoch}">']
    for did, dnum, label in districts:
        cand = gen_cd_candidates(did, n=3)
        votes = simulated_cd_votes_30s(did, k=len(cand))
        parts.append(f'  <ReportingUnit Name="{label}" DistrictId="{did}" District="{dnum}">')
        for (full, party), v in zip(cand, votes):
            first = full.split(" ", 1)[0]
            last  = full.split(" ", 1)[-1] if " " in full else ""
            parts.append(f'    <Candidate First="{first}" Last="{last}" Party="{party}" VoteCount="{v}"/>')
        parts.append(  "  </ReportingUnit>")
    parts.append("</ElectionResults>")
    return Response(content="\n".join(parts), media_type="application/xml")

# ---------------------------- Static & Index ---------------------- #

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=Response)
def read_index():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return Response(content=f.read(), media_type="text/html")
    except FileNotFoundError:
        return Response(content="<h1>API is running</h1>", media_type="text/html")

# ----------------------------- Local dev -------------------------- #

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "5022")))
