# app.py — AP Elections API simulator (randomizes every 30s)
#
# Endpoints:
#   - /api/ping
#   - /metrics                         -> {"total_calls": ..., "calls_per_minute": ...}
#   - /v2/elections/{date}?statepostal=XX&raceTypeId=G&raceId=0&level=ru&officeId=P|S|G
#       -> county-level <ReportingUnit ...><Candidate .../></ReportingUnit>
#   - /v2/districts/{date}?statepostal=XX&level=ru&officeId=H
#       -> congressional-district <ReportingUnit ...><Candidate .../></ReportingUnit>
#
# Change vs original:
#   - NEW: EPOCH_SECONDS (default 30). Every epoch all vote totals re-randomize.
#   - Names/parties and XML structure unchanged. Metrics preserved.
#
# Run (dev):
#   uvicorn app:app --reload --port 5022
#
# Env knobs:
#   EPOCH_SECONDS=30   # how often a totally new dataset is produced
#   UPDATE_BUCKETS=36  # kept for compatibility (not used for growth anymore)
#   BASELINE_DRIFT=0   # kept for compatibility (not used)
#   TICK_SECONDS=10    # kept for compatibility (not used)
#
# Requires cb_2024_us_cd119_500k.json present in working directory.

import os, time, re, hashlib, json
from typing import Dict, List, Tuple
import httpx

from fastapi import FastAPI, Query, Request
from fastapi.responses import PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

# ---------------------------- App & Metrics ---------------------------- #

app = FastAPI(title="AP Elections API Simulator (30s random epochs)")

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

UPDATE_BUCKETS  = int(os.getenv("UPDATE_BUCKETS", "36"))  # kept for compat
BASELINE_DRIFT  = int(os.getenv("BASELINE_DRIFT", "0"))   # kept for compat
TICK_SECONDS    = int(os.getenv("TICK_SECONDS", "10"))    # kept for compat
EPOCH_SECONDS   = int(os.getenv("EPOCH_SECONDS", "10"))   # << NEW

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
    h = hashlib.blake2b(seed.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(h, "big")

def _epoch_key() -> int:
    # A new integer every EPOCH_SECONDS; used to fully randomize totals
    return int(time.time() // max(1, EPOCH_SECONDS))

def apize_name(usps: str, canonical: str) -> str:
    n = canonical
    if n.startswith("Saint "): n = "St. " + n[6:]
    n = n.replace("Doña", "Dona")
    n = re.sub(r"\bLa\s+Salle\b", "LaSalle", n)
    n = n.replace("DeKalb", "De Kalb")
    return n

def county_suffix(usps: str, name: str) -> str:
    if name.lower().endswith("city"):       # Carson City NV, VA indep. cities, DC, etc.
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
    # Ensure REP & DEM among top two
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
    """
    County-level 3-way totals. Fully re-randomized each EPOCH_SECONDS window.
    """
    epoch = _epoch_key()
    base_seed = seeded_rng_u32(f"{fips}-{epoch}")
    r1 = (base_seed & 0xFFFF)
    r2 = ((base_seed >> 16) & 0xFFFF)

    # Random total per epoch: 2k–250k + jitter
    base_total = 2000 + (base_seed % 250000)

    # Random split with mild bias; ensure non-negative
    rep = int(base_total * (0.30 + (r1 % 40) / 100.0))  # 30–70%
    dem = int(base_total * (0.20 + (r2 % 55) / 100.0))  # 20–75%
    # Whatever remains goes to IND, but ensure >= 0
    ind = max(0, base_total - rep - dem)

    # Small shuffle so totals don't always rep+dem>ind in same pattern
    # (still deterministic within epoch)
    if (base_seed % 3) == 0 and ind > 0:
        shift = min(ind // 10, 500)
        rep += shift; ind -= shift
    elif (base_seed % 3) == 1 and ind > 0:
        shift = min(ind // 10, 500)
        dem += shift; ind -= shift

    return max(rep,0), max(dem,0), max(ind,0)

def simulated_cd_votes_30s(did: str, k: int = 3) -> List[int]:
    """
    District-level k-way totals. Fully re-randomized each EPOCH_SECONDS window.
    """
    epoch = _epoch_key()
    base_seed = seeded_rng_u32(f"{did}-{epoch}")

    # Total turnout per epoch ~ 150k–900k
    base_total = 150_000 + (base_seed % 750_000)

    # Random Dirichlet-like shares from seeded slices
    parts = []
    rem = base_total
    for i in range(k - 1):
        slice_i = int((0.15 + ((base_seed >> (i*3)) % 60)/100.0) * (rem / (k - i)))
        parts.append(max(1, slice_i))
        rem -= slice_i
    parts.append(max(1, rem))

    # Mild reordering per epoch for variety
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
            tail = re.findall(r"(\d{1,2})$", gid.replace("-", ""))
            dnum = int(tail[0]) if tail else 1

        label = district_label(dnum)
        STATE_CD_REGISTRY.setdefault(usps, []).append((gid, dnum, label))

    for usps in STATE_CD_REGISTRY:
        STATE_CD_REGISTRY[usps].sort(key=lambda t: (t[1], t[0]))

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
        rep, dem, ind = simulated_votes_30s(fips)
        parts.append(f'  <ReportingUnit Name="{apname}" FIPS="{fips}">')
        for (full, party), vv in zip(slate, (rep, dem, ind)):
            first = full.split(" ", 1)[0]
            last  = full.split(" ", 1)[-1] if " " in full else ""
            parts.append(f'    <Candidate First="{first}" Last="{last}" Party="{party}" VoteCount="{vv}"/>')
        parts.append(  "  </ReportingUnit>")
    parts.append("</ElectionResults>")
    return Response(content="\n".join(parts), media_type="application/xml")

# ----------------------- Congressional Districts API -------------- #

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
    # Optional: serve a local index if present
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return Response(content=f.read(), media_type="text/html")
    except FileNotFoundError:
        return Response(content="<h1>API is running</h1>", media_type="text/html")

# ----------------------------- Local dev -------------------------- #

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "5022")))
