import json, sys, urllib.request, datetime as dt
out = []
start = int(dt.datetime(2026,7,25,tzinfo=dt.timezone.utc).timestamp()*1000)
end = int(dt.datetime(2026,10,11,tzinfo=dt.timezone.utc).timestamp()*1000)
cur = start
while cur < end:
    url = f"https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=1h&startTime={cur}&endTime={end}&limit=1000"
    with urllib.request.urlopen(url, timeout=60) as r:
        rows = json.load(r)
    if not rows: break
    out.extend(rows)
    cur = rows[-1][0] + 3600_000
json.dump(out, open("/home/user/btc-dashboard/runs/noctua-disproof-2026-10-10/audits/scratch_A3/binance_1h.json","w"))
print(len(out), dt.datetime.fromtimestamp(out[0][0]/1000, dt.timezone.utc), dt.datetime.fromtimestamp(out[-1][0]/1000, dt.timezone.utc))
