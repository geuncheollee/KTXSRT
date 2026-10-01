"""Acquire public KRIC/SR observations; no submission supplements are required."""
from __future__ import annotations
import argparse, calendar, hashlib, json, re, shutil, sys
from datetime import date, datetime, timezone
from pathlib import Path
import holidays
import pandas as pd
import requests
from bs4 import BeautifulSoup
from experiment_config import PROTOCOL
ROOT = Path(__file__).resolve().parent
KRIC = "https://www.kric.go.kr/jsp/industry/rss/railcarkindpassList.jsp"
SR = "https://www.data.go.kr/data/15071484/fileData.do"
PAX = ["KTX", "새마을", "무궁화", "통근열차", "누리로", "KTX-산천", "KTX-호남", "KTX-이음", "KTX-청룡", "ITX-새마을", "ITX-청춘열차", "ITX-마음"]
# Public government announcement dates, used to exclude holidays unknown at origin.
ANNOUNCEMENTS = [("2023-10-02", "2023-08-31", "148919605"), ("2024-10-01", "2024-09-03", "148933400"), ("2025-01-27", "2025-01-14", "148938559"), ("2025-06-03", "2025-04-08", "148941443")]

def fetch(url, session, **kwargs):
    r = session.post(url, timeout=60, **kwargs) if "data" in kwargs else session.get(url, timeout=60, **kwargs)
    r.raise_for_status()
    return r

def kric_rows(html, year):
    rows = []
    for table in BeautifulSoup(html, "html.parser").find_all("table"):
        if "연월" not in table.get_text() or "수송인원" not in table.get_text():
            continue
        for tr in table.find_all("tr"):
            cells = [c.get_text(strip=True).replace(",", "") for c in tr.find_all(["td", "th"])]
            if cells and re.fullmatch(fr"{year}-\d{{2}}", cells[0]):
                if len(cells) < 27:
                    raise ValueError("KRIC schema changed: expected 27 columns")
                rows.append([cells[0]]+[float(c) if c not in ("", "-", "–") else 0 for c in cells[1:27]])
    if len(rows) != 12 or len({r[0] for r in rows}) != 12:
        raise ValueError(f"Expected 12 unique KRIC months for {year}; received {len(rows)}")
    return rows

def download_sources(args):
    raw = ROOT / "data/public_sources"
    raw.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0", "Referer": KRIC})
    kric_path, sr_path = raw / "kric_carkind_raw.csv", raw / "srt_20251231.csv"
    if args.kric_csv:
        shutil.copyfile(args.kric_csv, kric_path)
    else:
        rows = []
        for year in range(2021, 2026):
            r = fetch(KRIC, session, data={"q_fdate": str(year), "fdate": str(year)})
            r.encoding = r.apparent_encoding
            (raw / f"kric_{year}.html").write_text(r.text, encoding="utf-8")
            rows.extend(kric_rows(r.text, year))
            print(f"Downloaded KRIC {year}: 12 months", flush=True)
        columns = ["yyyymm", "total_pax", "total_pkm"]+[c+"_pax" for c in PAX]+[c+"_pkm" for c in PAX]
        pd.DataFrame(rows, columns=columns).to_csv(kric_path, index=False, encoding="utf-8-sig")
    if args.srt_csv:
        shutil.copyfile(args.srt_csv, sr_path)
    else:
        r = fetch(SR, session)
        r.encoding = "utf-8"
        (raw / "sr_source_page.html").write_text(r.text, encoding="utf-8")
        links = []
        for node in BeautifulSoup(r.text, "html.parser").find_all("script", type="application/ld+json"):
            try:
                distributions = json.loads(node.string or node.get_text()).get("distribution", [])
                for item in distributions if isinstance(distributions, list) else [distributions]:
                    if item.get("contentUrl"):
                        links.append(item["contentUrl"])
            except (ValueError, AttributeError):
                continue
        if not links:
            # Some portal releases contain unescaped newlines in JSON-LD descriptions.
            links = re.findall(r'"contentUrl"\s*:\s*"(https://www\.data\.go\.kr/cmm/cmm/fileDownload\.do\?[^"<>]+)"', r.text)
        if not links:
            raise RuntimeError("SR link unavailable. Download release 20251231 at the source page and use --srt-csv.")
        sr_path.write_bytes(fetch(links[0], session).content)
        print("Downloaded SR dataset 15071484", flush=True)
    return kric_path, sr_path

def build_panel(kric_path, sr_path):
    kric = pd.read_csv(kric_path, encoding="utf-8-sig")
    kric["date"] = pd.to_datetime(kric.yyyymm, format="%Y-%m").dt.strftime("%Y-%m-01")
    kric = kric[kric.date.between("2021-01-01", "2025-12-01")].copy()
    cols = [c+"_pax" for c in ("KTX", "KTX-산천", "KTX-호남", "KTX-이음", "KTX-청룡")]
    kric["ktx_pax_total"] = kric[cols].fillna(0).sum(axis=1).astype(int)
    for encoding in ("cp949", "utf-8-sig"):
        try:
            srt = pd.read_csv(sr_path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    if len(srt) != 12 or any(f"{y}년 전체" not in srt for y in range(2021, 2026)):
        raise ValueError("Expected SR release 20251231, with 12 rows and 2021-2025 totals")
    sr_counts = {}
    for _, row in srt.iterrows():
        month = int(re.search(r"\d+", str(row["운행월"])).group())
        for year in range(2021, 2026):
            total = int(float(str(row[f"{year}년 전체"]).replace(",", "")))
            routes = sum(int(float(str(row[f"{year}년 {route}"]).replace(",", "")))
                         for route in ("경부선", "호남선", "경전선", "동해선", "전라선")
                         if f"{year}년 {route}" in srt and pd.notna(row[f"{year}년 {route}"]))
            if total != routes:
                raise ValueError("SR route totals do not agree")
            sr_counts[f"{year}-{month:02d}-01"] = total
    kr = holidays.country_holidays("KR", years=range(2021, 2026), observed=True)
    holiday_dates = set(kr) | {date.fromisoformat(item[0]) for item in ANNOUNCEMENTS}
    rows = []
    for _, row in kric.sort_values("date").iterrows():
        first = date.fromisoformat(row.date)
        days = calendar.monthrange(first.year, first.month)[1]
        dates = [date(first.year, first.month, d) for d in range(1, days+1)]
        sr = sr_counts[row.date]
        rows.append({"date": row.date, "ktx_pax_total": int(row.ktx_pax_total), "srt_pax_total": sr,
                     "hsr_pax_total": int(row.ktx_pax_total)+sr, "month_days": days,
                     "weekday_count": sum(d.weekday() < 5 for d in dates),
                     "weekday_holidays": sum(d.weekday() < 5 and d in holiday_dates for d in dates)})
    panel = pd.DataFrame(rows)
    expected = pd.date_range("2021-01-01", "2025-12-01", freq="MS").strftime("%Y-%m-%d").tolist()
    if panel.date.tolist() != expected or not (panel.hsr_pax_total > 0).all():
        raise ValueError("Sources must provide 60 complete positive observations")
    panel["y_lag12"] = panel.hsr_pax_total.shift(12)
    return panel

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kric-csv", type=Path, help="Optional manually acquired raw KRIC CSV, in the README schema")
    parser.add_argument("--srt-csv", type=Path, help="Optional SR release 20251231 downloaded from its public page")
    args = parser.parse_args()
    paths = download_sources(args)
    panel = build_panel(*paths)
    target = ROOT / PROTOCOL["panel"]
    target.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(target, index=False, encoding="utf-8-sig")
    protocol = ROOT / "revision/forecast_protocol"
    protocol.mkdir(parents=True, exist_ok=True)
    (protocol / "protocol_v1.json").write_text(json.dumps(PROTOCOL, indent=2)+"\n", encoding="utf-8")
    pd.DataFrame([{"holiday_date": d, "announcement_date": a, "source_url": "https://www.korea.kr/news/policyNewsView.do?newsId="+i} for d,a,i in ANNOUNCEMENTS]).to_csv(protocol / "holiday_announcement_ledger.csv", index=False)
    manifest = {"downloaded_at_utc": datetime.now(timezone.utc).isoformat(), "kric_url": KRIC,
                "sr_url": SR, "sr_study_release": "20251231; registered 2026-02-23",
                "holidays_version": holidays.__version__,
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (ROOT / "data/public_sources/download_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print("Prepared 60 observed months from public sources, without supplements or imputation.", flush=True)
    from build_forecast_inputs import main as build_inputs
    build_inputs([])

if __name__ == "__main__":
    main()
