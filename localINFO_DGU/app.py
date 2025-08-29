# app.py
from flask import Flask, request, redirect, url_for, jsonify, render_template, send_file
import re
import requests
import urllib.parse
from pyproj import Transformer
from datetime import datetime
from dateutil.relativedelta import relativedelta
from collections import defaultdict
import os
import logging
import pandas as pd
import math
import io
import ssl
from requests.adapters import HTTPAdapter

app = Flask(__name__)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- SSL 오류 해결을 위한 커스텀 어댑터 및 세션 ---
class CustomHttpAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        context = ssl.create_default_context()
        context.set_ciphers('DEFAULT:@SECLEVEL=1')
        kwargs['ssl_context'] = context
        return super().init_poolmanager(*args, **kwargs)

session = requests.Session()
session.mount('https://', CustomHttpAdapter())
# ----------------------------------------------------

# API Keys
KAKAO_API_KEY = "7c5ffe1b2f9e318d2bfa882a539bb429"
AIRKOREA_SERVICE_KEY = "tlBcA73yJuLT1PSGixHpbHwLcINQEVtZ0g5xfd2E5/+qZUSmPK1hSFACjbw+pauS2glnKPhOPUcniVoBRkGfpA=="

# CSV 데이터 로드
HISTORICAL_DATA_FILE = 'annual_pm_averages.csv'
historical_data = None
try:
    logger.info(f"'{HISTORICAL_DATA_FILE}' 파일을 로드합니다.")
    try:
        historical_data = pd.read_csv(HISTORICAL_DATA_FILE, encoding='utf-8')
    except UnicodeDecodeError:
        logger.warning("UTF-8 로딩 실패. cp949로 재시도합니다.")
        historical_data = pd.read_csv(HISTORICAL_DATA_FILE, encoding='cp949')
    logger.info("CSV 파일 로드 성공.")
except FileNotFoundError:
    logger.error(f"경고: '{HISTORICAL_DATA_FILE}' 파일을 찾을 수 없습니다.")
except Exception as e:
    logger.error(f"'{HISTORICAL_DATA_FILE}' 파일 로드 중 오류: {e}")

# 유틸리티 함수들
def safe_round(value, decimals=1):
    if value is None or (isinstance(value, str) and value.strip() in ['', '-', 'N/A', 'nan', 'NaN', 'null', 'None']):
        return '-'
    try:
        return round(float(value), decimals)
    except (ValueError, TypeError):
        return '-'

def format_distance(distance_km):
    if distance_km is None or distance_km == '':
        return "측정불가"
    try:
        dist_km = float(distance_km)
        dist_m = dist_km * 1000
        if dist_m < 1000:
            return f"{int(round(dist_m))}m"
        else:
            return f"{dist_km:.1f}km"
    except (ValueError, TypeError) as e:
        logger.error(f"거리 포맷팅 오류: {e}")
        return "측정불가"

def format_month_display(month_str):
    """YYYYMM 형식을 YYYY-MM 형식으로 변환"""
    if not month_str or len(str(month_str)) != 6:
        return month_str
    try:
        month_str = str(month_str)
        year = month_str[:4]
        month = month_str[4:6]
        return f"{year}-{month}"
    except (ValueError, TypeError):
        return month_str

def preprocess_address(address: str) -> str:
    address = address.strip()
    address = re.sub(r'\s+', ' ', address)
    address = re.sub(r'[(),."\'`]', '', address)
    address = re.sub(r'\s+\d{1,4}(?:층|호|동)\s*$', '', address)
    return address

def is_valid_road_address(address: str) -> bool:
    pattern = r"^[가-힣]+\s[가-힣]+\s[가-힣]+\s[가-힣0-9]+(?:로|길)\s?\d{1,3}(?:-\d{1,3})?$"
    return bool(re.match(pattern, address.strip()))

def convert_to_tm(lat: float, lon: float):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True)
    x, y = transformer.transform(lon, lat)
    return x, y

def _mean(values):
    nums = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(nums) / len(nums) if nums else None

def _to_float(x):
    try:
        if x is None: return None
        s = str(x).strip()
        if s in ["", "-", "NA", "null", "None"]: return None
        return float(s)
    except (ValueError, TypeError):
        return None

def get_grade_label(grade: str) -> str:
    grade_map = {'1': '좋음', '2': '보통', '3': '나쁨', '4': '매우나쁨'}
    return grade_map.get(str(grade), "N/A")

def get_nearby_stations_with_network(tmX, tmY, limit=3):
    msr_url = "http://apis.data.go.kr/B552584/MsrstnInfoInqireSvc/getNearbyMsrstnList"
    msr_params = {"serviceKey": AIRKOREA_SERVICE_KEY, "returnType": "json", "tmX": tmX, "tmY": tmY, "ver": "1.0"}
    try:
        r = session.get(msr_url, params=msr_params, timeout=5)
        r.raise_for_status()
        items = r.json().get("response", {}).get("body", {}).get("items", []) or []
        detailed = []
        for it in items[:limit]:
            st_name = it.get("stationName")
            distance_km = it.get("tm")
            network_type = get_station_network_type(st_name)
            detailed.append({
                "stationName": st_name,
                "addr": it.get("addr"),
                "distance": distance_km,
                "network_type": network_type
            })
        return detailed
    except Exception as e:
        logger.error(f"측정소 조회 오류: {e}")
        return []

def get_station_network_type(station_name: str):
    if not station_name: return None
    url = "http://apis.data.go.kr/B552584/MsrstnInfoInqireSvc/getMsrstnList"
    params = {"serviceKey": AIRKOREA_SERVICE_KEY, "returnType": "json", "stationName": station_name}
    try:
        r = session.get(url, params=params, timeout=5)
        r.raise_for_status()
        items = r.json().get("response", {}).get("body", {}).get("items", []) or []
        return items[0].get("mangName") if items else None
    except Exception:
        return None

def get_realtime_pm(station_name: str):
    url = "http://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getMsrstnAcctoRltmMesureDnsty"
    p = {"serviceKey": AIRKOREA_SERVICE_KEY, "stationName": station_name, "dataTerm": "DAILY", "ver": "1.3", "pageNo": "1", "numOfRows": "1", "returnType": "json"}
    try:
        r = session.get(url, params=p, timeout=10)
        r.raise_for_status()
        response_text = r.text.strip()
        if not response_text or not response_text.startswith('{'):
            logger.error(f"실시간 API 유효하지 않은 JSON: {response_text[:100]}...")
            return None
        data = r.json()
        items = data.get("response", {}).get("body", {}).get("items", []) or []
        if not items: return None
        it = items[0]
        return {
            "timestamp": it.get("dataTime"),
            "pm10_ug_m3": safe_round(it.get("pm10Value"), 1),
            "pm2_5_ug_m3": safe_round(it.get("pm25Value"), 1),
            "pm10_category": get_grade_label(it.get("pm10Grade")),
            "pm2_5_category": get_grade_label(it.get("pm25Grade"))
        }
    except Exception as e:
        logger.error(f"실시간 데이터 조회 오류: {station_name} - {e}")
        return None

def _get_monthly_stats_from_api(station_name: str, begin_mm: str, end_mm: str):
    try:
        url = "http://apis.data.go.kr/B552584/ArpltnStatsSvc/getMsrstnAcctoRMmrg"
        p = {"serviceKey": AIRKOREA_SERVICE_KEY, "returnType": "json", "inqBginMm": begin_mm, "inqEndMm": end_mm, "msrstnName": station_name, "pageNo": "1", "numOfRows": "120"}
        r = session.get(url, params=p, timeout=10)
        r.raise_for_status()
        response_text = r.text.strip()
        if not response_text or not response_text.startswith('{'):
            logger.error(f"유효하지 않은 JSON 응답: {response_text[:100]}...")
            return []
        data = r.json() or {}
        header = data.get("response", {}).get("header", {})
        if header and header.get("resultCode") != "00":
            logger.error(f"AirKorea API Error - Code: {header.get('resultCode')}, Message: {header.get('resultMsg')}")
            return []
        items = data.get("response", {}).get("body", {}).get("items", []) or []
        if not items: return []
        logger.info(f"✅ API에서 '{station_name}' 데이터 {len(items)}건 조회 성공.")
        return [{"stationName": it.get("msrstnName") or station_name, "month": it.get("msurMm"), "pm10_avg": _to_float(it.get("pm10Value")), "pm25_avg": _to_float(it.get("pm25Value"))} for it in items]
    except Exception as e:
        logger.error(f"API 조회 실패 (_get_monthly_stats_from_api): {station_name} - {e}")
        return []

def _get_monthly_stats_from_csv(station_name: str, months_to_find: set):
    if historical_data is None or historical_data.empty or not months_to_find:
        return []
    try:
        df_station = historical_data[historical_data['측정소명'] == station_name].copy()
        if df_station.empty: return []
        df_station['yyyymm'] = df_station['년'].astype(str) + df_station['월'].astype(str).str.zfill(2)
        df_filtered = df_station[df_station['yyyymm'].isin(months_to_find)]
        csv_results = []
        for _, row in df_filtered.iterrows():
            csv_results.append({
                "stationName": row['측정소명'], "month": row['yyyymm'],
                "pm10_avg": _to_float(row.get('PM10')), "pm25_avg": _to_float(row.get('PM2.5'))
            })
        return csv_results
    except Exception as e:
        logger.error(f"CSV 데이터 처리 중 오류 발생 (_get_monthly_stats_from_csv): {e}")
        return []

def get_monthly_stats(station_name: str, begin_mm: str, end_mm: str):
    expected_months = set()
    try:
        current_dt = datetime.strptime(begin_mm, "%Y%m")
        end_dt = datetime.strptime(end_mm, "%Y%m")
        while current_dt <= end_dt:
            expected_months.add(current_dt.strftime("%Y%m"))
            current_dt += relativedelta(months=1)
    except ValueError:
        logger.error(f"오류: 날짜 형식 변환 실패 begin='{begin_mm}', end='{end_mm}'")
        return []
    api_data = _get_monthly_stats_from_api(station_name, begin_mm, end_mm)
    retrieved_months = {item['month'] for item in api_data}
    missing_months = expected_months - retrieved_months
    csv_data = []
    if missing_months:
        logger.info(f"'{station_name}'의 누락된 {len(missing_months)}개월 데이터를 CSV에서 찾습니다: {sorted(list(missing_months))}")
        csv_data = _get_monthly_stats_from_csv(station_name, missing_months)
    combined_data = api_data + csv_data
    combined_data.sort(key=lambda x: x.get('month', ''), reverse=True)
    return combined_data

def aggregate_annual_from_monthly(monthly_rows):
    bucket = defaultdict(lambda: {"pm10": [], "pm25": []})
    for row in monthly_rows:
        y = (row.get("month") or "")[:4]
        if y:
            if row.get("pm10_avg") is not None: bucket[y]["pm10"].append(row["pm10_avg"])
            if row.get("pm25_avg") is not None: bucket[y]["pm25"].append(row["pm25_avg"])
    return [{"year": y, "pm10_avg": _mean(v["pm10"]), "pm25_avg": _mean(v["pm25"])} for y, v in sorted(bucket.items())]

def prepare_chart_data(monthly_data, yearly_data):
    colors = ["#007bff", "#28a745", "#fd7e14", "#dc3545", "#6f42c1"]
    chart = {"monthly": {}, "yearly": {}}
    station_colors = {}
    unique_stations = list(set([item["stationName"] for item in monthly_data + yearly_data]))
    for idx, station in enumerate(unique_stations):
        station_colors[station] = colors[idx % len(colors)]
    for it in monthly_data:
        st = it["stationName"]
        if st not in chart["monthly"]:
            chart["monthly"][st] = {"labels": [], "pm10_data": [], "pm25_data": [], "color": station_colors.get(st, "#666")}
        chart["monthly"][st]["labels"].append(it["month_label"])
        chart["monthly"][st]["pm10_data"].append(it["pm10_avg"])
        chart["monthly"][st]["pm25_data"].append(it["pm25_avg"])
    for it in yearly_data:
        st = it["stationName"]
        if st not in chart["yearly"]:
            chart["yearly"][st] = {"labels": [], "pm10_data": [], "pm25_data": [], "color": station_colors.get(st, "#666")}
        chart["yearly"][st]["labels"].append(str(it["year"]))
        chart["yearly"][st]["pm10_data"].append(it["pm10_avg"])
        chart["yearly"][st]["pm25_data"].append(it["pm25_avg"])
    return chart

@app.route("/", methods=["GET"])
def index():
    q = request.args.get("q", "")
    error = request.args.get("error", "")
    return render_template('index.html', q=q, error=error)

@app.route("/search", methods=["POST"])
def search():
    q = (request.form.get("q") or "").strip()
    if not q:
        return redirect(url_for("index", error="주소/장소명을 입력하세요."))
    return redirect(url_for("air_quality_view", q=q))

@app.route("/air-quality", methods=["GET"])
def air_quality_view():
    raw_query = (request.args.get("q") or "").strip()
    if not raw_query:
        return redirect(url_for("index", error="주소/장소명을 입력하세요."))
    q = preprocess_address(raw_query)
    try:
        if is_valid_road_address(q):
            search_type = "도로명 주소"
            url = "https://dapi.kakao.com/v2/local/search/address.json"
        else:
            search_type = "장소명(키워드)"
            url = "https://dapi.kakao.com/v2/local/search/keyword.json"
        headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
        resp = session.get(url, headers=headers, params={"query": q}, timeout=6)
        resp.raise_for_status()
        docs = resp.json().get("documents", [])
        if not docs:
            return redirect(url_for("index", error=f"'{raw_query}'에 대한 검색 결과가 없습니다."))

        first = docs[0]
        display_address = first.get("road_address_name") or first.get("address_name") or "-"
        lat, lon = float(first.get("y")), float(first.get("x"))
        place_name = first.get("place_name")
        tmX, tmY = convert_to_tm(lat, lon)
        stations = get_nearby_stations_with_network(tmX, tmY, limit=3)
        if not stations:
            return redirect(url_for("index", error="가까운 측정소를 찾을 수 없습니다."))

        colors = ["#007bff", "#28a745", "#fd7e14", "#dc3545", "#6f42c1"]
        realtime_data = []
        for idx, s in enumerate(stations):
            rt = get_realtime_pm(s["stationName"]) or {}
            combined_data = {**s, **rt, "color": colors[idx % len(colors)]}
            combined_data["distance_display"] = format_distance(s.get("distance"))
            realtime_data.append(combined_data)

        today = datetime.today()
        end_mm_api = today.strftime("%Y%m")
        begin_mm_monthly = (today - relativedelta(months=12)).strftime("%Y%m")
        begin_mm_yearly = (today - relativedelta(months=60)).strftime("%Y%m")

        monthly_data, yearly_data = [], []
        for s in stations:
            station_name = s["stationName"]
            try:
                monthly_rows = get_monthly_stats(station_name, begin_mm_monthly, end_mm_api)
                for row in monthly_rows:
                    if row.get("month"):
                        # 월별 데이터에 표시용 형식 추가
                        monthly_data.append({
                            "stationName": station_name, 
                            "month_label": format_month_display(row["month"]),  # YYYY-MM 형식으로 표시
                            "month_raw": row["month"],  # 원본 YYYYMM 형식 (정렬용)
                            "pm10_avg": safe_round(row.get("pm10_avg"), 1), 
                            "pm25_avg": safe_round(row.get("pm25_avg"), 1)
                        })
                yearly_monthly_rows = get_monthly_stats(station_name, begin_mm_yearly, end_mm_api)
                annual_rows = aggregate_annual_from_monthly(yearly_monthly_rows)
                current_year = today.year
                for row in annual_rows:
                    year = int(row["year"])
                    if current_year - 4 <= year <= current_year:
                        pm10, pm25 = safe_round(row.get("pm10_avg"), 1), safe_round(row.get("pm25_avg"), 1)
                        if pm10 != '-' and pm25 != '-':
                            yearly_data.append({"stationName": station_name, "year": year, "pm10_avg": pm10, "pm25_avg": pm25})
            except Exception as e:
                logger.error(f"{station_name} 데이터 조회 중 오류: {e}")
                continue

        chart_data = prepare_chart_data(monthly_data, yearly_data)

        monthly_df_map, yearly_df_map = {}, {}
        stations_with_display = []
        for s in stations:
            station_with_display = s.copy()
            station_with_display["distance_display"] = format_distance(s.get("distance"))
            stations_with_display.append(station_with_display)
            st = s["stationName"]
            m_rows = [x for x in monthly_data if x["stationName"] == st]
            y_rows = [x for x in yearly_data if x["stationName"] == st]
            monthly_df_map[st] = pd.DataFrame(m_rows)
            yearly_df_map[st] = pd.DataFrame(y_rows)

        return render_template(
            'result.html',
            raw_query=raw_query,
            search_type=search_type,
            place_name=place_name,
            address=display_address,
            realtime=realtime_data,
            chart_data=chart_data,
            stations_with_display=stations_with_display,
            monthly_tables={s["stationName"]: monthly_df_map[s["stationName"]].to_dict(orient="records") for s in stations},
            yearly_tables={s["stationName"]: yearly_df_map[s["stationName"]].to_dict(orient="records") for s in stations}
        )
    except Exception as e:
        logger.error(f"대기질 조회 오류: {e}")
        return redirect(url_for("index", error=f"데이터 조회 중 오류가 발생했습니다: {str(e)}"))

@app.route("/download/<path:station>/<dtype>", methods=["GET"])
def download_station_csv(station, dtype):
    try:
        if dtype not in ("monthly", "yearly"):
            return jsonify({"error": "invalid type"}), 400
        today = datetime.today()
        end_mm = today.strftime("%Y%m")
        if dtype == "monthly":
            months = max(1, min(int(request.args.get("months", 12)), 60))
            begin_mm = (today - relativedelta(months=months)).strftime("%Y%m")
            rows = get_monthly_stats(station, begin_mm, end_mm)
            # CSV 다운로드에서도 YYYY-MM 형식 사용
            df_data = [{"연월": format_month_display(r["month"]), "PM10(㎍/㎥)": safe_round(r.get("pm10_avg"), 1), "PM2.5(㎍/㎥)": safe_round(r.get("pm25_avg"), 1)} for r in rows if r.get("month")]
            df = pd.DataFrame(df_data).sort_values('연월', ascending=False)
            filename = f"{station}_monthly_{months}months.csv"
        else:
            years = max(1, min(int(request.args.get("years", 5)), 10))
            months_to_fetch = years * 12 + 12
            begin_mm = (today - relativedelta(months=months_to_fetch)).strftime("%Y%m")
            monthly_rows = get_monthly_stats(station, begin_mm, end_mm)
            annual_rows = aggregate_annual_from_monthly(monthly_rows)
            current_year = today.year
            df_data = [{"연도": int(r["year"]), "PM10(㎍/㎥)": safe_round(r.get("pm10_avg"), 1), "PM2.5(㎍/㎥)": safe_round(r.get("pm25_avg"), 1)} for r in annual_rows if current_year - years + 1 <= int(r["year"]) <= current_year]
            df = pd.DataFrame(df_data).sort_values('연도', ascending=False)
            filename = f"{station}_yearly_{years}years.csv"
        buf = io.StringIO()
        df.to_csv(buf, index=False, encoding="utf-8-sig")
        mem = io.BytesIO(buf.getvalue().encode("utf-8-sig"))
        mem.seek(0)
        return send_file(mem, mimetype="text/csv", as_attachment=True, download_name=filename)
    except Exception as e:
        logger.error(f"CSV 다운로드 오류: {e}")
        return jsonify({"error": "download failed"}), 500

@app.route("/health", methods=["GET"])
def health():
    status = {"status": "ok", "timestamp": datetime.utcnow().isoformat()}
    status["csv_loaded"] = historical_data is not None and not historical_data.empty
    if status["csv_loaded"]:
        status["csv_records"] = len(historical_data)
    try:
        test_url = "http://apis.data.go.kr/B552584/MsrstnInfoInqireSvc/getNearbyMsrstnList"
        test_params = {"serviceKey": AIRKOREA_SERVICE_KEY, "returnType": "json", "tmX": 200000, "tmY": 450000, "ver": "1.0"}
        r = session.get(test_url, params=test_params, timeout=10)
        r.raise_for_status()
        response_text = r.text.strip()
        if not response_text or not response_text.startswith('{'):
            status.update({"api_status": "error", "api_error": f"Invalid JSON response: {response_text[:100]}..."})
        else:
            data = r.json()
            header = data.get("response", {}).get("header", {})
            if header.get("resultCode") == "00":
                status.update({"api_status": "ok", "api_message": "API connection successful"})
            else:
                status.update({"api_status": "error", "api_error": f"API Error Code: {header.get('resultCode')}, Message: {header.get('resultMsg')}"})
    except Exception as e:
        status.update({"api_status": "error", "api_error": str(e)})
    
    http_status = 200 if status.get("api_status") == "ok" else 503
    return jsonify(status), http_status

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)