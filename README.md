# 프로젝트 요약: 대기질 정보 조회 및 분석 웹 애플리케이션


## 📖 프로젝트 개요
이 프로젝트는 사용자가 입력한 주소나 장소명을 기반으로, 가장 가까운 측정소의 실시간 및 과거 대기질(미세먼지) 데이터를 조회하고 시각화하는 Flask 기반의 웹 애플리케이션입니다. 사용자는 간단한 검색만으로 현재 위치의 미세먼지 농도를 파악하고, 월별 및 연도별 데이터 추세를 차트와 표로 한눈에 비교 분석할 수 있습니다.

---

## ✨ 주요 기능
* 주소 및 장소명 기반 검색 (카카오 로컬 API 활용)
* 인근 측정소 최대 3곳의 정보 제공 (거리순)
* 실시간 대기질 정보 제공 (PM10, PM2.5 농도 및 등급)
* 과거 데이터 조회 및 시각화 (월별/연도별 추세 라인 차트)
* API 데이터 보완 (annual_pm_averages.csv 파일 활용)
* 월별, 연도별 데이터 CSV 파일 다운로드 기능
* 시스템 상태 확인 기능 (/health 엔드포인트)

---

## ⚙️ 기술 스택
* 백엔드: Python, Flask
* 프론트엔드: HTML, CSS, JavaScript
* 데이터 시각화: Chart.js
* 주요 라이브러리: requests, pandas, pyproj, python-dateutil

---

## 🚀 실행 방법

### 1. 사전 준비
* Python 3.x 설치
* `annual_pm_averages.csv` 데이터 파일 준비
* 카카오 및 공공데이터포털 API 키 발급

### 2. 프로젝트 설정
* `git clone https://github.com/ckdudwns/AirCleaner_final.git`
* `cd AirCleaner_New/localINFO_DGU`

### 3. 필수 라이브러리 설치
* 설치 명령어: `pip install Flask requests pandas pyproj python-dateutil`

### 4. API 키 설정
* `app.py` 파일 내 `KAKAO_API_KEY`와 `AIRKOREA_SERVICE_KEY` 변수에 발급받은 키 입력

### 5. 애플리케이션 실행
* 터미널에서 `python app.py` 명령어 실행
* 웹 브라우저에서 `http://127.0.0.1:5000` 주소로 접속

---

## 📋 필수 라이브러리 목록
* Flask: 웹 서버 프레임워크
* requests: 외부 API 통신
* pandas: CSV 파일 처리 및 데이터 가공
* pyproj: 좌표계 변환
* python-dateutil: 날짜 및 시간 계산
