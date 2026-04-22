import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List

# =========================
# Page Config
# =========================
st.set_page_config(page_title="AI-ONLABS Marketing Audit MVP", layout="wide")
st.title("🚀 AI-ONLABS Marketing Audit MVP (A/B Test + GA4)")
st.caption("Google Ads 및 GA4 리포트 전용 대시보드입니다. 모든 리포트는 일자(Day) 세그먼트 없이 누적으로 업로드하세요.")

# =========================
# Constants & Mapping
# =========================
GOOGLE_ADS_MAPPING = {
    "キャンペーン": "campaign",
    "キャンペーン タイプ": "channel",
    "広告グループ": "ad_group",
    "費用": "spend",
    "表示回数": "impressions",
    "クリック数": "clicks",
    "コンバージョン": "conversions",
    "コンバージョン値": "revenue",
    "キーワード": "keyword",
    "ランディング ページ": "landing_page",
}

# GA4 최신 일본어 컬럼 완벽 대응
INTERNAL_ALIASES = {
    "campaign": ["campaign", "campaign_name", "キャンペーン", "セッションのメインのキャンペーン", "セッションのメインのチャネル グループ（デフォルト チャネル グループ）"],
    "sessions": ["sessions", "session", "セッション"],
    "product_views": ["product_views", "view_item", "商品ビュー", "表示", "イベント数"],
    "add_to_cart": ["add_to_cart", "atc", "カートに追加", "カート追加"],
    "checkout": ["checkout", "begin_checkout", "チェックアウト"],
    "purchase": ["purchase", "transactions", "購入", "コンバージョン", "キーイベント"],
    "revenue": ["revenue", "sales", "value", "収益", "総収益", "合計収益"],
}

BRAND_TERMS_DEFAULT = ["brand", "official", "company", "ganzo"]
CATEGORY_TERMS_DEFAULT = ["wallet", "bag", "backpack", "tote", "crossbody", "財布", "バッグ"]
INFO_TERMS_DEFAULT = ["how", "what", "best", "review", "reviews", "compare", "おすすめ", "比較"]
COMPETITOR_TERMS_DEFAULT = ["amazon", "rakuten", "zozo", "temu", "shein", "楽天", "土屋鞄"]
NOISE_TERMS_DEFAULT = ["free", "download", "job", "求人", "中古", "used", "repair", "修理", "買取"]

# =========================
# Utilities & Preprocessing
# =========================
def safe_divide(num, den):
    if isinstance(den, pd.Series):
        return np.where(den > 0, num / den, 0)
    return num / den if den > 0 else 0

def clean_google_ads_report(file) -> pd.DataFrame:
    df = pd.read_csv(file, skiprows=2)
    df = df.rename(columns=GOOGLE_ADS_MAPPING)
    
    # '合計(Total)'이 포함된 구글 애즈 요약 행 제거
    mask = df.astype(str).apply(lambda x: x.str.contains("合計", na=False)).any(axis=1)
    df = df[~mask]

    # PyArrow Mixed Type Error 방지 (명시적 String 변환)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].fillna("Unknown").astype(str)
            
    # 숫자형 변환
    numeric_cols = ["spend", "impressions", "clicks", "conversions", "revenue"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", "", regex=False).str.replace(" --", "0", regex=False).str.replace("--", "0", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df

def prepare_internal_df(file) -> pd.DataFrame:
    # GA4의 상단 설명 메타데이터(#) 자동 무시
    df = pd.read_csv(file, comment='#')
    df.columns = [str(c).strip().lower() for c in df.columns]
    
    rename_map = {}
    for std, aliases in INTERNAL_ALIASES.items():
        for alias in aliases:
            if alias.lower() in df.columns:
                rename_map[alias.lower()] = std
                break
    df = df.rename(columns=rename_map)
    
    # 쉼표 제거 및 숫자 변환
    numeric_cols = ["sessions", "product_views", "add_to_cart", "checkout", "purchase", "revenue"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", "", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df

def status_color(action: str) -> str:
    action = str(action).upper()
    if "SCALE" in action or "KEEP" in action: return "#d4edda"
    if "OPTIMIZE" in action or "HOLD" in action: return "#fff3cd"
    if "REDUCE" in action or "KILL" in action or "REMOVE" in action: return "#f8d7da"
    if "INSUFFICIENT" in action: return "#e2e3e5"
    return ""

def style_action(val):
    return f"background-color: {status_color(val)}"

def display_table(df: pd.DataFrame, show_cols: List[str]):
    display_df = df[[c for c in show_cols if c in df.columns]].copy()
    styler = display_df.style
    
    if "action" in display_df.columns:
        styler = styler.map(style_action, subset=["action"])
        
    base_format = {
        "spend": "{:,.0f}", "impressions": "{:,.0f}", "clicks": "{:,.0f}",
        "CTR": "{:.2%}", "CPC": "{:,.2f}", "conversions": "{:,.0f}",
        "CVR": "{:.2%}", "CPA": "{:,.2f}", "ROAS": "{:.2f}",
        "sessions": "{:,.0f}", "product_views": "{:,.0f}", "add_to_cart": "{:,.0f}",
        "checkout": "{:,.0f}", "purchase": "{:,.0f}", "revenue": "{:,.0f}",
        "Session to Purchase": "{:.2%}"
    }
    valid_format = {k: v for k, v in base_format.items() if k in display_df.columns}
    st.dataframe(styler.format(valid_format, na_rep="-"), use_container_width=True)

# =========================
# Core KPI & Logic
# =========================
def calculate_media_kpi(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["CTR"] = safe_divide(df["clicks"], df["impressions"])
    df["CPC"] = safe_divide(df["spend"], df["clicks"])

    if "conversions" in df.columns:
        df["CVR"] = safe_divide(df["conversions"], df["clicks"])
        df["CPA"] = safe_divide(df["spend"], df["conversions"])
    else:
        df["CVR"], df["CPA"] = np.nan, np.nan

    if "revenue" in df.columns:
        df["ROAS"] = safe_divide(df["revenue"], df["spend"])
    else:
        df["ROAS"] = np.nan
    return df

def assign_action(row: pd.Series, thresholds: Dict[str, float]) -> str:
    impressions, clicks, spend = row.get("impressions", 0), row.get("clicks", 0), row.get("spend", 0)
    conversions, ctr, cvr, cpa, cpc = row.get("conversions", 0), row.get("CTR", 0), row.get("CVR", np.nan), row.get("CPA", np.nan), row.get("CPC", 0)

    if impressions < thresholds["min_impressions"] or clicks < thresholds["min_clicks"]: return "INSUFFICIENT DATA"

    if pd.isna(cvr):
        if ctr >= thresholds["high_ctr"] and cpc <= thresholds["max_cpc"]: return "OPTIMIZE"
        if ctr <= thresholds["low_ctr"] and spend >= thresholds["reduce_spend_limit"]: return "REDUCE"
        return "HOLD"

    if (clicks >= thresholds["kill_clicks"] or cpc > thresholds["max_cpc"]) and conversions <= 0 and spend >= thresholds["reduce_spend_limit"]: return "KILL"
    if conversions > 0 and cpa > thresholds["max_cpa"] and spend >= thresholds["reduce_spend_limit"]: return "REDUCE"
    if ctr >= thresholds["high_ctr"] and cvr >= thresholds["good_cvr"] and cpa <= thresholds["max_cpa"]: return "SCALE"
    if ctr >= thresholds["high_ctr"] and (cvr < thresholds["good_cvr"] or cpa > thresholds["max_cpa"]): return "OPTIMIZE"
    if ctr <= thresholds["low_ctr"] and spend >= thresholds["reduce_spend_limit"]: return "REDUCE"
    return "HOLD"

def classify_keyword(term: str, term_lists: Dict[str, List[str]]) -> str:
    if not isinstance(term, str) or term.strip() == "": return "Unknown"
    t = term.lower().strip()
    if any(x in t for x in term_lists["brand_terms"]): return "Brand"
    if any(x in t for x in term_lists["competitor_terms"]): return "Competitor"
    if any(x in t for x in term_lists["noise_terms"]): return "Noise"
    if any(x in t for x in term_lists["info_terms"]): return "Informational"
    if any(x in t for x in term_lists["category_terms"]): return "Category"
    return "Other"

# =========================
# Sidebar Controls
# =========================
st.sidebar.header("📁 1. A/B Test Reports (필수)")
st.sidebar.caption("전후 성과 비교를 위해 기준(Control)과 비교대상(Experiment) 캠페인 리포트를 업로드하세요.")
control_file = st.sidebar.file_uploader("1. キャンペーン (Control / 기준)", type=["csv"])
experiment_file = st.sidebar.file_uploader("2. キャンペーン (Experiment / 비교) - 선택", type=["csv"])

st.sidebar.header("📁 2. Detailed Reports (선택)")
adgroup_file = st.sidebar.file_uploader("3. 広告グループ (광고그룹)", type=["csv"])
keyword_file = st.sidebar.file_uploader("4. 検索キーワード (검색 키워드)", type=["csv"])
lp_file = st.sidebar.file_uploader("5. ランディング ページ (랜딩페이지)", type=["csv"])
internal_file = st.sidebar.file_uploader("6. GA4 (Internal) 데이터", type=["csv"])

st.sidebar.header("⚙️ 3. Action Thresholds")
thresholds = {
    "high_ctr": st.sidebar.slider("High CTR (%)", 1.0, 20.0, 5.0, 0.5) / 100,
    "low_ctr": st.sidebar.slider("Low CTR (%)", 0.1, 10.0, 3.0, 0.1) / 100,
    "good_cvr": st.sidebar.slider("Good CVR (%)", 0.1, 20.0, 2.0, 0.1) / 100,
    "max_cpa": st.sidebar.number_input("Max CPA", value=5000.0, step=100.0),
    "max_cpc": st.sidebar.number_input("Max CPC", value=1000.0, step=50.0),
    "reduce_spend_limit": st.sidebar.number_input("판단 최소 비용", value=1000.0, step=100.0),
    "min_impressions": st.sidebar.number_input("최소 노출수", value=100.0, step=50.0),
    "min_clicks": st.sidebar.number_input("최소 클릭수", value=10.0, step=1.0),
    "kill_clicks": st.sidebar.number_input("Kill 클릭 기준(전환 0)", value=30.0, step=1.0),
    "term_cost_remove": st.sidebar.number_input("키워드 제외 기준 비용", value=1000.0, step=100.0),
}

st.sidebar.header("🧠 4. Keyword Rules")
term_lists = {
    "brand_terms": [x.strip().lower() for x in st.sidebar.text_area("Brand terms", value=",".join(BRAND_TERMS_DEFAULT)).split(",") if x.strip()],
    "category_terms": [x.strip().lower() for x in st.sidebar.text_area("Category terms", value=",".join(CATEGORY_TERMS_DEFAULT)).split(",") if x.strip()],
    "info_terms": [x.strip().lower() for x in st.sidebar.text_area("Info terms", value=",".join(INFO_TERMS_DEFAULT)).split(",") if x.strip()],
    "competitor_terms": [x.strip().lower() for x in st.sidebar.text_area("Competitor terms", value=",".join(COMPETITOR_TERMS_DEFAULT)).split(",") if x.strip()],
    "noise_terms": [x.strip().lower() for x in st.sidebar.text_area("Noise terms", value=",".join(NOISE_TERMS_DEFAULT)).split(",") if x.strip()],
}

# =========================
# Main App
# =========================
if not control_file:
    st.info("👈 좌측 사이드바에서 기준이 되는 '1. キャンペーン レポート (Control)' CSV를 먼저 업로드해주세요.")
    st.stop()

try:
    ctrl_df = clean_google_ads_report(control_file)
    ctrl_df["Group"] = "Control"
except Exception as e:
    st.error(f"Control 리포트 로드 실패: {e}")
    st.stop()

exp_df = pd.DataFrame()
if experiment_file:
    try:
        exp_df = clean_google_ads_report(experiment_file)
        exp_df["Group"] = "Experiment"
    except Exception as e:
        st.warning(f"Experiment 리포트 로드 실패: {e}")

# 전체 캠페인 통합
combined_campaign = pd.concat([ctrl_df, exp_df], ignore_index=True) if not exp_df.empty else ctrl_df.copy()

# --- 1. Executive Summary (A/B Test Comparison) ---
st.subheader("📌 Executive Summary (Google Ads)")
if not exp_df.empty:
    st.caption("Experiment 리포트가 업로드되어 Control 대비 증감률(Delta)이 표시됩니다.")

def get_totals(df):
    spend = df['spend'].sum() if 'spend' in df.columns else 0
    clicks = df['clicks'].sum() if 'clicks' in df.columns else 0
    impr = df['impressions'].sum() if 'impressions' in df.columns else
