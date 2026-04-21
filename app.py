import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from typing import Dict, List, Optional, Tuple

# =========================
# Page Config
# =========================
st.set_page_config(page_title="AI-ONLABS Marketing Audit MVP", layout="wide")
st.title("🚀 AI-ONLABS Marketing Audit MVP (Google Ads ver)")
st.caption("Google Ads Raw 리포트(일본어/일별) 전용 대시보드입니다.")

# =========================
# Constants & Mapping
# =========================
# Google Ads 일본어 리포트 컬럼 매핑
GOOGLE_ADS_MAPPING = {
    "日": "date",
    "キャンペーン": "campaign",
    "キャンペーン タイプ": "channel",
    "広告グループ": "adset",
    "費用": "spend",
    "表示回数": "impressions",
    "クリック数": "clicks",
    "コンバージョン": "conversions",
    "コンバージョン値": "revenue",
    "検索語句": "search_term",
    "キーワード": "keyword"
}

INTERNAL_ALIASES = {
    "date": ["date", "day", "report_date"],
    "campaign": ["campaign", "campaign_name", "campaign name"],
    "sessions": ["sessions", "session"],
    "product_views": ["product_views", "product_view", "view_item", "views"],
    "add_to_cart": ["add_to_cart", "atc", "addtocart"],
    "checkout": ["checkout", "begin_checkout"],
    "purchase": ["purchase", "purchases", "transactions"],
    "revenue": ["revenue", "sales", "purchase_value", "value"],
}

INITIATIVE_ALIASES = {
    "initiative_name": ["initiative_name", "initiative", "test_name"],
    "start_date": ["start_date", "start", "launch_date"],
    "end_date": ["end_date", "end"],
    "target_campaign": ["target_campaign", "campaign", "campaign_name"],
}

BRAND_TERMS_DEFAULT = ["brand", "official", "company", "ganzo"]
CATEGORY_TERMS_DEFAULT = ["wallet", "bag", "backpack", "tote", "crossbody", "財布", "バッグ"]
INFO_TERMS_DEFAULT = ["how", "what", "best", "review", "reviews", "compare", "おすすめ", "比較"]
COMPETITOR_TERMS_DEFAULT = ["amazon", "rakuten", "zozo", "temu", "shein", "楽天"]
NOISE_TERMS_DEFAULT = ["free", "download", "job", "求人", "中古", "used", "repair"]

# =========================
# Utilities & Preprocessing
# =========================
def safe_divide(num, den):
    if isinstance(den, pd.Series):
        return np.where(den > 0, num / den, 0)
    return num / den if den > 0 else 0

def clean_google_ads_report(file) -> pd.DataFrame:
    # Google Ads 리포트는 상단 2줄이 설명이므로 skiprows=2 적용
    df = pd.read_csv(file, skiprows=2)
    
    # 일본어 컬럼명을 영문 표준명으로 변경
    df = df.rename(columns=GOOGLE_ADS_MAPPING)
    
    # 쉼표(,) 및 ' -- ' 결측치 제거 후 숫자형 변환
    numeric_cols = ["spend", "impressions", "clicks", "conversions", "revenue"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", "", regex=False).str.replace(" --", "0", regex=False).str.replace("--", "0", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            
    # 날짜 처리
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        
    return df

def prepare_internal_df(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().lower() for c in df.columns]
    rename_map = {}
    for std, aliases in INTERNAL_ALIASES.items():
        for alias in aliases:
            if alias in df.columns:
                rename_map[alias] = std
                break
    df = df.rename(columns=rename_map)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    numeric_cols = ["sessions", "product_views", "add_to_cart", "checkout", "purchase", "revenue"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df

def status_color(action: str) -> str:
    action = str(action).upper()
    if "SCALE" in action: return "#d4edda"
    if "OPTIMIZE" in action or "HOLD" in action: return "#fff3cd"
    if "REDUCE" in action or "KILL" in action: return "#f8d7da"
    if "INSUFFICIENT" in action: return "#e2e3e5"
    return ""

def style_action(val):
    return f"background-color: {status_color(val)}"

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

def classify_search_term(term: str, term_lists: Dict[str, List[str]]) -> str:
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
st.sidebar.header("📁 1. Google Ads Data Upload")
st.sidebar.caption("※ 반드시 '日(Day)' 세그먼트를 포함하여 다운로드하세요.")
campaign_file = st.sidebar.file_uploader("1. キャンペーン レポート (필수)", type=["csv"])
search_term_file = st.sidebar.file_uploader("2. 検索語句 レポート (선택)", type=["csv"])

st.sidebar.header("📁 2. Additional Data (Optional)")
internal_file = st.sidebar.file_uploader("Internal CSV", type=["csv"])
initiative_file = st.sidebar.file_uploader("Initiative CSV", type=["csv"])

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
    "term_cost_remove": st.sidebar.number_input("검색어 제외 기준 비용", value=1000.0, step=100.0),
}

st.sidebar.header("🧠 4. Search Term Rules")
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
if not campaign_file:
    st.info("👈 좌측에서 'キャンペーン レポート (일별 데이터 포함)' CSV를 업로드해주세요.")
    st.stop()

try:
    media_df = clean_google_ads_report(campaign_file)
    if "date" not in media_df.columns:
        st.error("🚨 데이터에 '日(date)' 컬럼이 없습니다. Google Ads에서 일(Day) 세그먼트를 추가해 다시 다운로드 해주세요.")
        st.stop()
except Exception as e:
    st.error(f"리포트 로드 실패: {e}")
    st.stop()

# --- Global Filters ---
st.sidebar.header("🔍 5. Date Filter")
min_date, max_date = media_df["date"].min().date(), media_df["date"].max().date()
date_range = st.sidebar.date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

if len(date_range) != 2:
    st.stop()

filtered_media = media_df[
    (media_df["date"] >= pd.to_datetime(date_range[0])) & 
    (media_df["date"] <= pd.to_datetime(date_range[1]))
].copy()

# --- 1. Executive Summary & KPIs ---
st.subheader("📌 Executive Summary (Campaigns)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Spend", f"¥ {filtered_media['spend'].sum():,.0f}")
c2.metric("Total Clicks", f"{filtered_media['clicks'].sum():,.0f}")
c3.metric("Avg. CTR", f"{safe_divide(filtered_media['clicks'].sum(), filtered_media['impressions'].sum()):.2%}")
c4.metric("Total Conversions", f"{filtered_media.get('conversions', pd.Series([0])).sum():,.0f}")

# --- 2. Campaign Audit ---
st.markdown("---")
st.subheader("📊 Campaign Audit")
numeric_cols = [c for c in ["spend", "impressions", "clicks", "conversions", "revenue"] if c in filtered_media.columns]
campaign_df = filtered_media.groupby("campaign", dropna=False)[numeric_cols].sum().reset_index()
campaign_df = calculate_media_kpi(campaign_df)
campaign_df["action"] = campaign_df.apply(lambda x: assign_action(x, thresholds), axis=1)

show_cols = ["campaign", "action", "spend", "impressions", "clicks", "CTR", "CPC", "conversions", "CVR", "CPA", "ROAS"]
show_cols = [c for c in show_cols if c in campaign_df.columns]

st.dataframe(
    campaign_df[show_cols].style.map(style_action, subset=["action"]).format({
        "spend": "{:,.0f}", "impressions": "{:,.0f}", "clicks": "{:,.0f}",
        "CTR": "{:.2%}", "CPC": "{:,.2f}", "conversions": "{:,.0f}",
        "CVR": "{:.2%}", "CPA": "{:,.2f}", "ROAS": "{:.2f}"
    }, na_rep="-"), use_container_width=True
)

# --- 3. Trends ---
st.markdown("---")
st.subheader("📈 Daily Trends")
daily_df = filtered_media.groupby("date", dropna=False)[numeric_cols].sum().reset_index()
daily_df = calculate_media_kpi(daily_df)

trend_metric = st.selectbox("Select Metric", ["spend", "clicks", "CTR", "conversions", "CPA", "ROAS"])
if trend_metric in daily_df.columns:
    fig = px.line(daily_df, x="date", y=trend_metric, markers=True, title=f"Daily {trend_metric}")
    st.plotly_chart(fig, use_container_width=True)

# --- 4. Search Term Audit ---
if search_term_file:
    st.markdown("---")
    st.subheader("🔎 Search Term Audit")
    term_df_raw = clean_google_ads_report(search_term_file)
    
    if "search_term" in term_df_raw.columns:
        term_df_raw["search_term"] = term_df_raw["search_term"].fillna("").astype(str)
        t_cols = [c for c in ["spend", "impressions", "clicks", "conversions", "revenue"] if c in term_df_raw.columns]
        term_df = term_df_raw.groupby(["campaign", "search_term"], dropna=False)[t_cols].sum().reset_index()
        term_df = calculate_media_kpi(term_df)
        
        term_df["term_class"] = term_df["search_term"].apply(lambda x: classify_search_term(x, term_lists))
        
        # Simple term action logic
        def assign_term_action(row):
            if row['spend'] >= thresholds['term_cost_remove'] and row.get('conversions', 0) == 0: return "REMOVE"
            if row.get('conversions', 0) > 0: return "SCALE"
            return "KEEP"
            
        term_df["action"] = term_df.apply(assign_term_action, axis=1)
        term_df = term_df.sort_values("spend", ascending=False)
        
        t_show = [c for c in ["campaign", "search_term", "term_class", "action", "spend", "clicks", "CTR", "conversions", "CPA"] if c in term_df.columns]
        st.dataframe(term_df[t_show].style.map(style_action, subset=["action"]).format({
            "spend": "{:,.0f}", "clicks": "{:,.0f}", "CTR": "{:.2%}", "conversions": "{:,.0f}", "CPA": "{:,.2f}"
        }, na_rep="-"), use_container_width=True)
    else:
        st.warning("업로드된 検索語句 레포트에 '検索語句' 컬럼이 없습니다.")

# --- 5. Internal Funnel ---
if internal_file:
    st.markdown("---")
    st.subheader("🛒 Internal Funnel")
    internal_df = prepare_internal_df(pd.read_csv(internal_file))
    
    if not internal_df.empty:
        agg = internal_df.sum(numeric_only=True)
        rows = [
            {"step": "Sessions", "value": agg.get("sessions", 0)},
            {"step": "Product Views", "value": agg.get("product_views", 0)},
            {"step": "Add to Cart", "value": agg.get("add_to_cart", 0)},
            {"step": "Checkout", "value": agg.get("checkout", 0)},
            {"step": "Purchase", "value": agg.get("purchase", 0)},
        ]
        funnel = pd.DataFrame(rows)
        funnel["from_prev_rate"] = safe_divide(funnel["value"], funnel["value"].shift(1))
        funnel.loc[0, "from_prev_rate"] = 1.0
        
        st.dataframe(funnel.style.format({"value": "{:,.0f}", "from_prev_rate": "{:.2%}"}), use_container_width=True)
