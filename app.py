import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from typing import Dict, List, Optional, Tuple

# =========================
# Page Config
# =========================
st.set_page_config(page_title="AI-ONLABS Marketing Audit MVP", layout="wide")
st.title("🚀 AI-ONLABS Marketing Audit MVP")
st.caption("Media-only audit supported. Internal / Initiative data are optional.")

# =========================
# Constants
# =========================
MEDIA_REQUIRED_MIN = ["date", "campaign", "spend", "impressions", "clicks"]

MEDIA_ALIASES = {
    "date": ["date", "day", "report_date"],
    "campaign": ["campaign", "campaign_name", "campaign name"],
    "channel": ["channel", "source", "platform", "media_source"],
    "adset": ["adset", "ad_set", "ad group", "adgroup", "ad_group"],
    "ad": ["ad", "creative", "ad_name", "ad name"],
    "keyword": ["keyword", "keyword_text", "search_keyword"],
    "search_term": ["search_term", "search term", "query", "search_query"],
    "spend": ["spend", "cost", "amount_spent"],
    "impressions": ["impressions", "impr"],
    "clicks": ["clicks", "click"],
    "conversions": ["conversions", "conv", "purchases", "purchase"],
    "revenue": ["revenue", "conversion_value", "sales", "purchase_value", "value"],
}

INTERNAL_ALIASES = {
    "date": ["date", "day", "report_date"],
    "campaign": ["campaign", "campaign_name", "campaign name"],
    "channel": ["channel", "source", "platform"],
    "sessions": ["sessions", "session"],
    "product_views": ["product_views", "product_view", "view_item", "views"],
    "add_to_cart": ["add_to_cart", "atc", "addtocart"],
    "checkout": ["checkout", "begin_checkout"],
    "purchase": ["purchase", "purchases", "transactions"],
    "revenue": ["revenue", "sales", "purchase_value", "value"],
}

INITIATIVE_ALIASES = {
    "initiative_name": ["initiative_name", "initiative", "test_name", "experiment_name"],
    "start_date": ["start_date", "start", "launch_date"],
    "end_date": ["end_date", "end"],
    "target_campaign": ["target_campaign", "campaign", "campaign_name"],
    "target_channel": ["target_channel", "channel", "source"],
    "description": ["description", "memo", "note"],
}

BRAND_TERMS_DEFAULT = ["brand", "official", "company"]
CATEGORY_TERMS_DEFAULT = ["wallet", "bag", "backpack", "tote", "shoulder bag", "crossbody", "pouch", "card case"]
INFO_TERMS_DEFAULT = ["how", "what", "best", "review", "reviews", "compare", "comparison", "cheap", "meaning"]
COMPETITOR_TERMS_DEFAULT = ["amazon", "rakuten", "zozo", "temu", "shein"]
NOISE_TERMS_DEFAULT = ["free", "download", "job", "求人", "中古", "used", "repair"]

# =========================
# Utilities
# =========================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def safe_divide(num, den):
    if isinstance(den, pd.Series):
        return np.where(den > 0, num / den, 0)
    return num / den if den > 0 else 0

def apply_aliases(df: pd.DataFrame, alias_map: Dict[str, List[str]]) -> pd.DataFrame:
    df = normalize_columns(df)
    rename_map = {}
    current_cols = set(df.columns)

    for standard_name, aliases in alias_map.items():
        if standard_name in current_cols:
            continue
        for alias in aliases:
            if alias in current_cols:
                rename_map[alias] = standard_name
                break

    return df.rename(columns=rename_map)

def validate_required_columns(df: pd.DataFrame, required_cols: List[str]) -> List[str]:
    return [col for col in required_cols if col not in df.columns]

def parse_dates(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    df = df.copy()
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
    return df

def convert_numeric_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df

def detect_text_column(df: pd.DataFrame, primary: str, secondary: str) -> Optional[str]:
    if primary in df.columns:
        return primary
    if secondary in df.columns:
        return secondary
    return None

def status_color(action: str) -> str:
    action = str(action).upper()
    if "SCALE" in action:
        return "#d4edda"
    if "OPTIMIZE" in action or "HOLD" in action or "MEASUREMENT CHECK" in action:
        return "#fff3cd"
    if "REDUCE" in action or "KILL" in action:
        return "#f8d7da"
    if "INSUFFICIENT" in action:
        return "#e2e3e5"
    return ""

def style_action(val):
    return f"background-color: {status_color(val)}"

# =========================
# Core KPI Logic
# =========================
def calculate_media_kpi(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["CTR"] = safe_divide(df["clicks"], df["impressions"])
    df["CPC"] = safe_divide(df["spend"], df["clicks"])

    if "conversions" in df.columns:
        df["CVR"] = safe_divide(df["conversions"], df["clicks"])
        df["CPA"] = safe_divide(df["spend"], df["conversions"])
    else:
        df["CVR"] = np.nan
        df["CPA"] = np.nan

    if "revenue" in df.columns:
        df["ROAS"] = safe_divide(df["revenue"], df["spend"])
    else:
        df["ROAS"] = np.nan

    return df

def calculate_internal_kpi(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "sessions" in df.columns and "product_views" in df.columns:
        df["View Rate"] = safe_divide(df["product_views"], df["sessions"])
    if "sessions" in df.columns and "add_to_cart" in df.columns:
        df["ATC Rate"] = safe_divide(df["add_to_cart"], df["sessions"])
    if "add_to_cart" in df.columns and "checkout" in df.columns:
        df["Checkout Rate"] = safe_divide(df["checkout"], df["add_to_cart"])
    if "sessions" in df.columns and "purchase" in df.columns:
        df["Purchase Rate"] = safe_divide(df["purchase"], df["sessions"])
    if "sessions" in df.columns and "revenue" in df.columns:
        df["Revenue per Session"] = safe_divide(df["revenue"], df["sessions"])
    return df

# =========================
# Audit Logic
# =========================
def assign_action(row: pd.Series, thresholds: Dict[str, float]) -> str:
    impressions = row.get("impressions", 0)
    clicks = row.get("clicks", 0)
    spend = row.get("spend", 0)
    conversions = row.get("conversions", 0)
    ctr = row.get("CTR", 0)
    cvr = row.get("CVR", np.nan)
    cpa = row.get("CPA", np.nan)
    cpc = row.get("CPC", 0)

    if impressions < thresholds["min_impressions"] or clicks < thresholds["min_clicks"]:
        return "INSUFFICIENT DATA"

    if pd.isna(cvr):
        if ctr >= thresholds["high_ctr"] and cpc <= thresholds["max_cpc"]:
            return "OPTIMIZE"
        if ctr <= thresholds["low_ctr"] and spend >= thresholds["reduce_spend_limit"]:
            return "REDUCE"
        return "HOLD"

    if clicks >= thresholds["kill_clicks"] and conversions <= 0 and spend >= thresholds["reduce_spend_limit"]:
        return "KILL"

    if cpc > thresholds["max_cpc"] and conversions <= 0 and spend >= thresholds["reduce_spend_limit"]:
        return "KILL"

    if conversions > 0 and cpa > thresholds["max_cpa"] and spend >= thresholds["reduce_spend_limit"]:
        return "REDUCE"

    if ctr >= thresholds["high_ctr"] and cvr >= thresholds["good_cvr"] and cpa <= thresholds["max_cpa"]:
        return "SCALE"

    if ctr >= thresholds["high_ctr"] and (cvr < thresholds["good_cvr"] or cpa > thresholds["max_cpa"]):
        return "OPTIMIZE"

    if ctr <= thresholds["low_ctr"] and spend >= thresholds["reduce_spend_limit"]:
        return "REDUCE"

    return "HOLD"

def classify_search_term(term: str, brand_terms: List[str], category_terms: List[str], info_terms: List[str], competitor_terms: List[str], noise_terms: List[str]) -> str:
    if not isinstance(term, str) or term.strip() == "":
        return "Unknown"

    t = term.lower().strip()

    if any(x in t for x in brand_terms):
        return "Brand"
    if any(x in t for x in competitor_terms):
        return "Competitor"
    if any(x in t for x in noise_terms):
        return "Noise"
    if any(x in t for x in info_terms):
        return "Informational"
    if any(x in t for x in category_terms):
        return "Category"
    return "Other"

def assign_search_term_action(row: pd.Series, thresholds: Dict[str, float]) -> str:
    cost = row.get("spend", 0)
    clicks = row.get("clicks", 0)
    conversions = row.get("conversions", 0)
    ctr = row.get("CTR", 0)
    cvr = row.get("CVR", 0)
    term_class = row.get("term_class", "Other")

    if clicks < thresholds["min_clicks_term"]:
        return "HOLD"
    if term_class == "Noise" and cost >= thresholds["term_cost_remove"] and conversions <= 0:
        return "REMOVE"
    if term_class == "Informational" and cost >= thresholds["term_cost_remove"] and conversions <= 0:
        return "REMOVE"
    if term_class == "Brand":
        return "KEEP"
    if conversions > 0 or cvr >= thresholds["term_good_cvr"]:
        return "SCALE"
    if ctr >= thresholds["high_ctr"] and conversions <= 0 and cost >= thresholds["term_cost_remove"]:
        return "OPTIMIZE"
    return "KEEP"

# =========================
# Preparation Functions
# =========================
def prepare_media_df(df: pd.DataFrame) -> pd.DataFrame:
    df = apply_aliases(df, MEDIA_ALIASES)
    missing = validate_required_columns(df, MEDIA_REQUIRED_MIN)
    if missing:
        raise ValueError(f"Media CSV missing required columns: {missing}")

    df = parse_dates(df, "date")
    numeric_cols = ["spend", "impressions", "clicks", "conversions", "revenue"]
    df = convert_numeric_columns(df, numeric_cols)

    if "channel" not in df.columns:
        df["channel"] = "Unknown"

    return df

def prepare_internal_df(df: pd.DataFrame) -> pd.DataFrame:
    df = apply_aliases(df, INTERNAL_ALIASES)
    if "date" in df.columns:
        df = parse_dates(df, "date")
    numeric_cols = ["sessions", "product_views", "add_to_cart", "checkout", "purchase", "revenue"]
    df = convert_numeric_columns(df, numeric_cols)
    return df

def prepare_initiative_df(df: pd.DataFrame) -> pd.DataFrame:
    df = apply_aliases(df, INITIATIVE_ALIASES)
    if "start_date" not in df.columns:
        raise ValueError("Initiative CSV must include start_date")
    df = parse_dates(df, "start_date")
    if "end_date" in df.columns:
        df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    else:
        df["end_date"] = pd.NaT
    if "initiative_name" not in df.columns:
        df["initiative_name"] = "Initiative"
    return df

def aggregate_campaign_kpis(media_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in ["spend", "impressions", "clicks", "conversions", "revenue"] if c in media_df.columns]
    agg = media_df.groupby(["channel", "campaign"], dropna=False)[numeric_cols].sum().reset_index()
    agg = calculate_media_kpi(agg)
    return agg

def build_daily_trend(media_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in ["spend", "impressions", "clicks", "conversions", "revenue"] if c in media_df.columns]
    trend = media_df.groupby("date", dropna=False)[numeric_cols].sum().reset_index()
    trend = calculate_media_kpi(trend)
    return trend

def compare_recent_windows(daily_df: pd.DataFrame, metric_cols: List[str], recent_days: int = 7) -> pd.DataFrame:
    if daily_df.empty:
        return pd.DataFrame()

    max_date = daily_df["date"].max()
    recent_start = max_date - pd.Timedelta(days=recent_days - 1)
    prev_end = recent_start - pd.Timedelta(days=1)
    prev_start = prev_end - pd.Timedelta(days=recent_days - 1)

    recent = daily_df[(daily_df["date"] >= recent_start) & (daily_df["date"] <= max_date)]
    prev = daily_df[(daily_df["date"] >= prev_start) & (daily_df["date"] <= prev_end)]

    rows = []
    for metric in metric_cols:
        if metric not in daily_df.columns:
            continue
        recent_val = recent[metric].mean() if not recent.empty else np.nan
        prev_val = prev[metric].mean() if not prev.empty else np.nan
        change_pct = ((recent_val - prev_val) / prev_val) if pd.notna(prev_val) and prev_val not in [0, 0.0] else np.nan
        rows.append({
            "metric": metric,
            f"recent_{recent_days}d_avg": recent_val,
            f"previous_{recent_days}d_avg": prev_val,
            "change_pct": change_pct,
        })
    return pd.DataFrame(rows)

def build_search_term_audit(media_df: pd.DataFrame, thresholds: Dict[str, float], term_lists: Dict[str, List[str]]) -> pd.DataFrame:
    term_col = detect_text_column(media_df, "search_term", "keyword")
    if term_col is None:
        return pd.DataFrame()

    group_cols = [term_col]
    if "campaign" in media_df.columns:
        group_cols.append("campaign")
    if "channel" in media_df.columns:
        group_cols.append("channel")

    numeric_cols = [c for c in ["spend", "impressions", "clicks", "conversions", "revenue"] if c in media_df.columns]
    term_df = media_df.groupby(group_cols, dropna=False)[numeric_cols].sum().reset_index()
    term_df = term_df.rename(columns={term_col: "term"})
    term_df = calculate_media_kpi(term_df)
    
    # 결측치 방어를 위한 str 형변환
    term_df["term"] = term_df["term"].fillna("").astype(str)
    
    term_df["term_class"] = term_df["term"].apply(
        lambda x: classify_search_term(
            x,
            term_lists["brand_terms"],
            term_lists["category_terms"],
            term_lists["info_terms"],
            term_lists["competitor_terms"],
            term_lists["noise_terms"],
        )
    )
    term_df["term_action"] = term_df.apply(lambda x: assign_search_term_action(x, thresholds), axis=1)
    return term_df.sort_values(["spend", "clicks"], ascending=[False, False])

def build_funnel_summary(internal_df: pd.DataFrame) -> pd.DataFrame:
    if internal_df.empty:
        return pd.DataFrame()
    agg = internal_df.sum(numeric_only=True)
    rows = [
        {"step": "Sessions", "value": agg.get("sessions", 0)},
        {"step": "Product Views", "value": agg.get("product_views", 0)},
        {"step": "Add to Cart", "value": agg.get("add_to_cart", 0)},
        {"step": "Checkout", "value": agg.get("checkout", 0)},
        {"step": "Purchase", "value": agg.get("purchase", 0)},
    ]
    funnel = pd.DataFrame(rows)
    # 변경된 부분: fillna()를 제거하고 직접 첫 번째 행에 1.0(100%)을 넣어줍니다.
    funnel["from_prev_rate"] = safe_divide(funnel["value"], funnel["value"].shift(1))
    funnel.loc[0, "from_prev_rate"] = 1.0 
    return funnel

def initiative_pre_post_analysis(media_df: pd.DataFrame, initiative_df: pd.DataFrame, pre_days: int = 7, post_days: int = 7) -> pd.DataFrame:
    if media_df.empty or initiative_df.empty:
        return pd.DataFrame()

    out = []
    for _, row in initiative_df.iterrows():
        start_date = row["start_date"]
        initiative_name = row.get("initiative_name", "Initiative")
        target_campaign = row.get("target_campaign", None)
        target_channel = row.get("target_channel", None)

        subset = media_df.copy()
        if pd.notna(target_campaign) and "campaign" in subset.columns:
            subset = subset[subset["campaign"] == target_campaign]
        if pd.notna(target_channel) and "channel" in subset.columns:
            subset = subset[subset["channel"] == target_channel]

        pre = subset[(subset["date"] >= start_date - pd.Timedelta(days=pre_days)) & (subset["date"] < start_date)]
        post = subset[(subset["date"] >= start_date) & (subset["date"] < start_date + pd.Timedelta(days=post_days))]

        def metric_mean(df: pd.DataFrame, metric: str):
            if df.empty or metric not in df.columns:
                return np.nan
            return df[metric].mean()

        pre_kpi = calculate_media_kpi(pre.groupby("date")[ [c for c in ["spend", "impressions", "clicks", "conversions", "revenue"] if c in pre.columns] ].sum().reset_index()) if not pre.empty else pd.DataFrame()
        post_kpi = calculate_media_kpi(post.groupby("date")[ [c for c in ["spend", "impressions", "clicks", "conversions", "revenue"] if c in post.columns] ].sum().reset_index()) if not post.empty else pd.DataFrame()

        for metric in ["spend", "clicks", "CTR", "CPC", "CVR", "CPA", "conversions", "ROAS"]:
            pre_val = metric_mean(pre_kpi, metric)
            post_val = metric_mean(post_kpi, metric)
            change_pct = ((post_val - pre_val) / pre_val) if pd.notna(pre_val) and pre_val not in [0, 0.0] else np.nan
            out.append({
                "initiative_name": initiative_name,
                "metric": metric,
                f"pre_{pre_days}d_avg": pre_val,
                f"post_{post_days}d_avg": post_val,
                "change_pct": change_pct,
                "target_campaign": target_campaign,
                "target_channel": target_channel,
            })
    return pd.DataFrame(out)

def generate_summary(campaign_df: pd.DataFrame, trend_compare_df: pd.DataFrame, term_df: pd.DataFrame) -> List[str]:
    messages = []

    if not campaign_df.empty:
        kill_count = (campaign_df["action"] == "KILL").sum()
        scale_count = (campaign_df["action"] == "SCALE").sum()
        optimize_count = (campaign_df["action"] == "OPTIMIZE").sum()
        messages.append(f"캠페인 기준: SCALE {scale_count}개 / OPTIMIZE {optimize_count}개 / KILL {kill_count}개")

        high_waste = campaign_df[(campaign_df["spend"] > 0) & (campaign_df.get("conversions", 0) <= 0)].sort_values("spend", ascending=False)
        if not high_waste.empty:
            top = high_waste.iloc[0]
            messages.append(f"가장 큰 낭비 후보: {top['campaign']} (spend {top['spend']:,.0f})")

    if not trend_compare_df.empty:
        bad = trend_compare_df[trend_compare_df["metric"].isin(["CTR", "conversions", "CPA"])]
        for _, r in bad.iterrows():
            if pd.notna(r["change_pct"]):
                if r["metric"] in ["CTR", "conversions"] and r["change_pct"] < -0.2:
                    messages.append(f"최근 추이 악화: {r['metric']}가 전 7일 대비 {r['change_pct']:.1%} 감소")
                if r["metric"] == "CPA" and r["change_pct"] > 0.2:
                    messages.append(f"최근 추이 악화: CPA가 전 7일 대비 {r['change_pct']:.1%} 상승")

    if not term_df.empty:
        remove_df = term_df[term_df["term_action"] == "REMOVE"].sort_values("spend", ascending=False)
        if not remove_df.empty:
            messages.append(f"제외 우선 후보 search term 수: {len(remove_df)}개")

    if not messages:
        messages.append("현재 필터 기준으로 뚜렷한 리스크 신호가 크지 않습니다.")

    return messages[:5]

# =========================
# Sidebar Controls
# =========================
st.sidebar.header("📁 1. Data Upload")
media_file = st.sidebar.file_uploader("Media CSV (required)", type=["csv"])
internal_file = st.sidebar.file_uploader("Internal CSV (optional)", type=["csv"])
initiative_file = st.sidebar.file_uploader("Initiative CSV (optional)", type=["csv"])

st.sidebar.header("⚙️ 2. Thresholds")
thresholds = {
    "high_ctr": st.sidebar.slider("High CTR (%)", 1.0, 20.0, 5.0, 0.5) / 100,
    "low_ctr": st.sidebar.slider("Low CTR (%)", 0.1, 10.0, 3.0, 0.1) / 100,
    "good_cvr": st.sidebar.slider("Good CVR (%)", 0.1, 20.0, 2.0, 0.1) / 100,
    "max_cpa": st.sidebar.number_input("Max CPA", value=5000.0, step=100.0),
    "max_cpc": st.sidebar.number_input("Max CPC", value=1000.0, step=50.0),
    "reduce_spend_limit": st.sidebar.number_input("Minimum spend for decision", value=500.0, step=100.0),
    "min_impressions": st.sidebar.number_input("Minimum impressions", value=100.0, step=50.0),
    "min_clicks": st.sidebar.number_input("Minimum clicks", value=10.0, step=1.0),
    "kill_clicks": st.sidebar.number_input("Kill after clicks with zero conv", value=30.0, step=1.0),
    "min_clicks_term": st.sidebar.number_input("Minimum term clicks", value=5.0, step=1.0),
    "term_cost_remove": st.sidebar.number_input("Term cost threshold for REMOVE", value=1000.0, step=100.0),
    "term_good_cvr": st.sidebar.slider("Good term CVR (%)", 0.1, 20.0, 2.0, 0.1) / 100,
}

st.sidebar.header("🧠 3. Search Term Rules")
brand_terms = [x.strip().lower() for x in st.sidebar.text_area("Brand terms (comma-separated)", value=",".join(BRAND_TERMS_DEFAULT)).split(",") if x.strip()]
category_terms = [x.strip().lower() for x in st.sidebar.text_area("Category terms", value=",".join(CATEGORY_TERMS_DEFAULT)).split(",") if x.strip()]
info_terms = [x.strip().lower() for x in st.sidebar.text_area("Informational terms", value=",".join(INFO_TERMS_DEFAULT)).split(",") if x.strip()]
competitor_terms = [x.strip().lower() for x in st.sidebar.text_area("Competitor terms", value=",".join(COMPETITOR_TERMS_DEFAULT)).split(",") if x.strip()]
noise_terms = [x.strip().lower() for x in st.sidebar.text_area("Noise terms", value=",".join(NOISE_TERMS_DEFAULT)).split(",") if x.strip()]

term_lists = {
    "brand_terms": brand_terms,
    "category_terms": category_terms,
    "info_terms": info_terms,
    "competitor_terms": competitor_terms,
    "noise_terms": noise_terms,
}

# =========================
# Main App
# =========================
if not media_file:
    st.info("👈 Please upload a Media CSV file to start the audit.")
    st.stop()

try:
    media_raw = pd.read_csv(media_file)
    media_df = prepare_media_df(media_raw)
except Exception as e:
    st.error(f"Failed to load Media CSV: {e}")
    st.stop()

internal_df = pd.DataFrame()
initiative_df = pd.DataFrame()

if internal_file:
    try:
        internal_raw = pd.read_csv(internal_file)
        internal_df = prepare_internal_df(internal_raw)
    except Exception as e:
        st.warning(f"Internal CSV could not be loaded: {e}")

if initiative_file:
    try:
        initiative_raw = pd.read_csv(initiative_file)
        initiative_df = prepare_initiative_df(initiative_raw)
    except Exception as e:
        st.warning(f"Initiative CSV could not be loaded: {e}")

# =========================
# Global Filters
# =========================
st.sidebar.header("🔍 4. Filters")
channel_options = sorted(media_df["channel"].dropna().astype(str).unique().tolist()) if "channel" in media_df.columns else []
selected_channels = st.sidebar.multiselect("Channel", channel_options, default=channel_options)

campaign_options = sorted(media_df["campaign"].dropna().astype(str).unique().tolist())
selected_campaigns = st.sidebar.multiselect("Campaign", campaign_options, default=campaign_options)

min_date = media_df["date"].min().date()
max_date = media_df["date"].max().date()
date_range = st.sidebar.date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

if len(date_range) != 2:
    st.error("Please choose both start and end dates.")
    st.stop()

filtered_media = media_df[
    media_df["channel"].astype(str).isin(selected_channels)
    & media_df["campaign"].astype(str).isin(selected_campaigns)
    & (media_df["date"] >= pd.to_datetime(date_range[0]))
    & (media_df["date"] <= pd.to_datetime(date_range[1]))
].copy()

if filtered_media.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

kpi_media = calculate_media_kpi(filtered_media)
campaign_df = aggregate_campaign_kpis(filtered_media)
campaign_df["action"] = campaign_df.apply(lambda x: assign_action(x, thresholds), axis=1)

daily_df = build_daily_trend(filtered_media)
trend_compare_df = compare_recent_windows(daily_df, ["spend", "clicks", "CTR", "conversions", "CPA", "CVR", "ROAS"], recent_days=7)
term_df = build_search_term_audit(filtered_media, thresholds, term_lists)

if not internal_df.empty:
    filtered_internal = internal_df.copy()
    if "date" in filtered_internal.columns:
        filtered_internal = filtered_internal[
            (filtered_internal["date"] >= pd.to_datetime(date_range[0]))
            & (filtered_internal["date"] <= pd.to_datetime(date_range[1]))
        ]
    if "channel" in filtered_internal.columns and selected_channels:
        filtered_internal = filtered_internal[filtered_internal["channel"].astype(str).isin(selected_channels)]
    if "campaign" in filtered_internal.columns and selected_campaigns:
        filtered_internal = filtered_internal[filtered_internal["campaign"].astype(str).isin(selected_campaigns)]
    filtered_internal = calculate_internal_kpi(filtered_internal)
else:
    filtered_internal = pd.DataFrame()

if not initiative_df.empty:
    initiative_impact_df = initiative_pre_post_analysis(filtered_media, initiative_df)
else:
    initiative_impact_df = pd.DataFrame()

summary_lines = generate_summary(campaign_df, trend_compare_df, term_df)

# =========================
# Overview
# =========================
st.subheader("📌 Executive Summary")
for line in summary_lines:
    st.write(f"- {line}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Spend", f"{filtered_media['spend'].sum():,.0f}")
c2.metric("Clicks", f"{filtered_media['clicks'].sum():,.0f}")
c3.metric("CTR", f"{safe_divide(filtered_media['clicks'].sum(), filtered_media['impressions'].sum()):.2%}")
if "conversions" in filtered_media.columns:
    c4.metric("Conversions", f"{filtered_media['conversions'].sum():,.0f}")
else:
    c4.metric("Campaigns", f"{filtered_media['campaign'].nunique():,}")

# =========================
# Campaign Audit
# =========================
st.markdown("---")
st.subheader("📊 Campaign Audit")
show_cols = [c for c in ["channel", "campaign", "action", "spend", "impressions", "clicks", "CTR", "CPC", "conversions", "CVR", "CPA", "revenue", "ROAS"] if c in campaign_df.columns]
styled_campaign = campaign_df[show_cols].style.map(style_action, subset=["action"]).format({
    "spend": "{:,.0f}",
    "impressions": "{:,.0f}",
    "clicks": "{:,.0f}",
    "CTR": "{:.2%}",
    "CPC": "{:,.2f}",
    "conversions": "{:,.0f}",
    "CVR": "{:.2%}",
    "CPA": "{:,.2f}",
    "revenue": "{:,.0f}",
    "ROAS": "{:.2f}",
}, na_rep="-")
st.dataframe(styled_campaign, use_container_width=True)

csv_campaign = campaign_df.to_csv(index=False).encode("utf-8-sig")
st.download_button("Download Campaign Audit CSV", data=csv_campaign, file_name="campaign_audit.csv", mime="text/csv")

# =========================
# Trends
# =========================
st.markdown("---")
st.subheader("📈 Time-series Trends")
trend_metric_options = [c for c in ["spend", "clicks", "CTR", "CPC", "conversions", "CVR", "CPA", "revenue", "ROAS"] if c in daily_df.columns]
trend_metric = st.selectbox("Metric", trend_metric_options, index=0)

fig = px.line(daily_df, x="date", y=trend_metric, markers=True, title=f"Daily {trend_metric} Trend")
if not initiative_df.empty and "start_date" in initiative_df.columns:
    for _, row in initiative_df.iterrows():
        if pd.notna(row["start_date"]):
            fig.add_vline(x=row["start_date"], line_dash="dash", line_color="red")
st.plotly_chart(fig, use_container_width=True)

if not trend_compare_df.empty:
    st.write("**Recent 7-day vs Previous 7-day**")
    st.dataframe(
        trend_compare_df.style.format({
            "recent_7d_avg": "{:,.2f}",
            "previous_7d_avg": "{:,.2f}",
            "change_pct": "{:.1%}",
        }, na_rep="-"),
        use_container_width=True,
    )

# =========================
# Opportunities & Risks
# =========================
st.markdown("---")
st.subheader("🏆 Opportunities & Risks")
left, right = st.columns(2)

with left:
    st.write("**Top Opportunities**")
    opp_cols = [c for c in ["campaign", "CTR", "CVR", "CPA", "spend", "action"] if c in campaign_df.columns]
    opp_df = campaign_df[campaign_df["action"].isin(["SCALE", "OPTIMIZE"])].sort_values(["CTR", "CVR"], ascending=[False, False])
    st.dataframe(opp_df[opp_cols].head(10).style.format({"CTR": "{:.2%}", "CVR": "{:.2%}", "CPA": "{:,.2f}", "spend": "{:,.0f}"}, na_rep="-"), use_container_width=True)

with right:
    st.write("**Top Risks**")
    risk_cols = [c for c in ["campaign", "spend", "CTR", "CPC", "conversions", "CPA", "action"] if c in campaign_df.columns]
    risk_df = campaign_df[campaign_df["action"].isin(["REDUCE", "KILL"])].sort_values("spend", ascending=False)
    st.dataframe(risk_df[risk_cols].head(10).style.format({"CTR": "{:.2%}", "CPC": "{:,.2f}", "CPA": "{:,.2f}", "spend": "{:,.0f}"}, na_rep="-"), use_container_width=True)

# =========================
# Search Term Audit
# =========================
st.markdown("---")
st.subheader("🔎 Search Term / Keyword Audit")
if term_df.empty:
    st.info("No search_term or keyword column found in Media data.")
else:
    term_filter = st.selectbox("Term Action Filter", ["All", "REMOVE", "KEEP", "SCALE", "OPTIMIZE"], index=0)
    display_term_df = term_df.copy()
    if term_filter != "All":
        display_term_df = display_term_df[display_term_df["term_action"] == term_filter]

    display_term_cols = [c for c in ["channel", "campaign", "term", "term_class", "term_action", "spend", "clicks", "CTR", "conversions", "CVR", "CPA"] if c in display_term_df.columns]
    st.dataframe(
        display_term_df[display_term_cols].head(200).style.format({
            "spend": "{:,.0f}",
            "clicks": "{:,.0f}",
            "CTR": "{:.2%}",
            "conversions": "{:,.0f}",
            "CVR": "{:.2%}",
            "CPA": "{:,.2f}",
        }, na_rep="-"),
        use_container_width=True,
    )

    csv_terms = display_term_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Download Search Term Audit CSV", data=csv_terms, file_name="search_term_audit.csv", mime="text/csv")

# =========================
# Internal Funnel
# =========================
st.markdown("---")
st.subheader("🛒 Internal Funnel (Optional)")
if filtered_internal.empty:
    st.info("Internal CSV not uploaded or no matching data after filters.")
else:
    funnel_df = build_funnel_summary(filtered_internal)
    metric_cols = st.columns(4)
    metric_cols[0].metric("Sessions", f"{filtered_internal['sessions'].sum():,.0f}" if "sessions" in filtered_internal.columns else "-")
    metric_cols[1].metric("ATC", f"{filtered_internal['add_to_cart'].sum():,.0f}" if "add_to_cart" in filtered_internal.columns else "-")
    metric_cols[2].metric("Purchase", f"{filtered_internal['purchase'].sum():,.0f}" if "purchase" in filtered_internal.columns else "-")
    if "ATC Rate" in filtered_internal.columns:
        metric_cols[3].metric("ATC Rate", f"{safe_divide(filtered_internal['add_to_cart'].sum(), filtered_internal['sessions'].sum()):.2%}")
    else:
        metric_cols[3].metric("ATC Rate", "-")

    st.dataframe(
        funnel_df.style.format({"value": "{:,.0f}", "from_prev_rate": "{:.2%}"}, na_rep="-"),
        use_container_width=True,
    )

# =========================
# Initiative Impact
# =========================
st.markdown("---")
st.subheader("🧪 Initiative Impact (Optional)")
if initiative_impact_df.empty:
    st.info("Initiative CSV not uploaded or insufficient initiative data.")
else:
    st.dataframe(
        initiative_impact_df.style.format({
            "pre_7d_avg": "{:,.2f}",
            "post_7d_avg": "{:,.2f}",
            "change_pct": "{:.1%}",
        }, na_rep="-"),
        use_container_width=True,
    )

    csv_initiative = initiative_impact_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Download Initiative Impact CSV", data=csv_initiative, file_name="initiative_impact.csv", mime="text/csv")

# =========================
# Raw Data Preview
# =========================
st.markdown("---")
with st.expander("Preview normalized Media data"):
    st.dataframe(filtered_media.head(100), use_container_width=True)
