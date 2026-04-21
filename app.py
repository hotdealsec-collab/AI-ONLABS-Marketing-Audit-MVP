import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

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
    "検索語句": "search_term",
    "キーワード": "keyword",
    "ランディング ページ": "landing_page",
    "最終ページ URL": "final_url",
    "広告見出し 1": "headline_1"
}

# GA4 데이터 컬럼 매핑 (영문/일문)
INTERNAL_ALIASES = {
    "campaign": ["campaign", "campaign_name", "campaign name", "キャンペーン"],
    "sessions": ["sessions", "session", "セッション"],
    "product_views": ["product_views", "product_view", "view_item", "views", "商品ビュー", "表示"],
    "add_to_cart": ["add_to_cart", "atc", "addtocart", "カートに追加", "カート追加"],
    "checkout": ["checkout", "begin_checkout", "チェックアウト"],
    "purchase": ["purchase", "purchases", "transactions", "購入", "コンバージョン"],
    "revenue": ["revenue", "sales", "purchase_value", "value", "収益", "総収益"],
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
    
    numeric_cols = ["spend", "impressions", "clicks", "conversions", "revenue"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", "", regex=False).str.replace(" --", "0", regex=False).str.replace("--", "0", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df

def prepare_internal_df(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = [str(c).strip().lower() for c in df.columns]
    rename_map = {}
    for std, aliases in INTERNAL_ALIASES.items():
        for alias in aliases:
            if alias in df.columns:
                rename_map[alias] = std
                break
    df = df.rename(columns=rename_map)
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
st.sidebar.header("📁 1. A/B Test Reports (Ads)")
st.sidebar.caption("기간 비교 또는 세팅 비교를 위해 업로드하세요.")
control_file = st.sidebar.file_uploader("1. キャンペーン (Control / 기준)", type=["csv"])
experiment_file = st.sidebar.file_uploader("2. キャンペーン (Experiment / 비교) - 선택", type=["csv"])

st.sidebar.header("📁 2. Detailed Reports (Optional)")
internal_file = st.sidebar.file_uploader("3. GA4 (Internal) 데이터", type=["csv"])
search_term_file = st.sidebar.file_uploader("4. 検索語句 (검색어)", type=["csv"])
adgroup_file = st.sidebar.file_uploader("5. 広告グループ (광고그룹)", type=["csv"])
ad_file = st.sidebar.file_uploader("6. 広告 (광고/소재)", type=["csv"])
lp_file = st.sidebar.file_uploader("7. ランディング ページ (랜딩페이지)", type=["csv"])

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
if not control_file:
    st.info("👈 좌측에서 기준이 되는 'キャンペーン レポート (Control)' CSV를 먼저 업로드해주세요.")
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

# 전체 캠페인 통합 (Control + Experiment)
combined_campaign = pd.concat([ctrl_df, exp_df], ignore_index=True) if not exp_df.empty else ctrl_df.copy()

# --- 1. Executive Summary (A/B Test Comparison) ---
st.subheader("📌 Executive Summary (Google Ads)")
if not exp_df.empty:
    st.caption("Experiment 리포트가 업로드되어 Control 대비 증감률(Delta)이 표시됩니다.")

def get_totals(df):
    spend = df['spend'].sum() if 'spend' in df.columns else 0
    clicks = df['clicks'].sum() if 'clicks' in df.columns else 0
    impr = df['impressions'].sum() if 'impressions' in df.columns else 0
    conv = df['conversions'].sum() if 'conversions' in df.columns else 0
    ctr = clicks / impr if impr > 0 else 0
    cpa = spend / conv if conv > 0 else 0
    return spend, clicks, ctr, conv, cpa

c_spend, c_clicks, c_ctr, c_conv, c_cpa = get_totals(ctrl_df)

if not exp_df.empty:
    e_spend, e_clicks, e_ctr, e_conv, e_cpa = get_totals(exp_df)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Spend", f"¥ {e_spend:,.0f}", f"{(e_spend - c_spend) / c_spend * 100:.1f}%" if c_spend > 0 else "0%", delta_color="inverse")
    col2.metric("Clicks", f"{e_clicks:,.0f}", f"{(e_clicks - c_clicks) / c_clicks * 100:.1f}%" if c_clicks > 0 else "0%")
    col3.metric("CTR", f"{e_ctr:.2%}", f"{(e_ctr - c_ctr) * 100:.2f}%p")
    col4.metric("Conversions", f"{e_conv:,.0f}", f"{(e_conv - c_conv) / c_conv * 100:.1f}%" if c_conv > 0 else "0%")
    
    cpa_delta = (e_cpa - c_cpa) / c_cpa * 100 if c_cpa > 0 else 0
    col5.metric("CPA", f"¥ {e_cpa:,.0f}", f"{cpa_delta:.1f}%" if c_cpa > 0 else "0%", delta_color="inverse")
else:
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Spend (Control)", f"¥ {c_spend:,.0f}")
    col2.metric("Clicks (Control)", f"{c_clicks:,.0f}")
    col3.metric("CTR (Control)", f"{c_ctr:.2%}")
    col4.metric("Conversions (Control)", f"{c_conv:,.0f}")
    col5.metric("CPA (Control)", f"¥ {c_cpa:,.0f}")

# --- 2. Campaign Audit ---
st.markdown("---")
st.subheader("📊 1. Campaign Audit (Ads)")
numeric_cols = [c for c in ["spend", "impressions", "clicks", "conversions", "revenue"] if c in combined_campaign.columns]
campaign_agg = combined_campaign.groupby(["Group", "campaign"], dropna=False)[numeric_cols].sum().reset_index()
campaign_agg = calculate_media_kpi(campaign_agg)
campaign_agg["action"] = campaign_agg.apply(lambda x: assign_action(x, thresholds), axis=1)

show_cols = ["Group", "campaign", "action", "spend", "impressions", "clicks", "CTR", "CPC", "conversions", "CVR", "CPA", "ROAS"]
show_cols = [c for c in show_cols if c in campaign_agg.columns]

st.dataframe(
    campaign_agg[show_cols].sort_values(by=["campaign", "Group"]).style.map(style_action, subset=["action"]).format({
        "spend": "{:,.0f}", "impressions": "{:,.0f}", "clicks": "{:,.0f}",
        "CTR": "{:.2%}", "CPC": "{:,.2f}", "conversions": "{:,.0f}",
        "CVR": "{:.2%}", "CPA": "{:,.2f}", "ROAS": "{:.2f}"
    }, na_rep="-"), use_container_width=True
)

# --- 3. Internal Funnel (GA4) ---
if internal_file:
    st.markdown("---")
    st.subheader("🛒 2. Internal Funnel (GA4)")
    internal_df = prepare_internal_df(internal_file)
    
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
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Sessions", f"{agg.get('sessions', 0):,.0f}")
        c2.metric("Total Purchases", f"{agg.get('purchase', 0):,.0f}")
        c3.metric("Session to Purchase Rate", f"{safe_divide(agg.get('purchase', 0), agg.get('sessions', 0)):.2%}")
        
        st.dataframe(funnel.style.format({"value": "{:,.0f}", "from_prev_rate": "{:.2%}"}, na_rep="-"), use_container_width=True)
        
        if "campaign" in internal_df.columns:
            st.markdown("**📌 Campaign Breakdown (GA4)**")
            camp_funnel = internal_df.groupby("campaign", dropna=False).sum(numeric_only=True).reset_index()
            camp_funnel["Session to Purchase"] = safe_divide(camp_funnel["purchase"], camp_funnel["sessions"])
            
            f_show = [c for c in ["campaign", "sessions", "product_views", "add_to_cart", "checkout", "purchase", "revenue", "Session to Purchase"] if c in camp_funnel.columns]
            st.dataframe(camp_funnel[f_show].sort_values("sessions", ascending=False).style.format({
                "sessions": "{:,.0f}", "product_views": "{:,.0f}", "add_to_cart": "{:,.0f}",
                "checkout": "{:,.0f}", "purchase": "{:,.0f}", "revenue": "{:,.0f}",
                "Session to Purchase": "{:.2%}"
            }, na_rep="-"), use_container_width=True)

# --- 4. Search Term Audit ---
if search_term_file:
    st.markdown("---")
    st.subheader("🔎 3. Search Term Audit")
    term_df_raw = clean_google_ads_report(search_term_file)
    
    if "search_term" in term_df_raw.columns:
        term_df_raw["search_term"] = term_df_raw["search_term"].fillna("").astype(str)
        t_cols = [c for c in ["spend", "impressions", "clicks", "conversions", "revenue"] if c in term_df_raw.columns]
        
        group_cols = ["search_term"]
        if "campaign" in term_df_raw.columns: group_cols.insert(0, "campaign")
            
        term_df = term_df_raw.groupby(group_cols, dropna=False)[t_cols].sum().reset_index()
        term_df = calculate_media_kpi(term_df)
        
        term_df["term_class"] = term_df["search_term"].apply(lambda x: classify_search_term(x, term_lists))
        
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

# --- 5. Ad Group Audit ---
if adgroup_file:
    st.markdown("---")
    st.subheader("📂 4. Ad Group Audit")
    ag_df_raw = clean_google_ads_report(adgroup_file)
    
    if "ad_group" in ag_df_raw.columns:
        ag_cols = [c for c in ["spend", "impressions", "clicks", "conversions", "revenue"] if c in ag_df_raw.columns]
        group_cols = ["ad_group"]
        if "campaign" in ag_df_raw.columns: group_cols.insert(0, "campaign")
            
        ag_df = ag_df_raw.groupby(group_cols, dropna=False)[ag_cols].sum().reset_index()
        ag_df = calculate_media_kpi(ag_df)
        ag_df["action"] = ag_df.apply(lambda x: assign_action(x, thresholds), axis=1)
        ag_df = ag_df.sort_values("spend", ascending=False)
        
        ag_show = [c for c in ["campaign", "ad_group", "action", "spend", "clicks", "CTR", "CPC", "conversions", "CPA", "ROAS"] if c in ag_df.columns]
        st.dataframe(ag_df[ag_show].style.map(style_action, subset=["action"]).format({
            "spend": "{:,.0f}", "clicks": "{:,.0f}", "CTR": "{:.2%}", "CPC": "{:,.2f}", "conversions": "{:,.0f}", "CPA": "{:,.2f}", "ROAS": "{:.2f}"
        }, na_rep="-"), use_container_width=True)

# --- 6. Ad / Creative Audit ---
if ad_file:
    st.markdown("---")
    st.subheader("🖼️ 5. Ad / Creative Audit")
    ad_df_raw = clean_google_ads_report(ad_file)
    
    ad_identifier = "headline_1" if "headline_1" in ad_df_raw.columns else ("final_url" if "final_url" in ad_df_raw.columns else None)
    
    if ad_identifier:
        ad_df_raw[ad_identifier] = ad_df_raw[ad_identifier].fillna("Unknown").astype(str)
        ad_cols = [c for c in ["spend", "impressions", "clicks", "conversions", "revenue"] if c in ad_df_raw.columns]
        
        group_cols = [ad_identifier]
        if "campaign" in ad_df_raw.columns: group_cols.insert(0, "campaign")
        if "ad_group" in ad_df_raw.columns: group_cols.insert(1, "ad_group")
            
        ad_df = ad_df_raw.groupby(group_cols, dropna=False)[ad_cols].sum().reset_index()
        ad_df = calculate_media_kpi(ad_df)
        ad_df = ad_df.sort_values("spend", ascending=False)
        
        ad_show = group_cols + [c for c in ["spend", "clicks", "CTR", "CPC", "conversions", "CPA"] if c in ad_df.columns]
        st.dataframe(ad_df[ad_show].style.format({
            "spend": "{:,.0f}", "clicks": "{:,.0f}", "CTR": "{:.2%}", "CPC": "{:,.2f}", "conversions": "{:,.0f}", "CPA": "{:,.2f}"
        }, na_rep="-"), use_container_width=True)

# --- 7. Landing Page Audit ---
if lp_file:
    st.markdown("---")
    st.subheader("🌐 6. Landing Page Audit")
    lp_df_raw = clean_google_ads_report(lp_file)
    
    if "landing_page" in lp_df_raw.columns:
        lp_df_raw["landing_page"] = lp_df_raw["landing_page"].fillna("Unknown").astype(str)
        lp_cols = [c for c in ["spend", "impressions", "clicks", "conversions", "revenue"] if c in lp_df_raw.columns]
        
        lp_df = lp_df_raw.groupby("landing_page", dropna=False)[lp_cols].sum().reset_index()
        lp_df = calculate_media_kpi(lp_df)
        lp_df = lp_df.sort_values("spend", ascending=False)
        
        lp_show = ["landing_page"] + [c for c in ["spend", "clicks", "CTR", "CPC", "conversions", "CPA", "ROAS"] if c in lp_df.columns]
        st.dataframe(lp_df[lp_show].style.format({
            "spend": "{:,.0f}", "clicks": "{:,.0f}", "CTR": "{:.2%}", "CPC": "{:,.2f}", "conversions": "{:,.0f}", "CPA": "{:,.2f}", "ROAS": "{:.2f}"
        }, na_rep="-"), use_container_width=True)
