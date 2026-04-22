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

    # PyArrow Mixed Type Error 방지
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
    df = pd.read_csv(file, comment='#')
    df.columns = [str(c).strip().lower() for c in df.columns]
    
    rename_map = {}
    for std, aliases in INTERNAL_ALIASES.items():
        for alias in aliases:
            if alias.lower() in df.columns:
                rename_map[alias.lower()] = std
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
    if "SCALE" in action or "KEEP
