# -*- coding: utf-8 -*-
"""
新ペースロジック判定アプリ

同じフォルダに置く推奨ファイル:
- app.py
- ペース１０年.csv
- 調教師１０年.csv

予想用CSVは画面からアップロードします。
例: おためし.csv
"""

from __future__ import annotations

import io
import math
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="新ペースロジック判定アプリ", layout="wide")

APP_DIR = Path(__file__).resolve().parent
PACE_FILE_CANDIDATES = [
    "ペース１０年.csv", "ペース10年.csv", "pace_10years.csv", "pace.csv"
]
TRAINER_FILE_CANDIDATES = [
    "調教師１０年.csv", "調教師10年.csv", "trainer_10years.csv", "trainer.csv"
]

# ============================================================
# 基本ユーティリティ
# ============================================================

def read_csv_auto(src) -> pd.DataFrame:
    """cp932/utf-8-sig/utf-8を自動判定してCSVを読む。"""
    if src is None:
        return pd.DataFrame()
    if hasattr(src, "getvalue"):
        raw = src.getvalue()
        for enc in ["cp932", "utf-8-sig", "utf-8"]:
            try:
                return pd.read_csv(io.BytesIO(raw), encoding=enc)
            except Exception:
                pass
        return pd.read_csv(io.BytesIO(raw), encoding="cp932", errors="ignore")
    path = Path(src)
    for enc in ["cp932", "utf-8-sig", "utf-8"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path, encoding="cp932", errors="ignore")


def find_existing_file(candidates: list[str]) -> Optional[Path]:
    for name in candidates:
        p = APP_DIR / name
        if p.exists():
            return p
    return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename = {}
    for c in df.columns:
        cc = str(c).strip()
        if cc == "Ｒ":
            rename[c] = "R"
        elif cc == "芝・ダ":
            rename[c] = "芝ダ"
        elif cc == "芝ダート":
            rename[c] = "芝ダ"
        elif cc == "走破タイム":
            rename[c] = "走破時計"
        elif cc == "前PCI":
            rename[c] = "前走PCI"
        elif cc == "前走Ave-3F":
            rename[c] = "前走Ave3F"
        elif cc == "Ave-3F":
            rename[c] = "Ave3F"
        elif cc == "上り3F":
            rename[c] = "上がり3F"
        elif cc == "上り3F順":
            rename[c] = "上がり3F順"
        else:
            rename[c] = cc
    df = df.rename(columns=rename)
    return df


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def normalize_race_no(x):
    if pd.isna(x):
        return np.nan
    m = re.search(r"\d+", str(x))
    return int(m.group()) if m else np.nan


def normalize_surface(x: object) -> str:
    s = str(x).strip()
    if "ダ" in s:
        return "ダ"
    if "芝" in s:
        return "芝"
    return s


def finish_to_int(x) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    zmap = str.maketrans("０１２３４５６７８９", "0123456789")
    s = s.translate(zmap)
    m = re.search(r"\d+", s)
    return float(m.group()) if m else np.nan


def safe_div(a, b):
    try:
        if pd.isna(a) or pd.isna(b) or float(b) == 0:
            return np.nan
        return float(a) / float(b)
    except Exception:
        return np.nan


def percentile_scores(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """0〜100のパーセンタイル点。"""
    s = pd.to_numeric(values, errors="coerce")
    if s.notna().sum() <= 1:
        return pd.Series(50, index=s.index, dtype=float)
    rank = s.rank(method="average", pct=True, ascending=not higher_is_better)
    # ascending=Falseなら高い値が上位pct小さめになるので反転補正
    if higher_is_better:
        rank = s.rank(method="average", pct=True, ascending=True)
    else:
        rank = s.rank(method="average", pct=True, ascending=False)
    return (rank * 100).clip(0, 100)


def pace_type_from_rpci(x) -> str:
    try:
        v = float(x)
    except Exception:
        return "不明"
    if v <= 46:
        return "ハイ/持続"
    if v < 50:
        return "やや速い"
    if v < 54:
        return "標準〜やや上がり"
    return "スロー上がり"


def score_fit(value, target, scale, neutral=60.0) -> float:
    if pd.isna(value) or pd.isna(target):
        return neutral
    return float(np.clip(100 - abs(float(value) - float(target)) * scale, 0, 100))


# ============================================================
# 前処理
# ============================================================

def prep_common(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)
    if "R" in df.columns:
        df["R"] = df["R"].apply(normalize_race_no)
    for c in ["日付", "場所", "レース名", "馬名", "騎手", "調教師", "馬場状態"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    if "芝ダ" in df.columns:
        df["芝ダ"] = df["芝ダ"].apply(normalize_surface)
    for c in ["距離", "頭数", "馬番", "前走RPCI", "前走PCI", "前走Ave3F", "前走上3F地点差", "前走頭数"]:
        if c in df.columns:
            df[c] = to_num(df[c])
    for c in ["RPCI", "PCI", "Ave3F", "上3F地点差", "頭数", "馬番", "3角", "4角", "3角.1", "4角.1"]:
        if c in df.columns:
            df[c] = to_num(df[c])
    if "着順" in df.columns:
        df["着順数値"] = df["着順"].apply(finish_to_int)
    return df


def choose_prev_corner_col(df: pd.DataFrame, base: str) -> Optional[str]:
    candidates = [base, f"{base}.1"]
    for c in candidates:
        if c in df.columns and df[c].notna().sum() > 0:
            return c
    return None


# ============================================================
# 過去データから基準表を作る
# ============================================================

@st.cache_data(show_spinner=False)
def build_reference_tables(pace_df: pd.DataFrame, trainer_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pace = prep_common(pace_df)
    trainer = prep_common(trainer_df) if not trainer_df.empty else pd.DataFrame()

    required = ["場所", "芝ダ", "距離", "馬場状態"]
    for c in required:
        if c not in pace.columns:
            raise ValueError(f"ペース10年CSVに必要項目がありません: {c}")

    # 好走判定
    if "着順数値" not in pace.columns:
        pace["着順数値"] = np.nan
    pace["is_top3"] = pace["着順数値"].between(1, 3)

    # 4角率
    if "4角" in pace.columns:
        pace["四角率"] = pace.apply(lambda r: safe_div(r.get("4角"), r.get("頭数")), axis=1)
    elif "4角.1" in pace.columns:
        pace["四角率"] = pace.apply(lambda r: safe_div(r.get("4角.1"), r.get("頭数")), axis=1)
    else:
        pace["四角率"] = np.nan

    pace["ペース型"] = pace.get("RPCI", pd.Series(index=pace.index, dtype=float)).apply(pace_type_from_rpci)

    # 条件別基準: 3着内馬中心で作る
    top3 = pace[pace["is_top3"]].copy()
    group_cols = ["場所", "芝ダ", "距離", "馬場状態"]
    cond = top3.groupby(group_cols, dropna=False).agg(
        条件出走数=("馬名", "size"),
        基準RPCI=("RPCI", "mean"),
        基準PCI=("PCI", "mean"),
        基準Ave3F=("Ave3F", "mean"),
        基準上3F地点差=("上3F地点差", "mean"),
        基準四角率=("四角率", "mean"),
        基準上がり3F=("上がり3F", "mean") if "上がり3F" in top3.columns else ("馬名", "size"),
    ).reset_index()
    # fallback用: 馬場状態を抜いた条件
    cond2 = top3.groupby(["場所", "芝ダ", "距離"], dropna=False).agg(
        条件出走数2=("馬名", "size"),
        基準RPCI2=("RPCI", "mean"),
        基準PCI2=("PCI", "mean"),
        基準Ave3F2=("Ave3F", "mean"),
        基準上3F地点差2=("上3F地点差", "mean"),
        基準四角率2=("四角率", "mean"),
    ).reset_index()

    # 騎手: ペース型ごとの好走率をパーセンタイル化
    j = pace.dropna(subset=["騎手"]).groupby(["騎手", "ペース型"], dropna=False).agg(
        騎手騎乗数=("馬名", "size"),
        騎手複勝数=("is_top3", "sum"),
    ).reset_index()
    j["騎手複勝率"] = j["騎手複勝数"] / j["騎手騎乗数"].replace(0, np.nan)
    # 小サンプル補正
    base_rate = pace["is_top3"].mean() if len(pace) else 0.22
    k = 30
    j["騎手補正率"] = (j["騎手複勝数"] + base_rate * k) / (j["騎手騎乗数"] + k)
    j["騎手ペース点"] = j.groupby("ペース型")["騎手補正率"].transform(lambda s: percentile_scores(s, True))
    j.loc[j["騎手騎乗数"] < 10, "騎手ペース点"] = 50

    # 調教師: trainerにペース型を付与して集計
    if not trainer.empty and "調教師" in trainer.columns:
        key = ["日付", "場所", "R", "馬番", "馬名"]
        pace_key_cols = [c for c in key if c in pace.columns]
        trainer_key_cols = [c for c in key if c in trainer.columns]
        if set(pace_key_cols) == set(trainer_key_cols) and len(pace_key_cols) >= 4:
            merge_cols = pace_key_cols
            p_small = pace[merge_cols + ["ペース型", "is_top3"]].copy()
            t = trainer.merge(p_small, on=merge_cols, how="left")
        else:
            t = trainer.copy()
            t["ペース型"] = "不明"
            if "着順数値" not in t.columns:
                t["着順数値"] = np.nan
            t["is_top3"] = t["着順数値"].between(1, 3)

        tr = t.dropna(subset=["調教師"]).groupby(["調教師", "ペース型"], dropna=False).agg(
            調教師出走数=("馬名", "size"),
            調教師複勝数=("is_top3", "sum"),
        ).reset_index()
        tr["調教師複勝率"] = tr["調教師複勝数"] / tr["調教師出走数"].replace(0, np.nan)
        tr["調教師補正率"] = (tr["調教師複勝数"] + base_rate * k) / (tr["調教師出走数"] + k)
        tr["調教師ペース点"] = tr.groupby("ペース型")["調教師補正率"].transform(lambda s: percentile_scores(s, True))
        tr.loc[tr["調教師出走数"] < 10, "調教師ペース点"] = 50
    else:
        tr = pd.DataFrame(columns=["調教師", "ペース型", "調教師出走数", "調教師複勝率", "調教師ペース点"])

    # cond2をcondに属性として持たせるのではなく、返すと複雑なのでcondに混ぜる用に保存
    cond.attrs["fallback"] = cond2
    return cond, j, tr


# ============================================================
# 予想CSVを判定
# ============================================================

def attach_condition_reference(pred: pd.DataFrame, cond: pd.DataFrame) -> pd.DataFrame:
    out = pred.merge(cond, on=["場所", "芝ダ", "距離", "馬場状態"], how="left")
    fb = cond.attrs.get("fallback")
    if fb is not None:
        out = out.merge(fb, on=["場所", "芝ダ", "距離"], how="left")
        # 馬場込み条件がない時だけfallback
        for a, b in [
            ("基準RPCI", "基準RPCI2"), ("基準PCI", "基準PCI2"),
            ("基準Ave3F", "基準Ave3F2"), ("基準上3F地点差", "基準上3F地点差2"),
            ("基準四角率", "基準四角率2"),
        ]:
            if a in out.columns and b in out.columns:
                out[a] = out[a].fillna(out[b])
    return out


def score_prediction(pred_df: pd.DataFrame, cond: pd.DataFrame, jockey_tbl: pd.DataFrame, trainer_tbl: pd.DataFrame) -> pd.DataFrame:
    df = prep_common(pred_df)

    # 7〜12Rのみ
    if "R" in df.columns:
        df = df[df["R"].between(7, 12)].copy()

    # 必須列がない場合は空を補う
    for c in ["前走RPCI", "前走PCI", "前走Ave3F", "前走上3F地点差", "前走頭数", "騎手", "調教師"]:
        if c not in df.columns:
            df[c] = np.nan

    # 前4角列
    prev4 = choose_prev_corner_col(df, "前4角")
    if prev4:
        df["前4角_使用"] = to_num(df[prev4])
    else:
        df["前4角_使用"] = np.nan
    df["前4角率"] = df.apply(lambda r: safe_div(r.get("前4角_使用"), r.get("前走頭数")), axis=1)

    df = attach_condition_reference(df, cond)

    # 今回条件の標準RPCIからペース型を推定
    df["今回ペース型"] = df["基準RPCI"].apply(pace_type_from_rpci)

    # ペース適性スコア
    df["RPCI一致点"] = [score_fit(v, t, 5.0, 60) for v, t in zip(df["前走RPCI"], df["基準RPCI"])]
    df["PCI一致点"] = [score_fit(v, t, 4.0, 60) for v, t in zip(df["前走PCI"], df["基準PCI"])]
    df["Ave3F一致点"] = [score_fit(v, t, 12.0, 60) for v, t in zip(df["前走Ave3F"], df["基準Ave3F"])]
    df["上3F地点差一致点"] = [score_fit(v, t, 25.0, 60) for v, t in zip(df["前走上3F地点差"], df["基準上3F地点差"])]
    df["位置一致点"] = [score_fit(v, t, 120.0, 60) for v, t in zip(df["前4角率"], df["基準四角率"])]

    df["ペーススコア"] = (
        df["RPCI一致点"] * 0.35 +
        df["PCI一致点"] * 0.15 +
        df["Ave3F一致点"] * 0.20 +
        df["上3F地点差一致点"] * 0.10 +
        df["位置一致点"] * 0.20
    ).round(1)

    # 騎手/調教師点を付与
    df = df.merge(
        jockey_tbl[["騎手", "ペース型", "騎手騎乗数", "騎手複勝率", "騎手ペース点"]].rename(columns={"ペース型": "今回ペース型"}),
        on=["騎手", "今回ペース型"], how="left"
    )
    df["騎手ペース点"] = df["騎手ペース点"].fillna(50).round(1)
    df["騎手騎乗数"] = df["騎手騎乗数"].fillna(0).astype(int)

    if not trainer_tbl.empty and "調教師" in df.columns:
        df = df.merge(
            trainer_tbl[["調教師", "ペース型", "調教師出走数", "調教師複勝率", "調教師ペース点"]].rename(columns={"ペース型": "今回ペース型"}),
            on=["調教師", "今回ペース型"], how="left"
        )
    else:
        df["調教師出走数"] = 0
        df["調教師複勝率"] = np.nan
        df["調教師ペース点"] = 50
    df["調教師ペース点"] = df["調教師ペース点"].fillna(50).round(1)
    df["調教師出走数"] = df["調教師出走数"].fillna(0).astype(int)

    # レース内順位
    race_key = ["日付", "場所", "R"]
    df["ペース順位"] = df.groupby(race_key)["ペーススコア"].rank(method="first", ascending=False).astype(int)

    # 評価分類
    def label_row(r):
        rank = int(r["ペース順位"])
        j70 = float(r.get("騎手ペース点", 0)) >= 70
        t70 = float(r.get("調教師ペース点", 0)) >= 70
        if rank == 1 and j70 and t70:
            return "S評価"
        if rank == 1 and (j70 or t70):
            return "A評価"
        if rank == 1:
            return "B評価"
        if rank in (2, 3) and j70:
            return "相手候補"
        if rank in (2, 3) and t70:
            return "注意候補"
        return ""

    df["評価"] = df.apply(label_row, axis=1)

    # 表示用の短評
    def comment_row(r):
        parts = []
        if r["ペース順位"] == 1:
            parts.append("ペース1位")
        elif r["ペース順位"] in (2, 3):
            parts.append(f"ペース{int(r['ペース順位'])}位")
        if r.get("騎手ペース点", 0) >= 70:
            parts.append("騎手70+")
        if r.get("調教師ペース点", 0) >= 70:
            parts.append("調教師70+")
        return " / ".join(parts)

    df["判定理由"] = df.apply(comment_row, axis=1)

    return df.sort_values(["日付", "場所", "R", "ペース順位", "馬番"]).reset_index(drop=True)


# ============================================================
# 成績集計: 着順/配当がある場合
# ============================================================

def clean_pay(x):
    if pd.isna(x):
        return 0.0
    s = str(x).replace(",", "").replace("円", "").replace("(", "").replace(")", "").strip()
    try:
        return float(s)
    except Exception:
        return 0.0


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    if "着順数値" not in df.columns or df["着順数値"].isna().all():
        return pd.DataFrame()
    d = df[df["評価"].astype(str) != ""].copy()
    if d.empty:
        return pd.DataFrame()
    for c in ["単勝配当", "複勝配当"]:
        if c not in d.columns:
            d[c] = 0
        d[c] = d[c].apply(clean_pay)
    rows = []
    order = ["S評価", "A評価", "B評価", "相手候補", "注意候補"]
    for label in order:
        x = d[d["評価"] == label]
        if x.empty:
            continue
        n = len(x)
        win = (x["着順数値"] == 1).sum()
        top3 = x["着順数値"].between(1, 3).sum()
        rows.append({
            "評価": label,
            "頭数": n,
            "勝率": round(win / n * 100, 1),
            "複勝率": round(top3 / n * 100, 1),
            "単勝回収率": round(x["単勝配当"].sum() / (n * 100) * 100, 1),
            "複勝回収率": round(x["複勝配当"].sum() / (n * 100) * 100, 1),
        })
    return pd.DataFrame(rows)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


# ============================================================
# Streamlit UI
# ============================================================

st.title("新ペースロジック判定アプリ")
st.caption("ペース1位を主役にして、騎手ペース70以上・調教師ペース70以上で S/A/B/相手候補/注意候補 を判定します。")

with st.expander("このアプリで使う評価ルール", expanded=False):
    st.markdown(
        """
- **S評価**：ペース1位 ＋ 騎手ペース70以上 ＋ 調教師ペース70以上
- **A評価**：ペース1位 ＋ 騎手70以上 or 調教師70以上
- **B評価**：ペース1位のみ
- **相手候補**：ペース2〜3位 ＋ 騎手70以上
- **注意候補**：ペース2〜3位 ＋ 調教師70以上 ※騎手70未満

※ 1〜6Rは自動で除外し、7〜12Rだけ判定します。  
※ 欠損値は中立点で処理します。
        """
    )

col1, col2, col3 = st.columns(3)
with col1:
    pred_file = st.file_uploader("予想用CSVをアップロード", type=["csv"], key="pred")
with col2:
    pace_upload = st.file_uploader("ペース10年CSV 任意", type=["csv"], key="pace")
with col3:
    trainer_upload = st.file_uploader("調教師10年CSV 任意", type=["csv"], key="trainer")

pace_path = find_existing_file(PACE_FILE_CANDIDATES)
trainer_path = find_existing_file(TRAINER_FILE_CANDIDATES)

try:
    if pace_upload is not None:
        pace_df = read_csv_auto(pace_upload)
        pace_source = "アップロード"
    elif pace_path is not None:
        pace_df = read_csv_auto(pace_path)
        pace_source = pace_path.name
    else:
        pace_df = pd.DataFrame()
        pace_source = "未読込"

    if trainer_upload is not None:
        trainer_df = read_csv_auto(trainer_upload)
        trainer_source = "アップロード"
    elif trainer_path is not None:
        trainer_df = read_csv_auto(trainer_path)
        trainer_source = trainer_path.name
    else:
        trainer_df = pd.DataFrame()
        trainer_source = "未読込"
except Exception as e:
    st.error(f"過去データの読み込みでエラー: {e}")
    st.stop()

st.info(f"過去ペースCSV: {pace_source} / 調教師CSV: {trainer_source}")

if pace_df.empty:
    st.warning("ペース10年CSVが必要です。同じフォルダに『ペース１０年.csv』を置くか、画面からアップロードしてください。")
    st.stop()

try:
    cond_tbl, jockey_tbl, trainer_tbl = build_reference_tables(pace_df, trainer_df)
except Exception as e:
    st.error(f"基準表の作成でエラー: {e}")
    st.stop()

st.success(f"基準表作成完了：条件 {len(cond_tbl):,}件 / 騎手 {len(jockey_tbl):,}件 / 調教師 {len(trainer_tbl):,}件")

if pred_file is None:
    st.warning("まず予想用CSVをアップロードしてください。")
    st.stop()

try:
    pred_df = read_csv_auto(pred_file)
    result = score_prediction(pred_df, cond_tbl, jockey_tbl, trainer_tbl)
except Exception as e:
    st.error(f"判定中にエラー: {e}")
    st.stop()

# サマリー
st.subheader("評価サマリー")
summary = result[result["評価"].astype(str) != ""].groupby("評価").agg(
    頭数=("馬名", "size"),
    平均ペーススコア=("ペーススコア", "mean"),
    平均騎手点=("騎手ペース点", "mean"),
    平均調教師点=("調教師ペース点", "mean"),
).reset_index()
order = pd.CategoricalDtype(["S評価", "A評価", "B評価", "相手候補", "注意候補"], ordered=True)
if not summary.empty:
    summary["評価"] = summary["評価"].astype(order)
    summary = summary.sort_values("評価")
    for c in ["平均ペーススコア", "平均騎手点", "平均調教師点"]:
        summary[c] = summary[c].round(1)
    st.dataframe(summary, use_container_width=True)
else:
    st.warning("評価対象馬がありませんでした。")

perf = summarize_results(result)
if not perf.empty:
    st.subheader("成績集計 ※着順・配当がある場合")
    st.dataframe(perf, use_container_width=True)

# レース別表示
st.subheader("レース別 推奨馬")
show_cols = [
    "日付", "場所", "R", "レース名", "馬番", "馬名", "騎手", "調教師", "評価", "判定理由",
    "ペース順位", "ペーススコア", "今回ペース型", "騎手ペース点", "調教師ペース点",
    "前走RPCI", "前走PCI", "前走Ave3F", "前走上3F地点差", "前4角_使用", "前走頭数"
]
show_cols = [c for c in show_cols if c in result.columns]
recommended = result[result["評価"].astype(str) != ""].copy()
st.dataframe(recommended[show_cols], use_container_width=True, height=520)

with st.expander("全馬順位を見る", expanded=False):
    all_cols = show_cols.copy()
    st.dataframe(result[all_cols], use_container_width=True, height=600)

# ダウンロード
st.subheader("CSV保存")
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.download_button(
        "評価付き全馬CSVを保存",
        data=to_csv_bytes(result),
        file_name="新ペースロジック_全馬判定.csv",
        mime="text/csv",
    )
with col_b:
    st.download_button(
        "推奨馬CSVを保存",
        data=to_csv_bytes(recommended),
        file_name="新ペースロジック_推奨馬.csv",
        mime="text/csv",
    )
with col_c:
    if not perf.empty:
        st.download_button(
            "評価別成績CSVを保存",
            data=to_csv_bytes(perf),
            file_name="新ペースロジック_評価別成績.csv",
            mime="text/csv",
        )

st.caption("注：このアプリ内の騎手70・調教師70は、過去データ内のペース型別成績をパーセンタイル化した信頼度です。")
