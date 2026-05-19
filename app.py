import io
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ペース本命ロジック", layout="wide")

# =========================
# 基本ユーティリティ
# =========================

ALIASES = {
    "R": ["R", "Ｒ", "レース", "レース番号"],
    "芝ダ": ["芝ダ", "芝・ダ", "芝ダート"],
    "場所": ["場所", "競馬場"],
    "日付": ["日付", "年月日"],
    "馬場状態": ["馬場状態", "馬場"],
    "頭数": ["頭数", "出走頭数"],
    "馬番": ["馬番", "番"],
    "馬名": ["馬名", "馬"],
    "騎手": ["騎手", "騎手名"],
    "調教師": ["調教師", "厩舎", "調教師名"],
    "着順": ["着順", "確定着順"],
    "単勝配当": ["単勝配当", "単勝", "単配", "単勝払戻"],
    "複勝配当": ["複勝配当", "複勝", "複配", "複勝払戻"],
}

PACE_FEATURE_ALIASES = {
    "RPCI": ["前走RPCI", "近走RPCI", "RPCI"],
    "PCI": ["前走PCI", "近走PCI", "PCI"],
    "Ave-3F": ["前走Ave-3F", "近走Ave-3F", "Ave-3F"],
    "上3F地点差": ["前走上3F地点差", "近走上3F地点差", "上3F地点差"],
    "4角": ["前走4角", "近走4角", "4角", "前走4角通過順"],
    "3角": ["前走3角", "近走3角", "3角", "前走3角通過順"],
    "平均1Fタイム": ["前走平均1Fタイム", "近走平均1Fタイム", "平均1Fタイム"],
    "平均速度": ["前走平均速度", "近走平均速度", "平均速度"],
    "上り3F平均速度": ["前走上り3F平均速度", "近走上り3F平均速度", "上り3F平均速度"],
}

BASE_COLS = ["日付", "場所", "R", "レース名", "芝ダ", "距離", "馬場状態", "頭数", "馬番", "馬名", "騎手", "調教師"]
PRED_REQUIRED = ["日付", "場所", "R", "芝ダ", "距離", "馬場状態", "頭数", "馬番", "馬名", "騎手", "調教師"]
PRED_PACE_REQUIRED = ["前走RPCI", "前走PCI", "前走Ave-3F", "前走上3F地点差", "前走4角"]


def read_csv_auto(file) -> pd.DataFrame:
    data = file.read() if hasattr(file, "read") else open(file, "rb").read()
    for enc in ["cp932", "utf-8-sig", "utf-8"]:
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            continue
    return pd.read_csv(io.BytesIO(data))


def norm_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip().replace("　", " ")


def to_num(s):
    if pd.isna(s):
        return np.nan
    s = str(s).strip()
    s = s.replace(",", "").replace("円", "")
    s = s.replace("(", "").replace(")", "")
    s = s.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else np.nan


def normalize_r(x):
    n = to_num(x)
    return int(n) if not pd.isna(n) else np.nan


def normalize_place(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 列名の正規化
    rename = {}
    for standard, names in ALIASES.items():
        for c in df.columns:
            if c in names:
                rename[c] = standard
                break
    df = df.rename(columns=rename)
    if "R" in df.columns:
        df["R"] = df["R"].map(normalize_r)
    if "芝ダ" in df.columns:
        df["芝ダ"] = df["芝ダ"].astype(str).str.replace("・", "").str.strip()
    if "場所" in df.columns:
        df["場所"] = df["場所"].map(norm_text).str.replace("競馬場", "", regex=False)
    if "馬場状態" in df.columns:
        df["馬場状態"] = df["馬場状態"].map(norm_text)
    if "距離" in df.columns:
        df["距離"] = df["距離"].map(to_num)
    if "頭数" in df.columns:
        df["頭数"] = df["頭数"].map(to_num)
    if "馬番" in df.columns:
        df["馬番"] = df["馬番"].map(to_num)
    if "着順" in df.columns:
        df["着順数値"] = df["着順"].map(to_num)
    for c in ["単勝配当", "複勝配当"]:
        if c in df.columns:
            df[c] = df[c].map(to_num)
    return df


def add_pace_feature_aliases(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for standard, names in PACE_FEATURE_ALIASES.items():
        if standard in df.columns:
            continue
        for c in names:
            if c in df.columns:
                df[standard] = df[c]
                break
    for c in ["RPCI", "PCI", "Ave-3F", "上3F地点差", "4角", "3角", "平均1Fタイム", "平均速度", "上り3F平均速度"]:
        if c in df.columns:
            df[c] = df[c].map(to_num)
    return df


def pace_type_from_rpci(x: float) -> str:
    if pd.isna(x):
        return "不明"
    if x <= 46:
        return "ハイ/持続"
    if x < 50:
        return "やや速い"
    if x < 54:
        return "標準〜やや上がり"
    return "スロー上がり"


def condition_key_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in ["場所", "芝ダ", "距離", "馬場状態"] if c in df.columns]


def make_condition_stats(hist: pd.DataFrame) -> pd.DataFrame:
    df = hist.copy()
    df = normalize_place(df)
    df = add_pace_feature_aliases(df)
    if "着順数値" not in df.columns and "着順" in df.columns:
        df["着順数値"] = df["着順"].map(to_num)
    df["好走"] = df["着順数値"].between(1, 3, inclusive="both")
    if "4角" in df.columns and "頭数" in df.columns:
        df["4角率"] = df["4角"] / df["頭数"].replace(0, np.nan)
    else:
        df["4角率"] = np.nan
    df["ペース型"] = df.get("RPCI", pd.Series(np.nan, index=df.index)).map(pace_type_from_rpci)
    keys = condition_key_cols(df)
    features = ["RPCI", "PCI", "Ave-3F", "上3F地点差", "4角率", "平均1Fタイム", "平均速度", "上り3F平均速度"]
    use_cols = keys + ["好走"] + [c for c in features if c in df.columns]
    top3 = df[df["好走"]].copy()
    agg = top3.groupby(keys, dropna=False).agg(
        レース好走頭数=("好走", "size"),
        標準RPCI=("RPCI", "mean"),
        標準PCI=("PCI", "mean"),
        標準Ave3F=("Ave-3F", "mean"),
        標準上3F地点差=("上3F地点差", "mean"),
        標準4角率=("4角率", "mean"),
        標準平均1F=("平均1Fタイム", "mean"),
        標準平均速度=("平均速度", "mean"),
        標準上り3F平均速度=("上り3F平均速度", "mean"),
        sdRPCI=("RPCI", "std"),
        sdPCI=("PCI", "std"),
        sdAve3F=("Ave-3F", "std"),
        sd上3F地点差=("上3F地点差", "std"),
        sd4角率=("4角率", "std"),
    ).reset_index()
    agg["標準ペース型"] = agg["標準RPCI"].map(pace_type_from_rpci)
    return agg


def make_rate_table(hist: pd.DataFrame, entity_col: str) -> pd.DataFrame:
    df = hist.copy()
    df = normalize_place(df)
    df = add_pace_feature_aliases(df)
    if entity_col not in df.columns:
        return pd.DataFrame()
    if "着順数値" not in df.columns and "着順" in df.columns:
        df["着順数値"] = df["着順"].map(to_num)
    df["好走"] = df["着順数値"].between(1, 3, inclusive="both").astype(int)
    df["ペース型"] = df.get("RPCI", pd.Series(np.nan, index=df.index)).map(pace_type_from_rpci)
    base = df["好走"].mean()
    keys = [entity_col, "ペース型"]
    tbl = df.groupby(keys, dropna=False).agg(
        件数=("好走", "size"),
        複勝率=("好走", "mean"),
    ).reset_index()
    # 70点以上が高適性になるように、全体平均との差をスコア化
    tbl["適性点"] = 50 + (tbl["複勝率"] - base) * 220 + np.minimum(tbl["件数"], 100) / 100 * 10
    tbl.loc[tbl["件数"] < 10, "適性点"] = np.minimum(tbl.loc[tbl["件数"] < 10, "適性点"], 60)
    tbl["適性点"] = tbl["適性点"].clip(0, 100).round(1)
    tbl = tbl.rename(columns={"適性点": f"{entity_col}ペース点", "件数": f"{entity_col}件数", "複勝率": f"{entity_col}複勝率"})
    return tbl


def calc_similarity_score(row: pd.Series) -> float:
    pairs = [
        ("RPCI", "標準RPCI", "sdRPCI", 0.30),
        ("PCI", "標準PCI", "sdPCI", 0.20),
        ("Ave-3F", "標準Ave3F", "sdAve3F", 0.15),
        ("上3F地点差", "標準上3F地点差", "sd上3F地点差", 0.15),
        ("4角率", "標準4角率", "sd4角率", 0.20),
    ]
    scores = []
    weights = []
    for src, target, sd, w in pairs:
        x = row.get(src, np.nan)
        t = row.get(target, np.nan)
        s = row.get(sd, np.nan)
        if pd.isna(x) or pd.isna(t):
            continue
        if pd.isna(s) or s == 0:
            # 4角率は小さいので最低sdを調整
            s = 0.12 if src == "4角率" else 3.0
        z = abs(x - t) / s
        fs = max(0, 100 - min(z, 3) * 33.333)
        scores.append(fs * w)
        weights.append(w)
    if not weights:
        return np.nan
    return float(np.sum(scores) / np.sum(weights))


def score_prediction(pred: pd.DataFrame, condition_stats: pd.DataFrame, jockey_tbl: pd.DataFrame, trainer_tbl: pd.DataFrame) -> pd.DataFrame:
    df = normalize_place(pred)
    df = add_pace_feature_aliases(df)
    # 4角率
    if "4角" in df.columns and "頭数" in df.columns:
        df["4角率"] = df["4角"] / df["頭数"].replace(0, np.nan)
    else:
        df["4角率"] = np.nan
    keys = condition_key_cols(df)
    cs = condition_stats.copy()
    df = df.merge(cs, on=keys, how="left")
    df["標準ペース型"] = df["標準ペース型"].fillna(df.get("RPCI", pd.Series(np.nan, index=df.index)).map(pace_type_from_rpci))
    df["ペース適性点"] = df.apply(calc_similarity_score, axis=1).round(1)

    # 騎手・調教師点
    if not jockey_tbl.empty and "騎手" in df.columns:
        df = df.merge(jockey_tbl, left_on=["騎手", "標準ペース型"], right_on=["騎手", "ペース型"], how="left")
    if not trainer_tbl.empty and "調教師" in df.columns:
        df = df.merge(trainer_tbl, left_on=["調教師", "標準ペース型"], right_on=["調教師", "ペース型"], how="left", suffixes=("", "_調教師"))
    if "騎手ペース点" not in df.columns:
        df["騎手ペース点"] = np.nan
    if "調教師ペース点" not in df.columns:
        df["調教師ペース点"] = np.nan
    df["騎手ペース点"] = df["騎手ペース点"].fillna(50)
    df["調教師ペース点"] = df["調教師ペース点"].fillna(50)

    df["総合点"] = (df["ペース適性点"].fillna(50) * 0.70 + df["騎手ペース点"] * 0.15 + df["調教師ペース点"] * 0.15).round(1)
    # レースキーと順位
    for c in ["日付", "場所", "R"]:
        if c not in df.columns:
            df[c] = ""
    df["レースキー"] = df["日付"].astype(str) + "_" + df["場所"].astype(str) + "_" + df["R"].astype(str)
    df["ペース順位"] = df.groupby("レースキー")["ペース適性点"].rank(ascending=False, method="first").astype("Int64")
    df["総合順位"] = df.groupby("レースキー")["総合点"].rank(ascending=False, method="first").astype("Int64")

    def label(r):
        pr = r["ペース順位"]
        j70 = r["騎手ペース点"] >= 70
        t70 = r["調教師ペース点"] >= 70
        if pr == 1 and j70 and t70:
            return "S評価"
        if pr == 1 and (j70 or t70):
            return "A評価"
        if pr == 1:
            return "B評価"
        if pr in [2, 3] and j70:
            return "相手候補"
        if pr in [2, 3] and t70:
            return "注意候補"
        return ""

    df["評価"] = df.apply(label, axis=1)
    df["買い方目安"] = df["評価"].map({
        "S評価": "超本命・単複/軸候補",
        "A評価": "強本命・軸候補",
        "B評価": "通常本命・単複穴候補",
        "相手候補": "ワイド/馬連/三連複の相手筆頭",
        "注意候補": "押さえ候補",
    }).fillna("")
    return df


def validate_required(df: pd.DataFrame, required: List[str]) -> List[str]:
    cols = set(normalize_place(df).columns)
    return [c for c in required if c not in cols]


def make_sample_csv() -> bytes:
    cols = BASE_COLS + ["前走RPCI", "前走PCI", "前走Ave-3F", "前走上3F地点差", "前走4角"]
    sample = pd.DataFrame([{c: "" for c in cols}])
    return sample.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

# =========================
# UI
# =========================

st.title("ペース本命ロジック アプリ")
st.caption("ペースロジックを主役にして、騎手70・調教師70を信頼度ラベルにするランキングアプリ")

with st.expander("必要CSV項目", expanded=True):
    st.markdown("""
### 1. 過去10年ペースCSV
TARGETから抜いた `ペース１０年.csv` を入れてください。最低限必要です。

`日付, 場所, R, レース名, 芝・ダ, 距離, 馬場状態, 頭数, 馬番, 馬名, 騎手, 着順, RPCI, PCI, Ave-3F, 上3F地点差, 4角`

### 2. 調教師CSV
`調教師１０年.csv` または調教師が入った過去データを入れてください。

`日付, 場所, R, 馬番, 馬名, 騎手, 調教師, 着順`

### 3. 予想用CSV
今回予想したい出走馬CSVです。未来レースでは結果後のRPCI/PCIは使えないので、**前走または近走の値**を入れてください。

`日付,場所,R,レース名,芝ダ,距離,馬場状態,頭数,馬番,馬名,騎手,調教師,前走RPCI,前走PCI,前走Ave-3F,前走上3F地点差,前走4角`
""")
    st.download_button("予想用CSVテンプレートをダウンロード", data=make_sample_csv(), file_name="pace_logic_input_template.csv", mime="text/csv")

col1, col2, col3 = st.columns(3)
with col1:
    hist_file = st.file_uploader("過去10年ペースCSV", type=["csv"], key="hist")
with col2:
    trainer_file = st.file_uploader("調教師CSV", type=["csv"], key="trainer")
with col3:
    pred_file = st.file_uploader("予想用CSV", type=["csv"], key="pred")

st.sidebar.header("しきい値")
jockey_thr = st.sidebar.slider("騎手ペース高適性", 50, 90, 70)
trainer_thr = st.sidebar.slider("調教師ペース高適性", 50, 90, 70)
st.sidebar.caption("表示ラベルは検証時の基準に合わせて初期値70です。")

if hist_file and trainer_file and pred_file:
    hist = read_csv_auto(hist_file)
    trainer_hist = read_csv_auto(trainer_file)
    pred = read_csv_auto(pred_file)

    hist_n = normalize_place(hist)
    hist_n = add_pace_feature_aliases(hist_n)
    trainer_n = normalize_place(trainer_hist)
    # 調教師CSVにRPCIがない場合は、ペースCSVから調教師を照合して作る
    if "調教師" in trainer_n.columns and "RPCI" not in trainer_n.columns:
        join_cols = [c for c in ["日付", "場所", "R", "馬番", "馬名"] if c in hist_n.columns and c in trainer_n.columns]
        add_cols = join_cols + [c for c in ["RPCI", "着順", "騎手", "調教師"] if c in hist_n.columns or c in trainer_n.columns]
        trainer_n = trainer_n.merge(hist_n[[c for c in ["日付", "場所", "R", "馬番", "馬名", "RPCI"] if c in hist_n.columns]], on=join_cols, how="left")

    missing_hist = [c for c in ["場所", "芝ダ", "距離", "馬場状態", "着順", "RPCI", "PCI"] if c not in hist_n.columns]
    missing_pred = validate_required(pred, PRED_REQUIRED)
    if missing_hist:
        st.error(f"過去ペースCSVに不足項目があります: {missing_hist}")
        st.stop()
    if missing_pred:
        st.error(f"予想用CSVに不足項目があります: {missing_pred}")
        st.stop()

    with st.spinner("過去データから標準ペース・騎手/調教師適性を作成中..."):
        cond_stats = make_condition_stats(hist_n)
        jockey_tbl = make_rate_table(hist_n, "騎手")
        # 調教師テーブルは調教師CSVにRPCIが入っている/照合済みで作成
        trainer_for_rate = trainer_n.copy()
        if "RPCI" not in trainer_for_rate.columns:
            trainer_for_rate = trainer_for_rate.merge(hist_n[[c for c in ["日付", "場所", "R", "馬番", "馬名", "RPCI"] if c in hist_n.columns]], on=[c for c in ["日付", "場所", "R", "馬番", "馬名"] if c in trainer_for_rate.columns and c in hist_n.columns], how="left")
        trainer_tbl = make_rate_table(trainer_for_rate, "調教師")
        result = score_prediction(pred, cond_stats, jockey_tbl, trainer_tbl)

    # スライダー値で評価ラベルだけ再判定
    def relabel(r):
        pr = r["ペース順位"]
        j70 = r["騎手ペース点"] >= jockey_thr
        t70 = r["調教師ペース点"] >= trainer_thr
        if pr == 1 and j70 and t70:
            return "S評価"
        if pr == 1 and (j70 or t70):
            return "A評価"
        if pr == 1:
            return "B評価"
        if pr in [2, 3] and j70:
            return "相手候補"
        if pr in [2, 3] and t70:
            return "注意候補"
        return ""
    result["評価"] = result.apply(relabel, axis=1)
    result["買い方目安"] = result["評価"].map({
        "S評価": "超本命・単複/軸候補",
        "A評価": "強本命・軸候補",
        "B評価": "通常本命・単複穴候補",
        "相手候補": "ワイド/馬連/三連複の相手筆頭",
        "注意候補": "押さえ候補",
    }).fillna("")

    st.success("ランキング作成完了")

    show_cols = [c for c in [
        "日付", "場所", "R", "レース名", "馬番", "馬名", "騎手", "調教師",
        "評価", "ペース順位", "ペース適性点", "騎手ペース点", "調教師ペース点", "総合点", "標準ペース型", "買い方目安"
    ] if c in result.columns]

    st.subheader("最終ランキング")
    display = result.sort_values(["日付", "場所", "R", "ペース順位", "総合順位"])[show_cols]
    st.dataframe(display, use_container_width=True, height=600)

    st.subheader("評価別件数")
    summary = result[result["評価"] != ""].groupby("評価").size().reindex(["S評価", "A評価", "B評価", "相手候補", "注意候補"]).fillna(0).astype(int).reset_index(name="頭数")
    st.dataframe(summary, use_container_width=True)

    out = result[show_cols].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button("ランキングCSVをダウンロード", data=out, file_name="pace_logic_ranking.csv", mime="text/csv")

    all_out = result.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button("全項目CSVをダウンロード", data=all_out, file_name="pace_logic_ranking_all_columns.csv", mime="text/csv")

else:
    st.info("過去10年ペースCSV、調教師CSV、予想用CSVの3つをアップロードしてください。")

st.markdown("---")
st.caption("注意：予想用CSVにはレース後にしか分からない今回RPCI/PCI/着順/配当を入れないでください。前走または近走の値だけで予想します。")
