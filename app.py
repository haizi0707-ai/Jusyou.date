import io
import re
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ペース本命ロジック", layout="wide")

APP_DIR = Path(__file__).resolve().parent
DEFAULT_CONDITION = APP_DIR / "pace_condition_summary.csv"
DEFAULT_JOCKEY = APP_DIR / "jockey_pace_summary.csv"
DEFAULT_TRAINER = APP_DIR / "trainer_pace_summary.csv"
DEFAULT_PRED = APP_DIR / "予想用.csv"

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
}

PACE_FEATURE_ALIASES = {
    "RPCI": ["前走RPCI", "近走RPCI", "RPCI"],
    "PCI": ["前走PCI", "近走PCI", "PCI"],
    "Ave-3F": ["前走Ave-3F", "近走Ave-3F", "Ave-3F"],
    "上3F地点差": ["前走上3F地点差", "近走上3F地点差", "上3F地点差"],
    "4角": ["前走4角", "近走4角", "4角", "前走4角通過順"],
    "3角": ["前走3角", "近走3角", "3角", "前走3角通過順"],
}

BASE_COLS = ["日付", "場所", "R", "レース名", "芝ダ", "距離", "頭数", "馬番", "馬名", "騎手", "調教師"]
PRED_REQUIRED = ["日付", "場所", "R", "芝ダ", "距離", "頭数", "馬番", "馬名", "騎手", "調教師"]


def read_csv_auto(file):
    data = file.read() if hasattr(file, "read") else open(file, "rb").read()
    for enc in ["cp932", "utf-8-sig", "utf-8"]:
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.read_csv(io.BytesIO(data))


def norm_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip().replace("　", " ")


def to_num(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace(",", "").replace("円", "")
    s = s.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else np.nan


def normalize_r(x):
    n = to_num(x)
    return int(n) if not pd.isna(n) else np.nan


def normalize(df):
    df = df.copy()
    rename = {}
    for std, names in ALIASES.items():
        if std in df.columns:
            continue
        for c in df.columns:
            if c in names:
                rename[c] = std
                break
    df = df.rename(columns=rename)
    for std, names in PACE_FEATURE_ALIASES.items():
        if std in df.columns:
            continue
        for c in names:
            if c in df.columns:
                df[std] = df[c]
                break
    if "R" in df.columns:
        df["R"] = df["R"].map(normalize_r)
    if "場所" in df.columns:
        df["場所"] = df["場所"].map(norm_text).str.replace("競馬場", "", regex=False)
    if "芝ダ" in df.columns:
        df["芝ダ"] = df["芝ダ"].astype(str).str.replace("・", "", regex=False).str.strip()
    if "馬場状態" in df.columns:
        df["馬場状態"] = df["馬場状態"].map(norm_text)
    for c in ["距離", "頭数", "馬番", "RPCI", "PCI", "Ave-3F", "上3F地点差", "4角", "3角"]:
        if c in df.columns:
            df[c] = df[c].map(to_num)
    return df


def pace_type_from_rpci(x):
    if pd.isna(x):
        return "不明"
    if x <= 46:
        return "ハイ/持続"
    if x < 50:
        return "やや速い"
    if x < 54:
        return "標準〜やや上がり"
    return "スロー上がり"


def condition_key_cols(df):
    return [c for c in ["場所", "芝ダ", "距離", "馬場状態"] if c in df.columns]


def calc_similarity_score(row):
    pairs = [
        ("RPCI", "標準RPCI", "sdRPCI", 0.30),
        ("PCI", "標準PCI", "sdPCI", 0.20),
        ("Ave-3F", "標準Ave3F", "sdAve3F", 0.15),
        ("上3F地点差", "標準上3F地点差", "sd上3F地点差", 0.15),
        ("4角率", "標準4角率", "sd4角率", 0.20),
    ]
    scores, weights = [], []
    for src, target, sd, w in pairs:
        x, t, s = row.get(src, np.nan), row.get(target, np.nan), row.get(sd, np.nan)
        if pd.isna(x) or pd.isna(t):
            continue
        if pd.isna(s) or s == 0:
            s = 0.12 if src == "4角率" else 3.0
        z = abs(x - t) / s
        score = max(0, 100 - min(z, 3) * 33.333)
        scores.append(score * w)
        weights.append(w)
    if not weights:
        return np.nan
    return float(np.sum(scores) / np.sum(weights))


def load_summary_tables():
    missing = [p.name for p in [DEFAULT_CONDITION, DEFAULT_JOCKEY, DEFAULT_TRAINER] if not p.exists()]
    if missing:
        st.error("同フォルダに不足ファイルがあります: " + ", ".join(missing))
        st.stop()
    cond = normalize(read_csv_auto(str(DEFAULT_CONDITION)))
    jockey = read_csv_auto(str(DEFAULT_JOCKEY))
    trainer = read_csv_auto(str(DEFAULT_TRAINER))
    return cond, jockey, trainer


def normalize_rate_table(tbl, entity_col):
    tbl = tbl.copy()
    rename = {}
    if entity_col not in tbl.columns:
        for c in tbl.columns:
            if entity_col in c:
                rename[c] = entity_col
                break
    score_col = f"{entity_col}ペース点"
    if score_col not in tbl.columns:
        for c in tbl.columns:
            if "適性点" in c or "ペース点" in c:
                rename[c] = score_col
                break
    tbl = tbl.rename(columns=rename)
    if "ペース型" not in tbl.columns and "標準ペース型" in tbl.columns:
        tbl = tbl.rename(columns={"標準ペース型": "ペース型"})
    if score_col in tbl.columns:
        tbl[score_col] = tbl[score_col].map(to_num)
    return tbl


def score_prediction(pred, cond_stats, jockey_tbl, trainer_tbl, jockey_thr, trainer_thr):
    df = normalize(pred)
    if "4角" in df.columns and "頭数" in df.columns:
        df["4角率"] = df["4角"] / df["頭数"].replace(0, np.nan)
    else:
        df["4角率"] = np.nan

    keys = condition_key_cols(df)
    df = df.merge(cond_stats, on=keys, how="left")
    if "標準ペース型" not in df.columns:
        df["標準ペース型"] = df.get("RPCI", pd.Series(np.nan, index=df.index)).map(pace_type_from_rpci)
    else:
        df["標準ペース型"] = df["標準ペース型"].fillna(df.get("RPCI", pd.Series(np.nan, index=df.index)).map(pace_type_from_rpci))

    df["ペース適性点"] = df.apply(calc_similarity_score, axis=1).round(1)

    jockey_tbl = normalize_rate_table(jockey_tbl, "騎手")
    trainer_tbl = normalize_rate_table(trainer_tbl, "調教師")
    if "騎手" in df.columns and {"騎手", "ペース型", "騎手ペース点"}.issubset(jockey_tbl.columns):
        df = df.merge(jockey_tbl[["騎手", "ペース型", "騎手ペース点"]], left_on=["騎手", "標準ペース型"], right_on=["騎手", "ペース型"], how="left")
    if "調教師" in df.columns and {"調教師", "ペース型", "調教師ペース点"}.issubset(trainer_tbl.columns):
        df = df.merge(trainer_tbl[["調教師", "ペース型", "調教師ペース点"]], left_on=["調教師", "標準ペース型"], right_on=["調教師", "ペース型"], how="left", suffixes=("", "_調教師"))

    if "騎手ペース点" not in df.columns:
        df["騎手ペース点"] = 50
    if "調教師ペース点" not in df.columns:
        df["調教師ペース点"] = 50
    df["騎手ペース点"] = df["騎手ペース点"].fillna(50)
    df["調教師ペース点"] = df["調教師ペース点"].fillna(50)
    df["総合点"] = (df["ペース適性点"].fillna(50) * 0.70 + df["騎手ペース点"] * 0.15 + df["調教師ペース点"] * 0.15).round(1)

    for c in ["日付", "場所", "R"]:
        if c not in df.columns:
            df[c] = ""
    df["レースキー"] = df["日付"].astype(str) + "_" + df["場所"].astype(str) + "_" + df["R"].astype(str)
    df["ペース順位"] = df.groupby("レースキー")["ペース適性点"].rank(ascending=False, method="first").astype("Int64")
    df["総合順位"] = df.groupby("レースキー")["総合点"].rank(ascending=False, method="first").astype("Int64")

    def label(r):
        pr = r["ペース順位"]
        j = r["騎手ペース点"] >= jockey_thr
        t = r["調教師ペース点"] >= trainer_thr
        if pr == 1 and j and t:
            return "S評価"
        if pr == 1 and (j or t):
            return "A評価"
        if pr == 1:
            return "B評価"
        if pr in [2, 3] and j:
            return "相手候補"
        if pr in [2, 3] and t:
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


def make_sample_csv():
    cols = BASE_COLS + ["前走RPCI", "前走PCI", "前走Ave-3F", "前走上3F地点差", "前走4角"]
    sample = pd.DataFrame([{c: "" for c in cols}])
    return sample.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def prepare_baba(pred):
    pred_n = normalize(pred)
    if "馬場状態" not in pred_n.columns:
        pred_n["馬場状態"] = "良"
    if "場所" not in pred_n.columns or "芝ダ" not in pred_n.columns:
        return pred_n
    st.subheader("当日馬場状態入力")
    st.caption("予想CSVの馬場状態を、競馬場×芝/ダごとに上書きします。")
    combos = pred_n[["場所", "芝ダ"]].dropna().drop_duplicates().sort_values(["場所", "芝ダ"])
    options = ["良", "稍重", "重", "不良"]
    cols = st.columns(min(3, max(1, len(combos))))
    for i, (_, row) in enumerate(combos.iterrows()):
        place, surface = str(row["場所"]), str(row["芝ダ"])
        mask = (pred_n["場所"].astype(str) == place) & (pred_n["芝ダ"].astype(str) == surface)
        current = pred_n.loc[mask, "馬場状態"].dropna().astype(str).tolist()
        default = current[0] if current and current[0] in options else "良"
        with cols[i % len(cols)]:
            selected = st.selectbox(f"{place}・{surface}", options, index=options.index(default), key=f"baba_{place}_{surface}")
        pred_n.loc[mask, "馬場状態"] = selected
    return pred_n


st.title("ペース本命ロジック アプリ")
st.caption("軽量版：過去10年の生データではなく、集計済みCSVを使います。GitHub 25MB制限対応版です。")

with st.expander("app.py と同じ場所に置くファイル", expanded=True):
    st.markdown("""
以下の3ファイルを `app.py` と同じフォルダに置いてください。

- `pace_condition_summary.csv`
- `jockey_pace_summary.csv`
- `trainer_pace_summary.csv`

巨大な `ペース１０年.csv` や `調教師１０年.csv` は不要です。
予想する時は、予想用CSVをアップロードするか、同じフォルダに `予想用.csv` を置いてください。
""")
    st.download_button("予想用CSVテンプレートをダウンロード", data=make_sample_csv(), file_name="予想用.csv", mime="text/csv")

pred_file = st.file_uploader("予想用CSV", type=["csv"])
use_local_pred = False
if pred_file is None and DEFAULT_PRED.exists():
    use_local_pred = st.checkbox("同フォルダの予想用.csvを使う", value=False)

jockey_thr = st.sidebar.slider("騎手ペース高適性", 50, 90, 70)
trainer_thr = st.sidebar.slider("調教師ペース高適性", 50, 90, 70)

pred_source = pred_file if pred_file is not None else (str(DEFAULT_PRED) if use_local_pred and DEFAULT_PRED.exists() else None)

cond_stats, jockey_tbl, trainer_tbl = load_summary_tables()

if pred_source:
    pred = read_csv_auto(pred_source)
    missing = [c for c in PRED_REQUIRED if c not in normalize(pred).columns]
    if missing:
        st.error(f"予想用CSVに不足項目があります: {missing}")
        st.stop()
    pred = prepare_baba(pred)
    result = score_prediction(pred, cond_stats, jockey_tbl, trainer_tbl, jockey_thr, trainer_thr)
    st.success("ランキング作成完了")
    show_cols = [c for c in [
        "日付", "場所", "R", "レース名", "馬番", "馬名", "騎手", "調教師",
        "評価", "ペース順位", "ペース適性点", "騎手ペース点", "調教師ペース点", "総合点", "標準ペース型", "買い方目安"
    ] if c in result.columns]
    display = result.sort_values(["日付", "場所", "R", "ペース順位", "総合順位"])[show_cols]
    st.subheader("最終ランキング")
    st.dataframe(display, use_container_width=True, height=600)
    summary = result[result["評価"] != ""].groupby("評価").size().reindex(["S評価", "A評価", "B評価", "相手候補", "注意候補"]).fillna(0).astype(int).reset_index(name="頭数")
    st.subheader("評価別件数")
    st.dataframe(summary, use_container_width=True)
    out = result[show_cols].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button("ランキングCSVをダウンロード", data=out, file_name="pace_logic_ranking.csv", mime="text/csv")
    all_out = result.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button("全項目CSVをダウンロード", data=all_out, file_name="pace_logic_ranking_all_columns.csv", mime="text/csv")
else:
    st.info("予想用CSVをアップロードしてください。集計済みCSVは同フォルダから自動読込します。")

st.markdown("---")
st.caption("注意：予想用CSVには、レース後にしか分からない今回RPCI/PCI/着順/配当は入れないでください。前走または近走の値で予想します。")
