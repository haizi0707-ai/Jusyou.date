# -*- coding: utf-8 -*-
"""
重賞⭐️フィルター抽出アプリ
- 重賞別_プロンプト用ロジック辞書.csv を読み込み
- 重賞タグ選択で専用プロンプトを生成
- ChatGPT等で埋めた出走馬該当CSVを貼り付け/アップロード
- S/A/B/補助条件＋消し条件で⭐️候補を1頭抽出
- 結果CSVを読み込み、成績更新用CSVを出力
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
import streamlit as st

APP_TITLE = "重賞⭐️専用フィルター抽出アプリ"
BASE_DIR = Path(__file__).parent
DEFAULT_LOGIC = BASE_DIR / "重賞別_プロンプト用ロジック辞書.csv"
DEFAULT_KESHI = BASE_DIR / "重賞別_消し条件候補.csv"
DEFAULT_SUMMARY = BASE_DIR / "重賞別_抽出サマリー.csv"


def find_csv_by_keyword(keyword: str) -> Path | None:
    """GitHub/iPhoneアップロード時のファイル名ゆれ対策。"""
    if keyword == "logic":
        patterns = ["*プロンプト*ロジック*.csv", "*ロジック辞書*.csv", "*prompt*logic*.csv"]
    elif keyword == "keshi":
        patterns = ["*消し条件*.csv", "*keshi*.csv"]
    else:
        patterns = ["*抽出サマリー*.csv", "*summary*.csv"]
    for pat in patterns:
        hits = sorted(BASE_DIR.glob(pat))
        if hits:
            return hits[0]
    return None

RANK_WEIGHT = {
    "S": 3.0,
    "A": 2.0,
    "B": 1.0,
    "補助": 0.5,
}

TRUE_VALUES = {"○", "〇", "◎", "TRUE", "True", "true", "1", "該当", "Y", "Yes", "YES", "yes"}
FALSE_VALUES = {"×", "✕", "FALSE", "False", "false", "0", "非該当", "N", "No", "NO", "no", ""}

st.set_page_config(page_title=APP_TITLE, page_icon="⭐", layout="wide")


def read_csv_auto(file_or_path) -> pd.DataFrame:
    """CP932/UTF-8-SIG混在対策つきCSV reader."""
    if file_or_path is None:
        return pd.DataFrame()
    if isinstance(file_or_path, (str, Path)):
        path = Path(file_or_path)
        if not path.exists():
            return pd.DataFrame()
        raw = path.read_bytes()
    else:
        raw = file_or_path.getvalue()
    for enc in ("utf-8-sig", "cp932", "utf-8"):
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc)
            df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
            return df
        except Exception:
            pass
    # 最後の保険
    return pd.read_csv(io.BytesIO(raw), encoding="utf-8", errors="replace")


@st.cache_data(show_spinner=False)
def load_default_logic() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    GitHub/iPhoneアップロード時にファイル名が少し変わっても、
    CSVの中身（列名）を見てロジック辞書を自動判定する。
    """
    required = {"重賞名", "条件ランク", "条件名"}

    # 1) まず全CSVを読んで、中身でロジック辞書を探す
    logic_df = pd.DataFrame()
    csv_paths = sorted(BASE_DIR.glob("*.csv"))
    for path in csv_paths:
        df = read_csv_auto(path)
        if not df.empty and required.issubset(set(df.columns)):
            logic_df = df
            break

    # 2) 見つからない場合だけ従来のファイル名探索
    if logic_df.empty:
        logic_path = DEFAULT_LOGIC if DEFAULT_LOGIC.exists() else find_csv_by_keyword("logic")
        logic_df = read_csv_auto(logic_path)

    # 消し条件・サマリーはファイル名で読む。失敗してもアプリは動く
    keshi_path = DEFAULT_KESHI if DEFAULT_KESHI.exists() else find_csv_by_keyword("keshi")
    summary_path = DEFAULT_SUMMARY if DEFAULT_SUMMARY.exists() else find_csv_by_keyword("summary")
    keshi_df = read_csv_auto(keshi_path)
    summary_df = read_csv_auto(summary_path)

    return logic_df, keshi_df, summary_df


def normalize_bool(x) -> bool:
    if pd.isna(x):
        return False
    s = str(x).strip()
    if s in TRUE_VALUES:
        return True
    if s in FALSE_VALUES:
        return False
    # 条件列に理由文が入った場合も、明確な否定がなければ該当扱いにしない
    return False


def clean_name(s: str) -> str:
    return str(s).strip().replace("\n", " ")


def condition_columns(logic_sub: pd.DataFrame, rank_filter: List[str]) -> List[str]:
    df = logic_sub[logic_sub["条件ランク"].isin(rank_filter)].copy()
    return df.sort_values(["表示順", "条件ランク", "条件名"])["条件名"].astype(str).tolist()


def keshi_columns(keshi_sub: pd.DataFrame, max_items: int) -> List[str]:
    if keshi_sub.empty:
        return []
    df = keshi_sub.sort_values(["リフト", "馬券内該当数", "全出走該当率"], ascending=[True, True, False]).head(max_items)
    return df["消し条件候補"].astype(str).tolist()


def build_prompt(race_name: str, logic_sub: pd.DataFrame, keshi_sub: pd.DataFrame, year: str, include_ranks: List[str], max_keshi: int) -> str:
    show_df = logic_sub[logic_sub["条件ランク"].isin(include_ranks)].copy()
    show_df = show_df.sort_values(["表示順", "条件ランク", "条件名"])
    keshi_list = keshi_columns(keshi_sub, max_keshi)

    base_cols = ["重賞名", "年", "馬番", "馬名"]
    cond_cols = show_df["条件名"].astype(str).tolist()
    cols = base_cols + cond_cols
    if keshi_list:
        cols += ["消し条件該当", "消し条件名"]
    cols += ["補足"]
    header = ",".join(cols)

    lines = []
    lines.append("あなたは重賞⭐️抽出用CSV作成AIです。")
    lines.append("")
    lines.append(f"【対象レース】{year}年 {race_name}")
    lines.append("")
    lines.append("【目的】")
    lines.append("この重賞の過去10年1〜3着馬から抽出された再現性フィルターに、今年の出走馬が該当するかを判定してください。")
    lines.append("ランキングではなく、アプリで⭐️候補を1頭抽出するための該当CSVを作成します。")
    lines.append("")
    lines.append("【最重要：出力形式】")
    lines.append("・最終回答は必ずCSV本文のみ")
    lines.append("・Markdown、コードブロック、説明文、箇条書き、表形式は一切禁止")
    lines.append("・回答の1文字目は必ずCSVヘッダーの『重賞名』から始める")
    lines.append("・CSV以外の前置き、後書き、注釈、確認コメントは一切入れない")
    lines.append("・全出走馬を1頭1行で出力する")
    lines.append("・列数と列名は下のCSVヘッダーと完全一致させる")
    lines.append("・条件に該当する場合は○、非該当は×で記入")
    lines.append("・判断不能な条件は空欄ではなく×にする")
    lines.append("・カンマを含む補足文は使わない。補足は短く、句読点は『、』ではなく『・』を使う")
    lines.append("")
    lines.append("【禁止】")
    lines.append("・オッズ、人気、予想印、AI指数、他者評価は使わない")
    lines.append("・レース結果後にしか分からない情報で補正しない")
    lines.append("・推測で雑に埋めない。分からない場合は×")
    lines.append("・ランキングや買い目や本命印は出さない")
    lines.append("")
    lines.append("【この重賞で見る条件】")
    for rank in ["S", "A", "B", "補助"]:
        part = show_df[show_df["条件ランク"] == rank]
        if part.empty:
            continue
        lines.append(f"\n■{rank}条件")
        for _, r in part.iterrows():
            lines.append(f"・{r['条件名']}（{r.get('根拠','')}）")
    if keshi_list:
        lines.append("\n■消し条件候補")
        for k in keshi_list:
            lines.append(f"・{k}")
    lines.append("")
    lines.append("【CSVヘッダー】")
    lines.append(header)
    lines.append("")
    lines.append("【出力例：形式だけ参考。馬名や内容は実データで作成】")
    sample_values = []
    for c in cols:
        if c == "重賞名":
            sample_values.append(race_name)
        elif c == "年":
            sample_values.append(year)
        elif c == "馬番":
            sample_values.append("1")
        elif c == "馬名":
            sample_values.append("サンプルホース")
        elif c == "消し条件該当":
            sample_values.append("×")
        elif c == "消し条件名":
            sample_values.append("")
        elif c == "補足":
            sample_values.append("条件該当のみ簡潔に記載")
        else:
            sample_values.append("○")
    lines.append(",".join(sample_values))
    lines.append("")
    lines.append("【最終確認】")
    lines.append("この依頼への回答は、上記CSVヘッダーから始まるCSV本文のみで返してください。")
    return "\n".join(lines)


def build_result_prompt(race_name: str, year: str) -> str:
    """レース結果をCSVだけで返してもらうためのプロンプト。"""
    header = "重賞名,年,馬名,着順,単勝配当,複勝配当"
    lines = []
    lines.append("あなたは競馬結果整理AIです。")
    lines.append("")
    lines.append(f"【対象レース】{year}年 {race_name}")
    lines.append("")
    lines.append("【目的】")
    lines.append("ネット上の公開情報で対象重賞の確定結果を確認し、アプリに貼り付ける結果CSVを作成してください。")
    lines.append("")
    lines.append("【最重要：出力形式】")
    lines.append("・最終回答は必ずCSV本文のみ")
    lines.append("・Markdown、コードブロック、説明文、箇条書き、表形式は禁止")
    lines.append("・回答の1文字目は必ずCSVヘッダーの『重賞名』から始める")
    lines.append("・CSV以外の前置き、後書き、注釈、確認コメントは一切入れない")
    lines.append("・1〜3着馬だけを1頭1行で出力する")
    lines.append("・列名と列数は下のCSVヘッダーと完全一致させる")
    lines.append("・単勝配当と複勝配当は円表記でよい。存在しない配当は空欄")
    lines.append("・馬名はアプリ側の判定CSVと照合するため正式馬名で書く")
    lines.append("")
    lines.append("【CSVヘッダー】")
    lines.append(header)
    lines.append("")
    lines.append("【出力例：形式だけ参考】")
    lines.append(f"{race_name},{year},サンプルホース,1,180円,110円")
    lines.append(f"{race_name},{year},サンプルホース2,2,,160円")
    lines.append(f"{race_name},{year},サンプルホース3,3,,220円")
    lines.append("")
    lines.append("この依頼への回答は、上記CSVヘッダーから始まるCSV本文のみで返してください。")
    return "\n".join(lines)


def normalize_horse_name(x) -> str:
    return str(x).strip().replace(" ", "").replace("　", "")


def to_numeric_pay(x):
    if pd.isna(x):
        return 0
    s = str(x).replace("円", "").replace(",", "").strip()
    if s in ("", "None", "none", "nan"):
        return 0
    try:
        return int(float(s))
    except Exception:
        return 0


def make_update_df(scored_df: pd.DataFrame, result_df: pd.DataFrame, race_name: str, year: str) -> pd.DataFrame:
    """③判定結果と結果CSVを照合し、成績更新用データを作る。"""
    if scored_df is None or scored_df.empty:
        return pd.DataFrame()
    if result_df is None or result_df.empty:
        return pd.DataFrame()
    pred = scored_df.copy()
    res = result_df.copy()
    if "馬名" not in pred.columns or "馬名" not in res.columns:
        return pd.DataFrame()
    pred["照合馬名"] = pred["馬名"].apply(normalize_horse_name)
    res["照合馬名"] = res["馬名"].apply(normalize_horse_name)
    keep_res = [c for c in ["照合馬名", "着順", "単勝配当", "複勝配当"] if c in res.columns]
    merged = pred.merge(res[keep_res], on="照合馬名", how="left")
    if "重賞名" not in merged.columns:
        merged.insert(0, "重賞名", race_name)
    if "年" not in merged.columns:
        merged.insert(1, "年", year)
    merged["着順_num"] = pd.to_numeric(merged.get("着順"), errors="coerce")
    merged["馬券内"] = merged["着順_num"].between(1, 3)
    merged["単勝配当_num"] = merged.get("単勝配当", pd.Series([0]*len(merged))).apply(to_numeric_pay)
    merged["複勝配当_num"] = merged.get("複勝配当", pd.Series([0]*len(merged))).apply(to_numeric_pay)
    merged["⭐️判定"] = merged["暫定順位"].apply(lambda x: "⭐️" if x == 1 else "") if "暫定順位" in merged.columns else ""
    merged["⭐️的中"] = merged.apply(lambda r: bool(r.get("⭐️判定") == "⭐️" and r.get("馬券内") == True), axis=1)
    out_cols = [c for c in [
        "重賞名", "年", "馬番", "馬名", "暫定順位", "⭐️判定", "着順", "馬券内", "⭐️的中",
        "単勝配当", "複勝配当", "S該当数", "A該当数", "B該当数", "補助該当数", "条件スコア", "消し該当", "消し条件名", "該当条件一覧"
    ] if c in merged.columns]
    return merged[out_cols]

def score_prediction(pred_df: pd.DataFrame, logic_sub: pd.DataFrame, keshi_sub: pd.DataFrame, include_ranks: List[str]) -> pd.DataFrame:
    df = pred_df.copy()
    if df.empty:
        return df

    if "馬名" not in df.columns:
        # ありがちな表記揺れを補正
        for c in df.columns:
            if "馬名" in str(c):
                df = df.rename(columns={c: "馬名"})
                break
    if "馬名" not in df.columns:
        st.error("CSVに『馬名』列がありません。")
        return pd.DataFrame()

    logic_use = logic_sub[logic_sub["条件ランク"].isin(include_ranks)].copy()
    cond_meta = logic_use[["条件名", "条件ランク"]].drop_duplicates()

    # 存在しない条件列はFalse扱い
    s_counts, a_counts, b_counts, h_counts, scores, matched = [], [], [], [], [], []
    for _, row in df.iterrows():
        score = 0.0
        s = a = b = h = 0
        hit_names = []
        for _, m in cond_meta.iterrows():
            col = str(m["条件名"])
            rank = str(m["条件ランク"])
            hit = normalize_bool(row[col]) if col in df.columns else False
            if hit:
                score += RANK_WEIGHT.get(rank, 0.0)
                hit_names.append(f"{rank}:{col}")
                if rank == "S": s += 1
                elif rank == "A": a += 1
                elif rank == "B": b += 1
                else: h += 1
        s_counts.append(s); a_counts.append(a); b_counts.append(b); h_counts.append(h); scores.append(score); matched.append(" / ".join(hit_names))

    df["S該当数"] = s_counts
    df["A該当数"] = a_counts
    df["B該当数"] = b_counts
    df["補助該当数"] = h_counts
    df["条件スコア"] = scores
    df["該当条件一覧"] = matched

    if "消し条件該当" in df.columns:
        df["消し該当"] = df["消し条件該当"].apply(normalize_bool)
    else:
        df["消し該当"] = False

    df["有効スコア"] = df.apply(lambda r: -999.0 if r["消し該当"] else r["条件スコア"], axis=1)
    df = df.sort_values(["有効スコア", "S該当数", "A該当数", "B該当数"], ascending=False).reset_index(drop=True)
    df.insert(0, "暫定順位", range(1, len(df) + 1))
    return df


def judge_star(scored: pd.DataFrame, min_s_rate: float, min_score_gap: float) -> Tuple[str, str, Optional[str]]:
    if scored.empty:
        return "見送り", "判定対象がありません。", None
    valid = scored[~scored["消し該当"]].copy()
    if valid.empty:
        return "見送り", "全馬が消し条件に該当しています。", None

    total_s = int(scored[[c for c in ["S該当数"] if c in scored.columns]]["S該当数"].max()) if "S該当数" in scored.columns else 0
    # 実際のS条件総数を外から渡していないので、行のS該当最大値からではなく後で補正できない。ここではシンプルに有効スコア重視。
    top = valid.iloc[0]
    second_score = valid.iloc[1]["有効スコア"] if len(valid) >= 2 else -999
    gap = float(top["有効スコア"] - second_score)

    if len(valid) >= 2 and gap < min_score_gap:
        return "準⭐️複数/見送り", f"1位と2位の差が{gap:.1f}点で小さいため、単独⭐️にはしません。", str(top["馬名"])
    if top["有効スコア"] <= 0:
        return "見送り", "有効スコアが低く、⭐️条件として弱いです。", str(top["馬名"])
    return "⭐️", f"消し条件なし、かつ条件一致が単独最上位です。2位との差は{gap:.1f}点です。", str(top["馬名"])


def csv_download(df: pd.DataFrame, filename: str, label: str):
    data = df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(label, data=data, file_name=filename, mime="text/csv")


# ---------------- UI ----------------
st.title(APP_TITLE)
st.caption("重賞ごとの過去10年1〜3着馬フィルターを使い、ランキングではなく⭐️候補1頭を抽出するためのアプリです。")

logic_df, keshi_df, summary_df = load_default_logic()

with st.sidebar:
    st.header("データ読み込み")
    up_logic = st.file_uploader("ロジック辞書CSV（任意）", type=["csv"], key="logic")
    up_keshi = st.file_uploader("消し条件候補CSV（任意）", type=["csv"], key="keshi")
    if up_logic is not None:
        logic_df = read_csv_auto(up_logic)
    if up_keshi is not None:
        keshi_df = read_csv_auto(up_keshi)

    st.divider()
    st.header("判定設定")
    include_ranks = st.multiselect("使用する条件ランク", ["S", "A", "B", "補助"], default=["S", "A", "B"])
    max_keshi = st.slider("プロンプトに入れる消し候補数", 0, 20, 8)
    min_gap = st.number_input("単独⭐️に必要な2位との差", min_value=0.0, max_value=20.0, value=2.0, step=0.5)

required_cols = {"重賞名", "条件ランク", "条件名"}
if logic_df.empty or not required_cols.issubset(set(logic_df.columns)):
    st.error("ロジック辞書CSVが読み込めません。『重賞名・条件ランク・条件名』列が必要です。")
    st.write("現在アプリが認識しているCSVファイル一覧：")
    st.write([p.name for p in BASE_DIR.glob("*.csv")])
    if not logic_df.empty:
        st.write("読み込めたCSVの列名：")
        st.write(list(logic_df.columns))
    st.info("GitHub上に『重賞別_プロンプト用ロジック辞書.csv』があるか確認してください。ファイル名が少し違う場合は、この画面の一覧を見て原因を確認できます。")
    st.stop()

races = sorted(logic_df["重賞名"].dropna().astype(str).unique().tolist())
selected_race = st.selectbox("対象重賞を選択", races)
year = st.text_input("対象年", value="2026")

logic_sub = logic_df[logic_df["重賞名"].astype(str) == selected_race].copy()
keshi_sub = keshi_df[keshi_df["重賞名"].astype(str) == selected_race].copy() if not keshi_df.empty and "重賞名" in keshi_df.columns else pd.DataFrame()
summary_sub = summary_df[summary_df["重賞名"].astype(str) == selected_race].copy() if not summary_df.empty and "重賞名" in summary_df.columns else pd.DataFrame()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["①ロジック確認", "②プロンプト作成", "③該当CSV判定", "④結果更新"])

with tab1:
    st.subheader(f"{selected_race} の抽出ロジック")
    if not summary_sub.empty:
        st.dataframe(summary_sub, use_container_width=True, hide_index=True)
    c1, c2 = st.columns([2, 1])
    with c1:
        show_cols = [c for c in ["表示順", "条件ランク", "条件名", "条件タイプ", "根拠", "年カバー数", "馬券内該当数", "リフト"] if c in logic_sub.columns]
        st.dataframe(logic_sub.sort_values(["表示順", "条件ランク", "条件名"])[show_cols], use_container_width=True, hide_index=True)
        csv_download(logic_sub, f"{selected_race}_ロジック条件.csv", "この重賞の条件CSVをダウンロード")
    with c2:
        st.markdown("#### 消し条件候補")
        if keshi_sub.empty:
            st.info("消し条件候補はありません。")
        else:
            show_k = [c for c in ["消し条件候補", "年カバー数", "馬券内該当数", "全出走該当率", "リフト"] if c in keshi_sub.columns]
            st.dataframe(keshi_sub[show_k].head(30), use_container_width=True, hide_index=True)

with tab2:
    st.subheader("専用プロンプト")
    prompt = build_prompt(selected_race, logic_sub, keshi_sub, year, include_ranks, max_keshi)
    st.text_area("この内容をコピーしてChatGPTに貼り付け", value=prompt, height=520)
    st.download_button("プロンプトをtxtでダウンロード", data=prompt.encode("utf-8"), file_name=f"{selected_race}_{year}_星抽出プロンプト.txt", mime="text/plain")

with tab3:
    st.subheader("今年の出走馬 該当CSVを貼り付け/アップロード")
    st.caption("②のプロンプトで作成したCSVを貼り付けると、条件一致数から⭐️候補を判定します。")
    up_pred = st.file_uploader("該当CSVアップロード", type=["csv"], key="pred_csv")
    pasted = st.text_area("またはCSVを直接貼り付け", height=220, placeholder="重賞名,年,馬番,馬名,...")

    pred_df = pd.DataFrame()
    if up_pred is not None:
        pred_df = read_csv_auto(up_pred)
    elif pasted.strip():
        pred_df = pd.read_csv(io.StringIO(pasted.strip()))

    if not pred_df.empty:
        scored = score_prediction(pred_df, logic_sub, keshi_sub, include_ranks)
        if not scored.empty:
            status, reason, star_name = judge_star(scored, 0.0, min_gap)
            st.session_state["last_scored"] = scored
            st.session_state["last_status"] = status
            st.session_state["last_star_name"] = star_name
            st.session_state["last_race_name"] = selected_race
            st.session_state["last_year"] = year
            if status == "⭐️":
                st.success(f"⭐️候補：{star_name}\n\n{reason}")
            elif "準" in status:
                st.warning(f"{status}：暫定最上位は {star_name}\n\n{reason}")
            else:
                st.info(f"{status}\n\n{reason}")

            out_cols = [c for c in ["暫定順位", "馬番", "馬名", "S該当数", "A該当数", "B該当数", "補助該当数", "条件スコア", "消し該当", "消し条件名", "補足", "該当条件一覧"] if c in scored.columns]
            st.dataframe(scored[out_cols], use_container_width=True, hide_index=True)
            csv_download(scored, f"{selected_race}_{year}_星判定結果.csv", "判定結果CSVをダウンロード")

with tab4:
    st.subheader("結果更新")
    st.caption("結果プロンプトをコピーして確定結果CSVを作成し、③の判定結果と照合して成績を更新します。")

    result_prompt = build_result_prompt(selected_race, year)
    st.markdown("#### 結果更新用プロンプト")
    st.text_area("この内容をコピーしてChatGPTに貼り付け", value=result_prompt, height=360)
    st.download_button("結果更新プロンプトをtxtでダウンロード", data=result_prompt.encode("utf-8"), file_name=f"{selected_race}_{year}_結果更新プロンプト.txt", mime="text/plain")

    st.markdown("---")
    st.markdown("#### 結果CSV読み込み")
    st.markdown("必要列：`重賞名,年,馬名,着順`。任意列：`単勝配当,複勝配当`。")
    up_result = st.file_uploader("結果CSVアップロード", type=["csv"], key="result_csv")
    pasted_result = st.text_area("または結果CSVを直接貼り付け", height=160, placeholder="重賞名,年,馬名,着順,単勝配当,複勝配当")

    res_df = pd.DataFrame()
    if up_result is not None:
        res_df = read_csv_auto(up_result)
    elif pasted_result.strip():
        res_df = pd.read_csv(io.StringIO(pasted_result.strip()))

    scored_for_update = st.session_state.get("last_scored", pd.DataFrame())
    if scored_for_update is None or scored_for_update.empty:
        st.warning("先に③該当CSV判定で判定結果を作ってください。結果更新はその判定結果と照合します。")

    # 任意で過去の成績履歴をアップロードして追記できるようにする
    st.markdown("#### 成績履歴に追記する場合")
    history_file = st.file_uploader("既存の成績履歴CSV（任意）", type=["csv"], key="history_csv")
    history_df = read_csv_auto(history_file) if history_file is not None else pd.DataFrame()

    if not res_df.empty:
        st.dataframe(res_df, use_container_width=True, hide_index=True)
        if not {"馬名", "着順"}.issubset(res_df.columns):
            st.error("結果CSVには『馬名』『着順』列が必要です。")
        elif scored_for_update is not None and not scored_for_update.empty:
            update_df = make_update_df(scored_for_update, res_df, selected_race, year)
            if update_df.empty:
                st.error("③の判定結果と結果CSVを照合できませんでした。馬名表記を確認してください。")
            else:
                st.markdown("#### 今回の更新内容")
                st.dataframe(update_df, use_container_width=True, hide_index=True)

                star_rows = update_df[update_df.get("⭐️判定", "") == "⭐️"] if "⭐️判定" in update_df.columns else pd.DataFrame()
                if not star_rows.empty:
                    r = star_rows.iloc[0]
                    if bool(r.get("馬券内", False)):
                        st.success(f"⭐️ {r.get('馬名')} は {r.get('着順')}着。馬券内的中です。")
                    else:
                        st.warning(f"⭐️ {r.get('馬名')} は {r.get('着順')}着。今回は馬券外です。")

                if st.button("この結果を成績履歴に反映", type="primary"):
                    if not history_df.empty:
                        combined = pd.concat([history_df, update_df], ignore_index=True)
                    else:
                        combined = update_df.copy()
                    st.session_state["updated_history"] = combined
                    st.success("アプリ内の一時履歴に反映しました。下のボタンからCSVをダウンロードしてください。")

                csv_download(update_df, f"{selected_race}_{year}_今回結果更新.csv", "今回の結果更新CSVをダウンロード")

    updated_history = st.session_state.get("updated_history", pd.DataFrame())
    if updated_history is not None and not updated_history.empty:
        st.markdown("#### 更新後の成績履歴")
        st.dataframe(updated_history, use_container_width=True, hide_index=True)
        csv_download(updated_history, "重賞星_成績履歴_更新後.csv", "更新後の成績履歴CSVをダウンロード")

    st.info("Streamlit Community Cloudでは、通常のボタン操作だけでGitHub上のCSVを直接書き換えて永続保存することはできません。現在の方式は、アプリ内で反映 → 更新後CSVをダウンロード → 必要に応じてGitHubへ再アップロード、という安全な運用です。")
