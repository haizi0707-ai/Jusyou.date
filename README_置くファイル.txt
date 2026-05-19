【GitHub 25MB制限対応・軽量版】

app.py と同じフォルダに置くファイルは以下です。

1. app.py
2. pace_condition_summary.csv
3. jockey_pace_summary.csv
4. trainer_pace_summary.csv
5. 予想用.csv（テンプレート。予想時に中身を差し替え）

巨大な以下のファイルは置かなくて大丈夫です。
・ペース１０年.csv
・調教師１０年.csv

この軽量版app.pyは、過去10年の生データではなく、集計済みCSVを読み込みます。
そのためGitHubの25MB制限に引っかかりません。

実行コマンド：
streamlit run app.py
