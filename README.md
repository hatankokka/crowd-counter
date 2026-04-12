# 🧑‍🤝‍🧑 空撮・集合写真 人数カウンター

空撮写真やデモ・集合写真の人数を推定するWebアプリです。  
Streamlit + Python で構築。**完全無料**で動作・公開できます。

## 機能

| モード | 説明 | 向いてる画像 |
|--------|------|------------|
| ✋ 手動クリック | 座標を指定して一人ずつマーク | どんな画像でも |
| 🤖 YOLO自動検出 | YOLOv8で人を自動検出 | 低〜中密度・人が識別できるサイズ |
| 🌡️ CSRNet密度推定 | 密度マップで人数を推定 | **高密度・空撮デモ写真** |

## デモ

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

> ※ デプロイ後に上記URLを自分のアプリのURLに変更してください

## ローカルでの実行

```bash
# リポジトリをクローン
git clone https://github.com/YOUR_USERNAME/crowd-counter.git
cd crowd-counter

# 依存ライブラリをインストール
pip install -r requirements.txt

# アプリを起動
streamlit run app.py
```

ブラウザで `http://localhost:8501` が開きます。

## Streamlit Cloud での公開手順

1. このリポジトリを GitHub にプッシュ
2. [share.streamlit.io](https://share.streamlit.io) にアクセス
3. GitHub アカウントでログイン
4. "New app" → リポジトリを選択 → `app.py` を指定
5. "Deploy!" をクリック

**無料枠:** CPU のみ・メモリ 1GB・常時稼働

## CSRNet 学習済み重みについて

高精度の密度推定には学習済み重みが必要です。

**自動ダウンロード:** アプリ起動時に自動でダウンロードを試みます。

**手動ダウンロード（自動が失敗した場合）:**
1. [CSRNet-pytorch](https://github.com/CommissarMa/CSRNet-pytorch) から重みを取得
2. `csrnet_partA.pth` という名前でこのフォルダに配置
3. アプリを再起動

> 重みがない場合でも VGG16 frontend のみのフォールバックモードで動作します（精度は下がります）

## 注意事項

- **推定値について:** CSRNet/YOLO の出力は推定値です。撮影角度・遮蔽・照明により誤差が生じます
- **ライセンス:** YOLOv8 は AGPL-3.0。本アプリを商用利用する場合は Ultralytics のライセンスを確認してください
- **プライバシー:** アップロードした画像はサーバーに保存されません（Streamlit Cloud のセッション内のみ）

## 技術スタック

- [Streamlit](https://streamlit.io) - Web UI
- [YOLOv8](https://github.com/ultralytics/ultralytics) - 物体検出
- [SAHI](https://github.com/obss/sahi) - 高解像度分割推論
- [CSRNet](https://github.com/CommissarMa/CSRNet-pytorch) - 群衆密度推定（Li et al., CVPR 2018）
- OpenCV / PyTorch / Pillow

## License

MIT
