# 🧑‍🤝‍🧑 空撮・集合写真 人数カウンター

空撮写真やデモ・集合写真の人数をAIで推定するWebアプリです。  
完全無料・登録不要で使えます。

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

---

## どちらのモードを使えばいい？

### 🤖 YOLO自動検出 を使う場合
- 一人ひとりの姿がはっきり見える写真
- 人と人の間に隙間がある（密集しすぎていない）
- 運動会・卒業式・小規模なイベントの写真

### 🌡️ CSRNet密度推定 を使う場合
- デモ行進・お祭り・花火大会など人が密集した写真
- 空から撮った写真（空撮・ドローン映像）
- 数百〜数万人規模の群衆を推定したいとき

> 迷ったら両方試して、結果を比べてみてください。

---

## 使い方

1. アプリを開く
2. 写真をアップロード（JPG / PNG / WEBP）
3. タブを選んで推定を実行
4. 結果画像をダウンロード

---

## ローカルで動かす場合

```bash
git clone https://github.com/YOUR_USERNAME/crowd-counter.git
cd crowd-counter
pip install -r requirements.txt
streamlit run app.py
```

ブラウザで `http://localhost:8501` が開きます。

---

## Streamlit Cloud で公開する手順

1. このリポジトリを GitHub に push
2. [share.streamlit.io](https://share.streamlit.io) にアクセス
3. GitHub アカウントでログイン
4. "New app" → リポジトリを選択 → `app.py` を指定
5. "Deploy!" をクリック → 数分で公開完了

---

## CSRNet の学習済み重みについて

アプリ起動時に自動ダウンロードを試みます。  
失敗した場合は以下の手順で手動ダウンロードしてください。

1. [CSRNet-pytorch](https://github.com/CommissarMa/CSRNet-pytorch) を開く
2. README のリンクから Google Drive にアクセス
3. `csrnet_partA.pth` という名前で `app.py` と同じフォルダに保存
4. アプリを再起動

重みがない場合もテクスチャ解析ベースの簡易推定で動作します。

---

## 注意事項

- 推定値には誤差があります。撮影角度・遮蔽・照明の影響を受けます
- アップロードした画像はサーバーに保存されません
- YOLO は AGPL-3.0 ライセンスです。商用利用の場合は確認してください

---

## 技術スタック

- [Streamlit](https://streamlit.io)
- [YOLOv8](https://github.com/ultralytics/ultralytics) + [SAHI](https://github.com/obss/sahi)
- [CSRNet](https://github.com/CommissarMa/CSRNet-pytorch)（Li et al., CVPR 2018）
- OpenCV / PyTorch / Pillow
