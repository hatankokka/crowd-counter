"""
空撮・集合写真 人数カウンター
Streamlit版 - GitHub + Streamlit Cloud で無料公開
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import json
import datetime
import io

# ─────────────────────────────────────────
# ページ設定
# ─────────────────────────────────────────
st.set_page_config(
    page_title="群衆人数カウンター",
    page_icon="🧑‍🤝‍🧑",
    layout="wide",
)

st.title("🧑‍🤝‍🧑 群衆カウンター")

with st.expander("📖 使い方・モード選択ガイド（はじめての方はこちら）", expanded=False):
    st.markdown("""
### どちらのタブを使えばいい？

| | 🤖 YOLO自動検出 | 🌡️ CSRNet密度推定 |
|---|---|---|
| **向いてる写真** | 一人ひとりが識別できる | 人が密集して見分けられない |
| **シーン例** | 運動会・卒業式・小規模イベント | デモ行進・花火大会・空撮 |
| **人数規模** | ～数百人 | 数百～数万人 |

---

#### 🤖 YOLO を使うのはこんなとき
- 写真の中で**一人ひとりの姿がはっきり見える**
- 人と人の間に隙間がある（密集しすぎていない）
- 運動会・卒業式・小規模なイベントの写真

#### 🌡️ CSRNet を使うのはこんなとき
- デモ行進・お祭り・花火大会など**人が密集した**写真
- 空から撮った写真（空撮・ドローン映像）
- 数百～数万人規模の群衆を推定したいとき

> 💡 迷ったら両方試して、結果を比べてみてください。

---

### 基本的な使い方
1. **写真をアップロード**（JPG / PNG / WEBP）
2. **タブを選んで**「実行」ボタンを押す
3. **結果を確認**して、必要なら画像をダウンロード

### 注意
1. **選んだモデルにより正確さに差が出ます。いろいろ試してみてください**
2. このアプリは**絶対的な数を保証するものではありません**
""")


# ─────────────────────────────────────────
# 共通ユーティリティ
# ─────────────────────────────────────────

def pil_to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

def bgr_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

def draw_points(img_array: np.ndarray, pts: list, auto_ids: set = None) -> np.ndarray:
    """
    点リストを画像上に描画。
    auto_ids に含まれるindexは青丸、それ以外は緑丸。
    300点以下なら番号も表示。
    """
    out = img_array.copy()
    n = len(pts)
    show_numbers = n <= 300
    auto_ids = auto_ids or set()

    for i, (x, y) in enumerate(pts, 1):
        color = (230, 80, 0) if (i - 1) in auto_ids else (0, 230, 80)
        cv2.circle(out, (x, y), 9, (0, 0, 0), -1)
        cv2.circle(out, (x, y), 8, color, -1)
        cv2.circle(out, (x, y), 8, (255, 255, 255), 1)
        if show_numbers:
            label = str(i)
            fs = 0.35 if n > 100 else 0.45
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            tx, ty = x - tw // 2, y + th // 2
            cv2.putText(out, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(out, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        (255, 255, 255), 1, cv2.LINE_AA)
    return out


def pil_to_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# ─────────────────────────────────────────
# セッション状態の初期化
# ─────────────────────────────────────────
defaults = {
    "original_bgr": None,
    "points": [],
    "auto_ids": set(),
    "history": [],
    "last_uploaded_name": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────
# タブでモード選択（モバイル対応）
# ─────────────────────────────────────────
tab_yolo, tab_csrnet = st.tabs([
    "🤖 YOLO自動検出",
    "🌡️ CSRNet密度推定",
])


# ─────────────────────────────────────────
# 画像アップロード（タブ共通）
# ─────────────────────────────────────────
uploaded = st.file_uploader(
    "📂 画像をアップロード（JPG / PNG / WEBP）",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)

if uploaded is not None:
    if uploaded.name != st.session_state.last_uploaded_name:
        img_pil = Image.open(uploaded).convert("RGB")
        st.session_state.original_bgr = pil_to_bgr(img_pil)
        st.session_state.points = []
        st.session_state.auto_ids = set()
        st.session_state.history = []
        st.session_state.last_uploaded_name = uploaded.name
        st.success(f"✅ 読み込み完了: {uploaded.name}  ({img_pil.width}×{img_pil.height}px)")

# ─────────────────────────────────────────
# タブ B: YOLO自動検出
# ─────────────────────────────────────────
with tab_yolo:
    st.subheader("🤖 YOLO自動検出")
    st.info("""
**こんなときに使ってください 👇**

✅ 写真の中で**一人ひとりの姿がはっきり見える**とき（顔や体が識別できる）
✅ 人と人の間に多少隙間があるとき
✅ 運動会・卒業式・小規模なイベントの写真など

❌ 人が密集しすぎて個人を識別できない写真には向いていません
　→ その場合は「🌡️ CSRNet密度推定」タブを使ってください
""")
    st.caption("🔵 青丸 = AIが自動で検出した人　🟢 緑丸 = 手動で追加した人")

    # YOLOのインポートを試みる
    try:
        from ultralytics import YOLO as _YOLO
        yolo_available = True
    except ImportError:
        yolo_available = False

    if not yolo_available:
        st.error("UltralyticsがインストールされていませんPip install ultralyticsを実行してください")
        st.code("pip install ultralytics sahi")
        st.stop()

    if st.session_state.original_bgr is None:
        st.warning("👆 まず画像をアップロードしてください")
        st.stop()

    orig = st.session_state.original_bgr
    h, w = orig.shape[:2]

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("**検出設定**")
        conf = st.slider("信頼度しきい値", 0.1, 0.8, 0.3, 0.05, key="yolo_conf",
                         help="低いほど検出数が増えるが誤検出も増える")
        use_sahi = st.checkbox("🔍 高解像度モード（SAHI）", key="yolo_sahi",
                               help="空撮など大きい画像向け。処理が重くなる")
        model_size = st.selectbox("モデルサイズ",
                                   ["yolov8n", "yolov8s", "yolov8m"],
                                   index=0,
                                   help="n=軽量 / s=バランス / m=高精度",
                                   key="yolo_model")

        if st.button("🤖 自動検出を実行", type="primary", use_container_width=True, key="yolo_detect"):
            with st.spinner("検出中..."):
                try:
                    model = _YOLO(f"{model_size}.pt")
                    img_pil = bgr_to_pil(orig)

                    if use_sahi:
                        try:
                            from sahi import AutoDetectionModel
                            from sahi.predict import get_sliced_prediction
                            sahi_model = AutoDetectionModel.from_pretrained(
                                model_type="yolov8",
                                model_path=f"{model_size}.pt",
                                confidence_threshold=conf,
                            )
                            sl = max(320, min(w, h) // 3)
                            result = get_sliced_prediction(
                                img_pil, sahi_model,
                                slice_height=sl, slice_width=sl,
                                overlap_height_ratio=0.2, overlap_width_ratio=0.2,
                            )
                            detected = [
                                (int((p.bbox.minx+p.bbox.maxx)/2),
                                 int(p.bbox.miny+(p.bbox.maxy-p.bbox.miny)*0.15))
                                for p in result.object_prediction_list
                                if p.category.name == "person"
                            ]
                        except ImportError:
                            st.warning("SAHIが未インストールです。通常モードで実行します。")
                            results = model(img_pil, conf=conf, classes=[0])
                            detected = [
                                ((x1+x2)//2, int(y1+(y2-y1)*0.15))
                                for box in results[0].boxes
                                for x1,y1,x2,y2 in [map(int, box.xyxy[0])]
                            ]
                    else:
                        results = model(img_pil, conf=conf, classes=[0])
                        detected = [
                            ((x1+x2)//2, int(y1+(y2-y1)*0.15))
                            for box in results[0].boxes
                            for x1,y1,x2,y2 in [map(int, box.xyxy[0])]
                        ]

                    st.session_state.history.append(
                        (st.session_state.points.copy(), st.session_state.auto_ids.copy())
                    )
                    n_prev = len(st.session_state.points)
                    st.session_state.points.extend(detected)
                    for i in range(n_prev, len(st.session_state.points)):
                        st.session_state.auto_ids.add(i)

                    st.success(f"✅ 検出完了: {len(detected)}人")
                    st.rerun()

                except Exception as e:
                    st.error(f"検出エラー: {e}")

        st.divider()
        st.markdown("**手動補正**")
        c1, c2 = st.columns(2)
        with c1:
            add_x = st.number_input("追加X", 0, w-1, w//2, key="add_x")
        with c2:
            add_y = st.number_input("追加Y", 0, h-1, h//2, key="add_y")
        if st.button("➕ 手動追加", key="yolo_add"):
            st.session_state.history.append(
                (st.session_state.points.copy(), st.session_state.auto_ids.copy()))
            st.session_state.points.append((int(add_x), int(add_y)))
            st.rerun()

        bc1, bc2 = st.columns(2)
        with bc1:
            if st.button("↩ 戻す", use_container_width=True, key="yolo_undo"):
                if st.session_state.history:
                    h_pts, h_ids = st.session_state.history.pop()
                    st.session_state.points = h_pts
                    st.session_state.auto_ids = h_ids
                st.rerun()
        with bc2:
            if st.button("🔄 リセット", use_container_width=True, key="yolo_reset"):
                st.session_state.points = []
                st.session_state.auto_ids = set()
                st.session_state.history = []
                st.rerun()

        st.metric("合計人数", len(st.session_state.points),
                  delta=f"自動:{len(st.session_state.auto_ids)} 手動:{len(st.session_state.points)-len(st.session_state.auto_ids)}")

    with col2:
        result = draw_points(orig, st.session_state.points, st.session_state.auto_ids)
        st.image(bgr_to_pil(result), use_container_width=True)

        if st.session_state.points:
            result_pil = bgr_to_pil(result)
            st.download_button(
                "💾 結果画像をダウンロード",
                data=pil_to_bytes(result_pil),
                file_name=f"crowd_yolo_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png",
                use_container_width=True,
                key="dl_yolo_img",
            )


# ─────────────────────────────────────────
# タブ C: CSRNet密度推定 + 複数手法比較
# ─────────────────────────────────────────
with tab_csrnet:
    st.subheader("🌡️ CSRNet密度推定")
    st.info("""
**こんなときに使ってください 👇**

✅ デモ行進・お祭り・花火大会など**人が密集した**写真
✅ 空から撮った写真（空撮・ドローン映像）
✅ 人の顔や体がよく見えないくらい密集しているとき
✅ 数百〜数万人規模の群衆を推定したいとき

❌ 人が少なく、一人ひとりはっきり見える写真には向いていません
　→ その場合は「🤖 YOLO自動検出」タブを使ってください

📊 **複数の異なるアルゴリズムで同時に推定し、推定範囲を表示します**
""")

    import os
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.font_manager as fm

    # 日本語フォントをファイルパスから直接ロード（キャッシュ不要で確実）
    @st.cache_resource
    def _get_jp_font_prop():
        import glob
        search_paths = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf",
            "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf",
            "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
        ]
        search_paths += glob.glob("/usr/share/fonts/**/*CJK*.tt[cf]", recursive=True)
        search_paths += glob.glob("/usr/share/fonts/**/*noto*jp*.tt[cf]", recursive=True)
        search_paths += glob.glob("/usr/share/fonts/**/*Noto*CJK*.tt[cf]", recursive=True)
        for path in search_paths:
            if os.path.exists(path):
                try:
                    return fm.FontProperties(fname=path)
                except Exception:
                    continue
        return None

    _JP_FONT = _get_jp_font_prop()

    def _title(ax, text, **kwargs):
        if _JP_FONT:
            kwargs["fontproperties"] = _JP_FONT
        ax.set_title(text, **kwargs)

    try:
        import torch
        import torch.nn as nn
        from torchvision import models as tv_models, transforms
        torch_available = True
    except ImportError:
        torch_available = False

    if st.session_state.original_bgr is None:
        st.warning("👆 まず画像をアップロードしてください")
        st.stop()

    orig     = st.session_state.original_bgr
    orig_pil = bgr_to_pil(orig)
    W, H     = orig_pil.size

    # ── ヒートマップ合成 ──
    def make_overlay(pil_img, density, a=0.55):
        w_, h_ = pil_img.size
        d_norm = density / (density.max() + 1e-8)
        hm = (cm.jet(d_norm)[:, :, :3] * 255).astype(np.uint8)
        hm_pil = Image.fromarray(hm).resize((w_, h_), Image.BILINEAR)
        return Image.blend(pil_img.convert("RGB"), hm_pil, alpha=float(a))

    def fig_to_pil(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        buf.seek(0)
        plt.close(fig)
        return Image.open(buf).copy()

    # ══════════════════════════════════════
    # 推定手法1: エッジ密度法
    # Canny エッジの密度を人の密集度の代理指標として使う
    # 根拠: 人体・衣服の輪郭が密集するほどエッジが増える
    # ══════════════════════════════════════
    @st.cache_data(show_spinner=False)
    def method_edge(img_bytes: bytes, cell_px: int = 32):
        nparr = np.frombuffer(img_bytes, np.uint8)
        bgr   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h_, w_ = gray.shape

        edges = cv2.Canny(gray, 25, 75).astype(np.float32) / 255.0
        rows, cols = h_ // cell_px, w_ // cell_px
        sm = np.zeros((rows, cols), np.float32)
        for r in range(rows):
            for c in range(cols):
                sm[r, c] = edges[r*cell_px:(r+1)*cell_px,
                                  c*cell_px:(c+1)*cell_px].mean()
        sm = cv2.GaussianBlur(sm, (5, 5), 0)
        density = np.maximum(cv2.resize(sm, (w_, h_), interpolation=cv2.INTER_CUBIC), 0)

        # 高エッジ領域を群衆エリアとしてピクセル数を集計
        thr = density.mean() + density.std() * 0.5
        crowd_px = int((density > thr).sum())
        # 経験的換算: 空撮で人1人あたり約 350〜500 px²（解像度依存）
        px_per_person = max(200, (W * H) // 3000)
        count = max(1, crowd_px // px_per_person)
        return density, count

    # ══════════════════════════════════════
    # 推定手法2: 局所分散法
    # 局所領域の輝度分散が高い = 人体・服の混在 = 密集
    # 根拠: 均一な背景（道路・建物）は分散低、群衆は高
    # ══════════════════════════════════════
    @st.cache_data(show_spinner=False)
    def method_variance(img_bytes: bytes, cell_px: int = 32):
        nparr = np.frombuffer(img_bytes, np.uint8)
        bgr   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h_, w_ = bgr.shape[:2]

        rows, cols = h_ // cell_px, w_ // cell_px
        sm = np.zeros((rows, cols), np.float32)
        for r in range(rows):
            for c in range(cols):
                patch = gray[r*cell_px:(r+1)*cell_px, c*cell_px:(c+1)*cell_px]
                sm[r, c] = float(patch.std())
        sm = cv2.GaussianBlur(sm, (5, 5), 0)
        density = np.maximum(cv2.resize(sm, (w_, h_), interpolation=cv2.INTER_CUBIC), 0)

        thr = density.mean() + density.std() * 0.4
        crowd_px = int((density > thr).sum())
        px_per_person = max(200, (W * H) // 3000)
        count = max(1, crowd_px // px_per_person)
        return density, count

    # ══════════════════════════════════════
    # 推定手法3: HSV彩度分散法
    # 彩度の局所分散が高い = 多彩な服色 = 人の密集
    # 根拠: 人は服の色が多様。道路・建物は単色に近い
    # ══════════════════════════════════════
    @st.cache_data(show_spinner=False)
    def method_hsv(img_bytes: bytes, cell_px: int = 32):
        nparr = np.frombuffer(img_bytes, np.uint8)
        bgr   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        hsv   = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        sat   = hsv[:, :, 1]  # 彩度チャンネル
        h_, w_ = bgr.shape[:2]

        rows, cols = h_ // cell_px, w_ // cell_px
        sm = np.zeros((rows, cols), np.float32)
        for r in range(rows):
            for c in range(cols):
                patch = sat[r*cell_px:(r+1)*cell_px, c*cell_px:(c+1)*cell_px]
                # 彩度の平均 + 分散を組み合わせる
                sm[r, c] = float(patch.mean() * 0.4 + patch.std() * 0.6)
        sm = cv2.GaussianBlur(sm, (5, 5), 0)
        density = np.maximum(cv2.resize(sm, (w_, h_), interpolation=cv2.INTER_CUBIC), 0)

        thr = density.mean() + density.std() * 0.4
        crowd_px = int((density > thr).sum())
        px_per_person = max(200, (W * H) // 3000)
        count = max(1, crowd_px // px_per_person)
        return density, count

    # ══════════════════════════════════════
    # 推定手法4: CSRNet（学習済みモデル）
    # ══════════════════════════════════════
    if torch_available:
        @st.cache_resource(show_spinner="CSRNet初期化中（初回のみ）...")
        def get_csrnet():
            WEIGHTS = "csrnet_partA.pth"
            def is_valid(p): return os.path.exists(p) and os.path.getsize(p) > 1_000_000
            def try_dl():
                try:
                    import gdown
                    for gid in ["1nnIHPaV9RGqK8JHL645zmRvkNrahD9ru",
                                "190bB3q3R7o-PEe7MbxHiNqf7M_bsEBEt"]:
                        try:
                            gdown.download(f"https://drive.google.com/uc?id={gid}",
                                           WEIGHTS, quiet=True)
                            if is_valid(WEIGHTS): return True
                        except Exception: continue
                except ImportError: pass
                try:
                    from huggingface_hub import hf_hub_download
                    import shutil
                    shutil.copy(hf_hub_download(
                        repo_id="nickmuchi/csrnet-crowd-counting",
                        filename="csrnet_partA.pth"), WEIGHTS)
                    if is_valid(WEIGHTS): return True
                except Exception: pass
                return False

            def make_layers(cfg, ic=3, dil=False):
                d, layers = (2 if dil else 1), []
                for v in cfg:
                    if v == "M": layers += [nn.MaxPool2d(2, 2)]
                    else:
                        layers += [nn.Conv2d(ic, v, 3, padding=d, dilation=d), nn.ReLU(True)]
                        ic = v
                return nn.Sequential(*layers)

            class CSRNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.DEFAULT)
                    self.frontend = nn.Sequential(*list(vgg.features.children())[:23])
                    self.backend  = make_layers([512,512,512,256,128,64], ic=512, dil=True)
                    self.output   = nn.Conv2d(64, 1, 1)
                    for m in list(self.backend.modules()) + [self.output]:
                        if isinstance(m, nn.Conv2d):
                            nn.init.normal_(m.weight, std=0.01)
                            if m.bias is not None: nn.init.constant_(m.bias, 0)
                def forward(self, x):
                    return self.output(self.backend(self.frontend(x)))

            if not is_valid(WEIGHTS): try_dl()
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = CSRNet().to(dev).eval()
            loaded = False
            if is_valid(WEIGHTS):
                try:
                    ckpt = torch.load(WEIGHTS, map_location=dev)
                    if isinstance(ckpt, dict):
                        ckpt = ckpt.get("state_dict", ckpt.get("model", ckpt))
                    model.load_state_dict(ckpt, strict=False)
                    loaded = True
                except Exception: pass
            return model, dev, loaded

        csrnet_model, csr_device, weights_loaded = get_csrnet()
    else:
        weights_loaded = False

    # ── ステータスバナー ──
    if weights_loaded:
        st.success("✅ CSRNet 学習済み重みを読み込みました（手法4が有効）")
    else:
        st.info("ℹ️ CSRNet重みなし → 手法1〜3のみで推定します（手法4を追加するには `csrnet_partA.pth` を配置）")

    # ── UI ──
    col1, col2 = st.columns([1, 2])
    with col1:
        cell_px  = st.slider("分析セルサイズ (px)", 16, 64, 28, 4, key="csrnet_cell",
                              help="小さいほど細かく分析。大きいほど高速")
        alpha    = st.slider("ヒートマップ透明度", 0.2, 0.9, 0.55, 0.05, key="csrnet_alpha")
        if weights_loaded:
            use_tile  = st.checkbox("CSRNet タイル分割", value=True, key="csrnet_tile")
            tile_size = st.slider("タイルサイズ", 256, 1024, 512, 128, key="csrnet_tilesize",
                                   disabled=not use_tile)
        run = st.button("🔍 全手法で推定を実行", type="primary", use_container_width=True, key="csrnet_run")

    # ── CSRNet推論関数 ──
    if torch_available and weights_loaded:
        TF = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        def csr_infer(pil_img):
            w_, h_ = pil_img.size
            t = TF(pil_img).unsqueeze(0).to(csr_device)
            with torch.no_grad():
                d = csrnet_model(t)[0, 0].cpu().numpy()
            return np.maximum(cv2.resize(d, (w_, h_), interpolation=cv2.INTER_CUBIC), 0)

        def csr_tile(pil_img, ts=512):
            w_, h_ = pil_img.size
            if w_ <= ts and h_ <= ts: return csr_infer(pil_img)
            img_np = np.array(pil_img)
            ds, ws = np.zeros((h_, w_), np.float32), np.zeros((h_, w_), np.float32)
            stride = ts - 64
            ys = sorted(set(list(range(0, max(1,h_-ts), stride)) + [max(0,h_-ts)]))
            xs = sorted(set(list(range(0, max(1,w_-ts), stride)) + [max(0,w_-ts)]))
            prog = st.progress(0, text="CSRNetタイル推論中...")
            total = len(ys) * len(xs)
            for idx, (y, x) in enumerate([(y,x) for y in ys for x in xs]):
                y2, x2 = min(y+ts,h_), min(x+ts,w_)
                d = csr_infer(Image.fromarray(img_np[y:y2,x:x2]))
                th, tw = y2-y, x2-x
                ds[y:y2,x:x2] += cv2.resize(d,(tw,th)) if d.shape!=(th,tw) else d
                ws[y:y2,x:x2] += 1.0
                prog.progress((idx+1)/total, text=f"CSRNet: {idx+1}/{total}")
            prog.empty()
            return ds / np.maximum(ws, 1e-8)

    # ── 実行 ──
    if run:
        buf_bytes = io.BytesIO()
        orig_pil.save(buf_bytes, format="PNG")
        img_bytes = buf_bytes.getvalue()

        results = {}  # name -> (density, count, description)

        with st.spinner("手法1: エッジ密度法..."):
            d1, c1 = method_edge(img_bytes, cell_px=int(cell_px))
            results["edge"] = (d1, c1, "①エッジ密度法",
                "Cannyエッジの密度から推定。人体輪郭が密集するほど高くなる。")

        with st.spinner("手法2: 局所分散法..."):
            d2, c2 = method_variance(img_bytes, cell_px=int(cell_px))
            results["variance"] = (d2, c2, "②局所分散法",
                "輝度の局所分散から推定。人・服が混在するほど分散が高くなる。")

        with st.spinner("手法3: HSV彩度法..."):
            d3, c3 = method_hsv(img_bytes, cell_px=int(cell_px))
            results["hsv"] = (d3, c3, "③HSV彩度法",
                "服の色彩分散から推定。多様な服色が混在するほどスコアが高くなる。")

        if torch_available and weights_loaded:
            with st.spinner("手法4: CSRNet（学習済みモデル）..."):
                d4 = csr_tile(orig_pil, ts=int(tile_size)) if use_tile else csr_infer(orig_pil)
                c4 = int(d4.sum())
                results["csrnet"] = (d4, c4, "④CSRNet",
                    "ShanghaiTechデータで学習済みの深層学習モデル。高密度群衆に特化。")

        counts = [v[1] for v in results.values()]
        count_min  = min(counts)
        count_max  = max(counts)
        count_mean = int(sum(counts) / len(counts))
        count_std  = int((sum((c - count_mean)**2 for c in counts) / len(counts))**0.5)

        # ── メイン結果 ──
        with col2:
            # 手法ごとのヒートマップを1枚に並べる
            n = len(results)
            fig, axes = plt.subplots(1, n+1, figsize=(5*(n+1), 5))
            fig.patch.set_facecolor("#111")
            axes[0].imshow(np.array(orig_pil))
            _title(axes[0], "元画像", color="white", fontsize=11)
            axes[0].axis("off")
            for ax, (name, (d, c, label, _)) in zip(axes[1:], results.items()):
                overlay = make_overlay(orig_pil, d, a=float(alpha))
                ax.imshow(np.array(overlay))
                _title(ax, f"{label}\n推定: {c:,}人", color="white", fontsize=10)
                ax.axis("off")
            plt.tight_layout()
            _compare_fig_pil = fig_to_pil(fig)
            st.image(_compare_fig_pil, use_container_width=True)

        # ── 推定サマリー ──
        st.divider()
        st.markdown("### 📊 推定サマリー")

        # 大きく範囲を表示
        st.markdown(
            f"""
            <div style="background:#1e3a5f;border-radius:12px;padding:20px 30px;margin:8px 0;">
              <div style="font-size:14px;color:#aac4e8;margin-bottom:6px;">複数手法による推定範囲</div>
              <div style="font-size:42px;font-weight:bold;color:#ffffff;">
                {count_min:,} 〜 {count_max:,} 人
              </div>
              <div style="font-size:18px;color:#7ec8e3;margin-top:4px;">
                平均: {count_mean:,} 人　±{count_std:,}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 各手法の個別結果
        cols_r = st.columns(len(results))
        for col_r, (name, (_, count, label, desc)) in zip(cols_r, results.items()):
            with col_r:
                st.metric(label, f"{count:,} 人")
                st.caption(desc)

        st.divider()
        st.markdown("""
        **📌 推定値の読み方**
        - 複数手法が近い値を示すほど信頼性が高い
        - 手法によって前提が違うので、**範囲の中央付近**が最も現実的な推定
        - CSRNet（手法4）は学習済みモデルで最も根拠が強いが、撮影条件に依存
        - 最終的な数字は手動クリックモードで部分確認して補正するのが最も確実
        """)

        # ダウンロード
        ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        best_density = list(results.values())[0][0]
        best_overlay = make_overlay(orig_pil, best_density, a=float(alpha))
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "💾 ヒートマップ画像をダウンロード",
                data=pil_to_bytes(best_overlay),
                file_name=f"crowd_heatmap_{ts_str}.png",
                mime="image/png",
                use_container_width=True,
            )
        with dl2:
            st.download_button(
                "📊 比較図をダウンロード",
                data=pil_to_bytes(_compare_fig_pil),
                file_name=f"crowd_compare_{ts_str}.png",
                mime="image/png",
                use_container_width=True,
            )

    else:
        with col2:
            st.image(orig_pil, use_container_width=True, caption="アップロードされた画像")

# ─────────────────────────────────────────
# フッター
# ─────────────────────────────────────────
st.divider()
st.caption("🧑‍🤝‍🧑 Crowd Counter | CSRNet: Li et al., CVPR 2018 | YOLO: Ultralytics")
