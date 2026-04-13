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

st.title("🧑‍🤝‍🧑 空撮・集合写真 人数カウンター")
st.caption("手動クリック / YOLO自動検出 / CSRNet密度推定 の3モードに対応")

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
# サイドバー: モード選択 & 共通設定
# ─────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 設定")
    mode = st.radio(
        "推定モード",
        ["✋ 手動クリック", "🤖 YOLO自動検出", "🌡️ CSRNet密度推定"],
        help=(
            "手動: 確実・低密度向け\n"
            "YOLO: 自動・中密度向け\n"
            "CSRNet: 高密度・空撮向け"
        ),
    )
    st.divider()
    st.markdown("""
    **使い方**
    1. 画像をアップロード
    2. モードを選択
    3. 結果を確認・保存
    
    **モード選択の目安**
    | 密度 | おすすめ |
    |------|---------|
    | 低（識別可能） | YOLO |
    | 中 | YOLO + SAHI |
    | 高（空撮デモ） | CSRNet |
    | 正確に確認 | 手動 |
    """)


# ─────────────────────────────────────────
# 画像アップロード
# ─────────────────────────────────────────
uploaded = st.file_uploader(
    "📂 画像をアップロード（JPG / PNG / WEBP）",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded is not None:
    # 新しい画像がアップロードされたらリセット
    if uploaded.name != st.session_state.last_uploaded_name:
        img_pil = Image.open(uploaded).convert("RGB")
        st.session_state.original_bgr = pil_to_bgr(img_pil)
        st.session_state.points = []
        st.session_state.auto_ids = set()
        st.session_state.history = []
        st.session_state.last_uploaded_name = uploaded.name
        st.success(f"✅ 読み込み完了: {uploaded.name}  ({img_pil.width}×{img_pil.height}px)")

# ─────────────────────────────────────────
# モード A: 手動クリック
# ─────────────────────────────────────────
if mode == "✋ 手動クリック":
    st.subheader("✋ 手動クリックモード")
    st.info("画像の上でクリックした座標を入力して人をマークします。"
            "Streamlit上での直接クリックマーキングは制限があるため、"
            "座標を直接指定する方式を採用しています。")

    if st.session_state.original_bgr is not None:
        orig = st.session_state.original_bgr
        h, w = orig.shape[:2]

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("**📌 座標を入力してマーク**")
            c1, c2 = st.columns(2)
            with c1:
                px = st.number_input("X座標", min_value=0, max_value=w-1, value=w//2, step=1)
            with c2:
                py = st.number_input("Y座標", min_value=0, max_value=h-1, value=h//2, step=1)

            bc1, bc2, bc3 = st.columns(3)
            with bc1:
                if st.button("➕ 追加", use_container_width=True, type="primary"):
                    st.session_state.history.append(st.session_state.points.copy())
                    if len(st.session_state.history) > 50:
                        st.session_state.history.pop(0)
                    st.session_state.points.append((int(px), int(py)))
                    st.rerun()
            with bc2:
                if st.button("↩ 戻す", use_container_width=True):
                    if st.session_state.history:
                        st.session_state.points = st.session_state.history.pop()
                    st.rerun()
            with bc3:
                if st.button("🔄 リセット", use_container_width=True):
                    st.session_state.points = []
                    st.session_state.auto_ids = set()
                    st.session_state.history = []
                    st.rerun()

            # 直近の点を削除
            if st.session_state.points:
                st.markdown("**🗑 近くの点を削除**")
                dc1, dc2 = st.columns(2)
                with dc1:
                    del_x = st.number_input("削除X", min_value=0, max_value=w-1,
                                             value=st.session_state.points[-1][0], step=1, key="del_x")
                with dc2:
                    del_y = st.number_input("削除Y", min_value=0, max_value=h-1,
                                             value=st.session_state.points[-1][1], step=1, key="del_y")
                if st.button("最も近い点を削除", use_container_width=True):
                    pts = st.session_state.points
                    dists = [((p[0]-del_x)**2 + (p[1]-del_y)**2, i) for i, p in enumerate(pts)]
                    _, ni = min(dists)
                    st.session_state.history.append(pts.copy())
                    st.session_state.points.pop(ni)
                    st.rerun()

            st.metric("現在の人数", len(st.session_state.points))

        with col2:
            result = draw_points(orig, st.session_state.points, st.session_state.auto_ids)
            st.image(bgr_to_pil(result), use_container_width=True, caption="マーク済み画像")

        # 保存
        if st.session_state.points:
            st.divider()
            sc1, sc2 = st.columns(2)
            result_pil = bgr_to_pil(draw_points(orig, st.session_state.points))
            with sc1:
                st.download_button(
                    "💾 マーク済み画像をダウンロード",
                    data=pil_to_bytes(result_pil),
                    file_name=f"crowd_manual_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png",
                    use_container_width=True,
                )
            with sc2:
                json_data = json.dumps({
                    "count": len(st.session_state.points),
                    "points": [{"id": i+1, "x": x, "y": y}
                               for i, (x, y) in enumerate(st.session_state.points)]
                }, indent=2)
                st.download_button(
                    "📋 座標データ (JSON) をダウンロード",
                    data=json_data,
                    file_name=f"crowd_points_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True,
                )
    else:
        st.warning("👆 まず画像をアップロードしてください")


# ─────────────────────────────────────────
# モード B: YOLO自動検出
# ─────────────────────────────────────────
elif mode == "🤖 YOLO自動検出":
    st.subheader("🤖 YOLO自動検出モード")
    st.caption("🔵 青丸 = YOLO自動検出　🟢 緑丸 = 手動追加")

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
        conf = st.slider("信頼度しきい値", 0.1, 0.8, 0.3, 0.05,
                         help="低いほど検出数が増えるが誤検出も増える")
        use_sahi = st.checkbox("🔍 高解像度モード（SAHI）",
                               help="空撮など大きい画像向け。処理が重くなる")
        model_size = st.selectbox("モデルサイズ",
                                   ["yolov8n", "yolov8s", "yolov8m"],
                                   index=0,
                                   help="n=軽量 / s=バランス / m=高精度")

        if st.button("🤖 自動検出を実行", type="primary", use_container_width=True):
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
        if st.button("➕ 手動追加"):
            st.session_state.history.append(
                (st.session_state.points.copy(), st.session_state.auto_ids.copy()))
            st.session_state.points.append((int(add_x), int(add_y)))
            st.rerun()

        bc1, bc2 = st.columns(2)
        with bc1:
            if st.button("↩ 戻す", use_container_width=True):
                if st.session_state.history:
                    h_pts, h_ids = st.session_state.history.pop()
                    st.session_state.points = h_pts
                    st.session_state.auto_ids = h_ids
                st.rerun()
        with bc2:
            if st.button("🔄 リセット", use_container_width=True):
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
            )


# ─────────────────────────────────────────
# モード C: CSRNet密度推定
# ─────────────────────────────────────────
elif mode == "🌡️ CSRNet密度推定":
    st.subheader("🌡️ CSRNet 群衆密度推定")
    st.caption("高密度・空撮デモ写真向け | 赤いほど人が密集しています")

    # PyTorchのインポートを試みる
    try:
        import torch
        import torch.nn as nn
        from torchvision import models as tv_models, transforms
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        torch_available = True
    except ImportError:
        torch_available = False

    if not torch_available:
        st.error("PyTorchがインストールされていません")
        st.code("pip install torch torchvision")
        st.stop()

    if st.session_state.original_bgr is None:
        st.warning("👆 まず画像をアップロードしてください")
        st.stop()

    orig = st.session_state.original_bgr
    orig_pil = bgr_to_pil(orig)
    W, H = orig_pil.size

    # ── CSRNet定義（キャッシュ）──
    @st.cache_resource(show_spinner="CSRNetを初期化中...")
    def get_csrnet():
        import os

        def make_layers(cfg, in_channels=3, dilation=False):
            d = 2 if dilation else 1
            layers = []
            for v in cfg:
                if v == 'M':
                    layers += [nn.MaxPool2d(2, 2)]
                else:
                    layers += [nn.Conv2d(in_channels, v, 3, padding=d, dilation=d),
                               nn.ReLU(inplace=True)]
                    in_channels = v
            return nn.Sequential(*layers)

        class CSRNet(nn.Module):
            def __init__(self):
                super().__init__()
                vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.DEFAULT)
                self.frontend = nn.Sequential(*list(vgg.features.children())[:23])
                self.backend  = make_layers([512,512,512,256,128,64],
                                             in_channels=512, dilation=True)
                self.output   = nn.Conv2d(64, 1, 1)
                for m in list(self.backend.modules()) + [self.output]:
                    if isinstance(m, nn.Conv2d):
                        nn.init.normal_(m.weight, std=0.01)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)

            def forward(self, x):
                return self.output(self.backend(self.frontend(x)))

        # ── 重みファイルの自動ダウンロード ──
        WEIGHTS = "csrnet_partA.pth"

        def is_valid_weights(path):
            return os.path.exists(path) and os.path.getsize(path) > 1_000_000

        def try_download():
            # 試行1: gdown (Google Drive)
            # CommissarMa/CSRNet-pytorch で配布されている PartA 学習済み重み
            try:
                import gdown
                gdrive_ids = [
                    "1nnIHPaV9RGqK8JHL645zmRvkNrahD9ru",
                    "190bB3q3R7o-PEe7MbxHiNqf7M_bsEBEt",
                ]
                for gid in gdrive_ids:
                    try:
                        gdown.download(
                            f"https://drive.google.com/uc?id={gid}",
                            WEIGHTS, quiet=True
                        )
                        if is_valid_weights(WEIGHTS):
                            return True
                    except Exception:
                        continue
            except ImportError:
                pass

            # 試行2: HuggingFace Hub
            try:
                from huggingface_hub import hf_hub_download
                import shutil
                path = hf_hub_download(
                    repo_id="nickmuchi/csrnet-crowd-counting",
                    filename="csrnet_partA.pth",
                )
                shutil.copy(path, WEIGHTS)
                if is_valid_weights(WEIGHTS):
                    return True
            except Exception:
                pass

            return False

        if not is_valid_weights(WEIGHTS):
            try_download()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CSRNet().to(device).eval()

        weights_loaded = False
        if is_valid_weights(WEIGHTS):
            try:
                ckpt = torch.load(WEIGHTS, map_location=device)
                if isinstance(ckpt, dict):
                    ckpt = ckpt.get("state_dict", ckpt.get("model", ckpt))
                model.load_state_dict(ckpt, strict=False)
                weights_loaded = True
            except Exception:
                pass

        return model, device, weights_loaded

    import os
    model_csrnet, device, weights_loaded = get_csrnet()

    if not weights_loaded:
        st.warning(
            "⚠️ 学習済み重みの自動ダウンロードに失敗しました。"
            "VGG16 frontendのみのフォールバックモードで動作します（精度は低め）。\n\n"
            "**手動ダウンロード手順:**\n"
            "1. https://github.com/CommissarMa/CSRNet-pytorch を開く\n"
            "2. READMEのGoogle DriveリンクからPartAの重みをダウンロード\n"
            "3. `csrnet_partA.pth` という名前で `app.py` と同じフォルダに保存\n"
            "4. アプリを再起動"
        )
    else:
        st.success("✅ 学習済み重みを読み込みました（高精度モード）")

    # 設定
    col1, col2 = st.columns([1, 2])
    with col1:
        use_tile = st.checkbox("🔲 タイル分割推論", value=True,
                               help="高解像度・空撮画像向け。処理が重くなります")
        tile_size = st.slider("タイルサイズ (px)", 256, 1024, 512, 128,
                              disabled=not use_tile)
        alpha = st.slider("ヒートマップ透明度", 0.2, 0.9, 0.55, 0.05)

        run = st.button("🔍 密度推定を実行", type="primary", use_container_width=True)

    # ── 推論関数 ──
    TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    def infer_once(pil_img):
        w_, h_ = pil_img.size
        t = TRANSFORM(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            d = model_csrnet(t)[0, 0].cpu().numpy()
        d = cv2.resize(d, (w_, h_), interpolation=cv2.INTER_CUBIC)
        return np.maximum(d, 0)

    def tile_infer(pil_img, ts=512, overlap=64):
        w_, h_ = pil_img.size
        if w_ <= ts and h_ <= ts:
            return infer_once(pil_img)
        img_np = np.array(pil_img)
        d_sum = np.zeros((h_, w_), dtype=np.float32)
        w_sum = np.zeros((h_, w_), dtype=np.float32)
        stride = ts - overlap
        ys = sorted(set(list(range(0, max(1,h_-ts), stride)) + [max(0,h_-ts)]))
        xs = sorted(set(list(range(0, max(1,w_-ts), stride)) + [max(0,w_-ts)]))
        total = len(ys) * len(xs)
        prog = st.progress(0, text="タイル推論中...")
        for idx, (y, x) in enumerate([(y,x) for y in ys for x in xs]):
            y2, x2 = min(y+ts, h_), min(x+ts, w_)
            tile = Image.fromarray(img_np[y:y2, x:x2])
            d = infer_once(tile)
            th, tw = y2-y, x2-x
            d_sum[y:y2, x:x2] += cv2.resize(d, (tw, th)) if d.shape != (th,tw) else d
            w_sum[y:y2, x:x2] += 1.0
            prog.progress((idx+1)/total, text=f"タイル推論中... {idx+1}/{total}")
        prog.empty()
        return d_sum / np.maximum(w_sum, 1e-8)

    def make_overlay(pil_img, density, a=0.55):
        w_, h_ = pil_img.size
        d_norm = density / (density.max() + 1e-8)
        hm = (cm.jet(d_norm)[:,:,:3] * 255).astype(np.uint8)
        hm_pil = Image.fromarray(hm).resize((w_,h_), Image.BILINEAR)
        return Image.blend(pil_img.convert("RGB"), hm_pil, alpha=float(a))

    if run:
        with st.spinner("推論中..."):
            if use_tile:
                density = tile_infer(orig_pil, ts=int(tile_size))
            else:
                max_side = 1024
                if max(W, H) > max_side:
                    scale = max_side / max(W, H)
                    img_small = orig_pil.resize((int(W*scale), int(H*scale)), Image.LANCZOS)
                else:
                    img_small = orig_pil
                density = infer_once(img_small)
                density = cv2.resize(density, (W, H))

            count = int(density.sum())
            high_ratio = float((density / (density.max()+1e-8) > 0.5).mean() * 100)
            overlay = make_overlay(orig_pil, density, a=alpha)

            # 詳細比較図（matplotlibで3列）
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.patch.set_facecolor('#1a1a1a')
            axes[0].imshow(np.array(orig_pil))
            axes[0].set_title("元画像", color='white', fontsize=12)
            axes[1].imshow(np.array(overlay))
            axes[1].set_title("ヒートマップ（赤=密集）", color='white', fontsize=12)
            im = axes[2].imshow(density, cmap='hot')
            axes[2].set_title("密度マップ（生データ）", color='white', fontsize=12)
            cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
            cbar.set_label('密度値', color='white')
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
            for ax in axes: ax.axis('off')
            plt.tight_layout()
            fig_buf = io.BytesIO()
            fig.savefig(fig_buf, format='png', bbox_inches='tight')
            fig_buf.seek(0)
            plt.close(fig)

        with col2:
            st.image(overlay, use_container_width=True,
                     caption="ヒートマップ（赤=密集 / 青=少ない）")

        # 結果表示
        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("🧑‍🤝‍🧑 推定人数", f"{count:,} 人")
        m2.metric("🔥 高密度エリア", f"{high_ratio:.1f}%")
        m3.metric("📐 画像サイズ", f"{W}×{H}")

        st.caption("⚠️ 推定値です。撮影角度・遮蔽・モデルの限界により誤差が生じます。")

        st.image(Image.open(fig_buf), use_container_width=True,
                 caption="詳細分析（元画像 / ヒートマップ / 密度マップ）")

        # ダウンロード
        dc1, dc2 = st.columns(2)
        with dc1:
            st.download_button(
                "💾 ヒートマップ画像をダウンロード",
                data=pil_to_bytes(overlay),
                file_name=f"csrnet_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png",
                use_container_width=True,
            )
        with dc2:
            st.download_button(
                "📊 詳細分析図をダウンロード",
                data=fig_buf.getvalue(),
                file_name=f"csrnet_detail_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png",
                use_container_width=True,
            )
    else:
        with col2:
            if st.session_state.original_bgr is not None:
                st.image(orig_pil, use_container_width=True, caption="アップロードされた画像")

# ─────────────────────────────────────────
# フッター
# ─────────────────────────────────────────
st.divider()
st.caption("🧑‍🤝‍🧑 Crowd Counter | CSRNet: Li et al., CVPR 2018 | YOLO: Ultralytics")
