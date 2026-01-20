import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.signal import butter, filtfilt, iirnotch
import re, os
from scipy.signal import welch, find_peaks


# ========== 字型與顯示（避免方塊字與 1e-5 科學記號） ==========
matplotlib.rcParams['font.sans-serif'] = [
    'Microsoft JhengHei', 'SimHei', 'Noto Sans CJK TC', 'Arial Unicode MS', 'sans-serif'
]
matplotlib.rcParams['axes.unicode_minus'] = False  # 用 ASCII 負號

# ========== 使用者參數 ==========
csv_path = r"D:\1\OneDrive\桌面\123.csv"   # 你的 CSV 檔
save_dir = os.path.dirname(csv_path)                            # 存圖資料夾
show_full_duration =    True                            # False=只顯示 ROI；True=顯示整段
# 濾波（處理後波形）
hp_fc = 0.5          # 高通截止(Hz) 去漂移（把低於 0.5 Hz 的成分濾掉）
lp_fc = 8.0          # 低通截止(Hz) 抑制高頻雜訊（把高於 8 Hz 的雜訊壓掉）
use_notch = True     # 電源陷波，醫療儀器常會受到電源供應的干擾，在台灣電力頻率是 60 Hz。
notch_f0 = 60.0      # 60 (這裡選 60.0 表示要壓制 60 Hz 的電源干擾。依電力頻率）

# ========== 小工具 ==========
def clean_label(s: str) -> str:
    """把欄名淨化，僅保留中英數與空白/底線，避免圖上出現奇怪符號"""
    s = re.sub(r'[^0-9A-Za-z\u4e00-\u9fa5_ ]+', '', str(s))
    return s.strip() or "通道"

def estimate_fs(time):
    """以時間中位數差估取樣率，對不等距更穩"""
    if len(time) > 1:
        dt = float(np.median(np.diff(time)))
        if not np.isfinite(dt) or dt <= 0:
            dt = 1.0
    else:
        dt = 1.0
    return 1.0 / dt, dt

def bandpass(x, fs, hp=None, lp=None, order=4):
    """帶通濾波：去漂移與高頻雜訊"""
    y = x.copy()
    if hp and hp > 0 and fs > 2*hp:
        b, a = butter(order, hp/(fs/2), btype='high')
        y = filtfilt(b, a, y)
    if lp and lp > 0 and fs > 2*lp:
        b, a = butter(order, lp/(fs/2), btype='low')
        y = filtfilt(b, a, y)
    return y

def notch(x, fs, f0=60.0, Q=30):
    """電源陷波：抑制 50/60Hz 干擾"""
    if fs <= 2*f0:
        return x
    b, a = iirnotch(w0=f0/(fs/2), Q=Q)
    return filtfilt(b, a, x)

def find_active_segment(x, fs, win_sec=2.0, thresh=3.0, min_len_sec=5.0):
    """
    自動找「最長且穩定」的有效區段 ROI：
    視窗標準差 > 中位數*thresh 視為有訊號；回傳最佳 [a,b) 索引
    """
    n = len(x)
    win = int(max(1, round(win_sec*fs)))
    pad = win // 2
    x0 = x - np.nanmean(x)
    xp = np.pad(x0, (pad, pad), mode='edge')
    mov_mean = np.convolve(xp, np.ones(win)/win, mode='valid')
    mov_sq_mean = np.convolve(xp**2, np.ones(win)/win, mode='valid')
    mov_std = np.sqrt(np.maximum(mov_sq_mean - mov_mean**2, 1e-12))

    med = np.median(mov_std)
    active = mov_std > (thresh * med)

    min_len = int(min_len_sec * fs)
    best = (0, 0); start = None
    for i, v in enumerate(active):
        if v and start is None:
            start = i
        if (not v or i == len(active)-1) and start is not None:
            end = i if not v else i+1
            if end - start >= min_len and (end - start) > (best[1]-best[0]):
                best = (start, end)
            start = None

    if best[1] - best[0] <= 0:
        return 0, n
    a = max(0, best[0] - int(0.5*fs))   # 兩端各多留 0.5 秒，避免切太緊
    b = min(n, best[1] + int(0.5*fs))
    return a, b


def make_psd_signals(x_roi_raw, fs, use_notch=True, notch_f0=60.0):
    # 低頻圖：保留 0.3–30 Hz（看基頻與倍頻 → 低頻凸）
    x_for_low  = bandpass(x_roi_raw, fs, hp=0.3, lp=30.0, order=4)
    if use_notch:
        x_for_low = notch(x_for_low, fs, f0=notch_f0, Q=30)

    # 高頻圖：去掉心跳主頻，用 12 Hz 高通，不做低通（必要時再 notch）
    x_for_high = bandpass(x_roi_raw, fs, hp=12.0, lp=None, order=4)
    if use_notch:
        x_for_high = notch(x_for_high, fs, f0=notch_f0, Q=30)
    return x_for_low, x_for_high


def welch_psd(x, fs, seg_sec=6.0):
    """Welch PSD；Hamming、50% overlap，較平滑、接近範例"""
    nperseg = max(256, int(seg_sec * fs))
    noverlap = nperseg // 2
    f, Pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap,
                   window='hamming', detrend='constant', scaling='density')
    return f, Pxx

def smooth_psd(Pxx, win=5):
    """簡單滑動平均，讓曲線更像範例的平滑度"""
    if win <= 1: return Pxx
    k = np.ones(win) / win
    return np.convolve(Pxx, k, mode='same')

def style_axes(ax):
    """統一外觀：細網格、去右上框線、科學記號、字體大小"""
    ax.grid(True, linewidth=0.6, alpha=0.8)
    ax.set_facecolor("#fcfcfc")
    ax.tick_params(labelsize=9)
    for s in ['top','right']:
        ax.spines[s].set_visible(False)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1E'))  # 顯示 1.2E-7

# ========== 讀檔與時間軸處理 ==========
df = pd.read_csv(csv_path)

# 第一欄時間；轉數值並去除非數字
time_raw = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
valid_t = np.isfinite(time_raw)
time = time_raw[valid_t]

# 依時間排序（避免亂序）
sort_idx = np.argsort(time) if len(time) > 1 else None
if sort_idx is not None:
    time = time[sort_idx]

# 估取樣率
fs, dt = estimate_fs(time)
print(f"取樣率 fs ≈ {fs:.3f} Hz  dt ≈ {dt:.6f} s")




# --- 取樣率與時間單位自檢 ---
if fs < 5 and np.nanmax(time) > 100:  # 很可能時間是毫秒
    time = time / 1000.0
    fs, dt = estimate_fs(time)
    print("偵測到時間可能是毫秒，已自動換算為秒。")

print(f"取樣率 fs ≈ {fs:.3f} Hz (Nyquist ≈ {fs/2:.1f} Hz)")
if fs/2 < 50:
    print(f"注意：Nyquist 僅 {fs/2:.1f} Hz，高頻圖最多只能畫到 {fs/2:.1f} Hz，無法完整顯示 50 Hz。")





# 所有通道（時間欄之後）
channels = df.columns[1:]
nch = len(channels)

# 先把所有通道的 原始/處理後/時間軸 都整理好，方便一次畫
orig_list = []   # 每通道的原始信號（依顯示設定：ROI 或整段）
proc_list = []   # 每通道的處理後信號（依顯示設定：ROI 或整段）
time_list = []   # 對應的時間軸
labels = []      # 清理過的通道標籤

for col in channels:
    # 取出該通道並對齊有效時間與排序
    x = pd.to_numeric(df[col], errors="coerce").to_numpy()
    x = x[valid_t]
    if sort_idx is not None:
        x = x[sort_idx]

    # 缺值處理（以均值填補）
    if np.any(~np.isfinite(x)):
        m = np.nanmean(x)
        x = np.where(np.isfinite(x), x, m)

    # 保留一份原始
    x_raw = x.copy()

    # 淨化：帶通 + 陷波
    x_proc = bandpass(x, fs, hp=hp_fc, lp=lp_fc, order=4)
    if use_notch:
        x_proc = notch(x_proc, fs, f0=notch_f0, Q=30)

    if show_full_duration:
        # 顯示整段：時間與信號不裁切
        t_show = time
        orig_show = x_raw
        proc_show = x_proc
    else:
        # 顯示 ROI：用處理後信號找有效段，截同一段原始與時間
        a, b = find_active_segment(x_proc, fs, win_sec=2.0, thresh=3.0, min_len_sec=5.0)
        t_show = time[a:b]
        orig_show = x_raw[a:b]
        proc_show = x_proc[a:b]

    time_list.append(t_show)
    orig_list.append(orig_show)
    proc_list.append(proc_show)
    labels.append(clean_label(col))

# ========== 繪圖 1：原始波形 ==========
fig1, axes1 = plt.subplots(
    nrows=nch, ncols=1,
    figsize=(12, max(2.4*nch, 6)),
    constrained_layout=True
)
if nch == 1:
    axes1 = np.array([axes1])

for i in range(nch):
    ax = axes1[i]
    ax.plot(time_list[i], orig_list[i], linewidth=1.0)
    ax.set_title(f"原始 {labels[i]}", fontsize=11, pad=6)
    if i == nch - 1:
        ax.set_xlabel("時間 秒", fontsize=10)
    ax.set_ylabel("振幅", fontsize=10)
    ax.grid(True, linewidth=0.4)
    ax.ticklabel_format(style='plain', axis='y')  # 關閉 1e-5
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.6f'))

# 存圖
raw_png = os.path.join(save_dir, "raw_waveforms.png")
fig1.savefig(raw_png, dpi=150)
print("已存圖：", raw_png)

# ========== 繪圖 2：處理後波形 ==========
fig2, axes2 = plt.subplots(
    nrows=nch, ncols=1,
    figsize=(12, max(2.4*nch, 6)),
    constrained_layout=True
)
if nch == 1:
    axes2 = np.array([axes2])

for i in range(nch):
    ax = axes2[i]
    ax.plot(time_list[i], proc_list[i], linewidth=1.0)
    ax.set_title(f"處理後 {labels[i]}", fontsize=11, pad=6)
    if i == nch - 1:
        ax.set_xlabel("時間 秒", fontsize=10)
    ax.set_ylabel("振幅", fontsize=10)
    ax.grid(True, linewidth=0.4)
    ax.ticklabel_format(style='plain', axis='y')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.6f'))

processed_png = os.path.join(save_dir, "processed_waveforms.png")
fig2.savefig(processed_png, dpi=150)
print("已存圖：", processed_png)




# ========== 範例風格：一張圖包含「處理後波形 / 低頻 / 高頻」 ==========
example_channel_index = 4  # 要畫的通道

# 取該通道的處理後 ROI 與時間軸
x_roi = proc_list[example_channel_index]
t_roi = time_list[example_channel_index]
name  = labels[example_channel_index]

# 頻譜（平滑）
# （改為：從「原始 ROI」產生兩條專用訊號，避免被 8Hz 低通吃掉高頻）
x_roi_raw = orig_list[example_channel_index]
x_low, x_high = make_psd_signals(x_roi_raw, fs, use_notch=use_notch, notch_f0=notch_f0)
fL, PxxL = welch_psd(x_low,  fs, seg_sec=6.0)
fH, PxxH = welch_psd(x_high, fs, seg_sec=6.0)
PxxL_s = smooth_psd(PxxL, win=5)
PxxH_s = smooth_psd(PxxH, win=5)

# 各頻帶設定（和範例更接近）
LOW_MAX = 30.0            # 低頻圖 x 軸到 30 Hz（範例左下）
HF_MIN, HF_MAX = 13.0, 50.0  # 高頻圖 13–50 Hz（範例右下）
xmax = min(HF_MAX, fs/2 - 0.01)

# 計算高頻帶的基準（紅色虛線：用中位數）
hf_mask = (fH >= HF_MIN) & (fH <= xmax)
hf_floor = np.median(PxxH_s[hf_mask]) if np.any(hf_mask) else np.nan

# 找高頻帶的最高峰，畫箭頭
hf_peak_f = hf_peak_p = None
if np.any(hf_mask):
    idx_hf = np.argmax(PxxH_s[hf_mask])
    f_hf   = fH[hf_mask][idx_hf]
    p_hf   = PxxH_s[hf_mask][idx_hf]
    hf_peak_f, hf_peak_p = float(f_hf), float(p_hf)

# 版面配置：上 1 大（波形），下 2 小（低頻 / 高頻）
import matplotlib.gridspec as gridspec
fig4 = plt.figure(figsize=(12, 7), constrained_layout=True)
gs   = gridspec.GridSpec(2, 2, figure=fig4, height_ratios=[1.8, 1.2])

# ---- 上：處理後波形（ROI） ----
ax_top = fig4.add_subplot(gs[0, :])
ax_top.plot(t_roi, x_roi, linewidth=1.1)
ymin, ymax = float(np.min(x_roi)), float(np.max(x_roi))
ax_top.set_ylim(ymin, ymax + 0.15 * (ymax - ymin))
ax_top.text(0.5, 1.06, f"{name} 處理後波形",
            ha='center', va='bottom', fontsize=12,
            transform=ax_top.transAxes, clip_on=False)

ax_top.set_xlabel("時間 (秒)", fontsize=10)
ax_top.set_ylabel("振幅", fontsize=10)
ax_top.grid(True, linewidth=0.6, alpha=0.8)
ax_top.ticklabel_format(style='plain', axis='y')
ax_top.yaxis.set_major_formatter(FormatStrFormatter('%.6f'))
for s in ['top','right']:
    ax_top.spines[s].set_visible(False)

# ---- 左下：低頻圖（0–30 Hz）----
ax_low = fig4.add_subplot(gs[1, 0])
ax_low.plot(fL, PxxL_s, linewidth=1.1)
ax_low.set_xlim(0, min(LOW_MAX, fs/2 - 0.01))
ax_low.set_title(f"{name} 低頻圖", fontsize=11, pad=6)
ax_low.set_xlabel("頻率 (Hz)", fontsize=10)
style_axes(ax_low)

# 標出基頻（0.3–5 Hz），凸起更直觀
lo_mask = (fL >= 0.3) & (fL <= 5.0)
if np.any(lo_mask):
    f_lo, P_lo = fL[lo_mask], PxxL_s[lo_mask]
    df = max(fL[1]-fL[0], 1e-6)
    min_dist = max(1, int(0.5/df))  # 至少 0.5 Hz 的峰距
    pk_idx, _ = find_peaks(P_lo, distance=min_dist)
    if pk_idx.size:
        main = pk_idx[np.argmax(P_lo[pk_idx])]
        f0, p0 = f_lo[main], P_lo[main]
        ax_low.plot([f0], [p0], 'o')
        ax_low.annotate(f"基頻 ~{f0:.2f} Hz", xy=(f0, p0),
                        xytext=(f0 + 2.0, p0 * 1.15),
                        arrowprops=dict(arrowstyle='->', lw=1.0))

# ---- 右下：高頻圖（13–50 Hz） ----
ax_hf = fig4.add_subplot(gs[1, 1])
ax_hf.plot(fH, PxxH_s, linewidth=1.1)
ax_hf.set_xlim(0, max(50, xmax))  # 讓 0–50 的尺度呈現更像範例
ax_hf.set_title(f"{name} 高頻圖", fontsize=11, pad=6)
ax_hf.set_xlabel("頻率 (Hz)", fontsize=10)
style_axes(ax_hf)

# 淡色區塊標示 13–50 Hz，紅色虛線為高頻帶基準線（中位數）
if fs/2 > HF_MIN:
    ax_hf.axvspan(HF_MIN, xmax, alpha=0.12, label='13–50 Hz')
if np.isfinite(hf_floor):
    ax_hf.axhline(hf_floor, linestyle='--', linewidth=1.0, color='tab:red')
    ax_hf.text(HF_MIN + 0.5, hf_floor*1.05, "高頻帶基準", fontsize=9, color='tab:red')

# 高頻峰值箭頭（接近範例右下的粉色箭頭效果）
if hf_peak_f is not None:
    ax_hf.plot([hf_peak_f], [hf_peak_p], 'o')
    ax_hf.annotate(f"{hf_peak_f:.1f} Hz 峰", xy=(hf_peak_f, hf_peak_p),
                   xytext=(hf_peak_f + 3, hf_peak_p * 1.18),
                   arrowprops=dict(arrowstyle='->', lw=1.2))

example_png = os.path.join(save_dir, f"{name}_example_style.png")
fig4.savefig(example_png, dpi=180, bbox_inches='tight')
print("已存圖：", example_png)



plt.show()