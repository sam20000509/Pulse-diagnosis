import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import MultipleLocator
from scipy import signal, fft
import os
import platform
import tkinter as tk
from tkinter import filedialog
from matplotlib.gridspec import GridSpec
import warnings
from collections import Counter 

# --- 1. 系統設定 ---
def set_chinese_font():
    system_name = platform.system()
    if system_name == "Windows":
        font_candidates = ['msjh.ttc', 'simhei.ttf', 'arialuni.ttf']
        font_path = None
        for f in font_candidates:
            p = os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts', f)
            if os.path.exists(p):
                font_path = p
                break
        
        if font_path:
            prop = fm.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = prop.get_name()
            plt.rcParams['font.sans-serif'] = [prop.get_name()]
        else:
            plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
    elif system_name == "Darwin":
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    else:
        plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False 

set_chinese_font()
warnings.filterwarnings("ignore")

class PrecisePulseSystem:
    def __init__(self, file_path):
        self.file_path = file_path
        self.fs = None
        self.signals = {} 
        
        self.organ_map = {1: "肝", 2: "腎", 3: "脾", 4: "肺", 5: "胃"}
        self.position_map = {
            'lcun': '左寸(心)', 'lguan': '左關(肝)', 'lchi': '左尺(腎)',
            'rcun': '右寸(肺)', 'rguan': '右關(脾)', 'rchi': '右尺(命)'
        }
        self.diagnosis_results = {}

    def clean_column_name(self, col):
        return str(col).replace("'", "").replace('"', "").strip().lower()

    def load_data(self):
        try:
            print(f"--- 讀取: {os.path.basename(self.file_path)} ---")
            if self.file_path.lower().endswith('.csv'):
                df = pd.read_csv(self.file_path)
            else:
                df = pd.read_excel(self.file_path, engine='openpyxl')

            df.columns = [self.clean_column_name(c) for c in df.columns]
            time_cols = [c for c in df.columns if 'time' in c]
            if time_cols:
                self.time = df[time_cols[0]].values
                sig_cols = [c for c in df.columns if c != time_cols[0] and 'unnamed' not in c]
            else:
                self.time = df.iloc[:, 0].values
                sig_cols = df.columns[1:]

            dt = np.mean(np.diff(self.time))
            self.fs = 1.0 / dt if dt > 0 else 3200.0
            for col in sig_cols:
                self.signals[col] = df[col].values
            return True
        except Exception as e:
            print(f"讀取錯誤: {e}")
            return False

    def process_signal(self, raw_sig):
        sig = raw_sig.astype(float)
        sig = signal.detrend(sig)
        if self.fs > 500: # 降噪
             sos = signal.butter(4, 30, 'low', fs=self.fs, output='sos')
             sig = signal.sosfiltfilt(sos, sig)
        return sig

    def analyze_channel(self, sig):
        n = len(sig)
        yf = fft.fft(sig)
        xf = fft.fftfreq(n, 1/self.fs)
        
        half_n = n // 2
        freqs = xf[:half_n]
        amps = 2.0/n * np.abs(yf[:half_n])
        
        mask = (freqs > 0.6) & (freqs < 2.5) 
        if not np.any(mask): return None
        
        c1_idx = np.where(mask)[0][np.argmax(amps[mask])]
        c1_freq = freqs[c1_idx]
        c1_amp = amps[c1_idx]
        
        harmonics = {}
        for k in range(1, 11):
            target = c1_freq * k
            s_mask = (freqs > target-0.3) & (freqs < target+0.3)
            if np.any(s_mask):
                l_idx = np.argmax(amps[s_mask])
                harmonics[k] = {'freq': freqs[s_mask][l_idx], 'amp': amps[s_mask][l_idx], 'ratio': (amps[s_mask][l_idx]/c1_amp)*100}
            else:
                harmonics[k] = {'freq': target, 'amp': 0, 'ratio': 0}

        # --- 時域波形 ---
        peaks, _ = signal.find_peaks(sig, distance=self.fs*0.5)
        avg_wave = None
        idx_h1 = 0
        idx_h3 = 0
        si = 0
        
        if len(peaks) > 2:
            period = int(np.mean(np.diff(peaks)))
            offset_start = int(period * 0.15) 
            
            cycles = []
            for p in peaks[:-1]:
                start_idx = p - offset_start
                end_idx = p + period - offset_start
                if start_idx >= 0 and end_idx < len(sig):
                    seg = sig[start_idx : end_idx]
                    if seg.max() != seg.min():
                        seg = (seg - seg.min()) / (seg.max() - seg.min())
                    cycles.append(signal.resample(seg, 100))
                    
            if cycles:
                avg_wave = np.mean(cycles, axis=0)
                idx_h1 = np.argmax(avg_wave)
                offset_h3 = int(len(avg_wave) * 0.15) 
                idx_h3 = idx_h1 + offset_h3
                if idx_h3 >= 100: idx_h3 = 99
                val_h1 = avg_wave[idx_h1] if avg_wave[idx_h1] > 0 else 1e-5
                si = (avg_wave[idx_h3] / val_h1) * 100

        return {
            'freqs': freqs, 'amps': amps, 'harmonics': harmonics,
            'avg_wave': avg_wave,
            'idx_h1': idx_h1, 'idx_h3': idx_h3,
            'si': si, 'hr': c1_freq * 60
        }

    def get_pulse_type(self, si, hr):
        if hr > 90: return "數脈"
        if si > 70: return "弦脈"
        if si < 35: return "滑脈"
        return "平脈"

    def get_diagnosis_text(self, data):
        h = data['harmonics']
        si = data['si']
        hr = data['hr']
        msgs = []
        
        p_type = ""
        if hr > 90: p_type = "數脈 (熱/亢進)"
        elif si > 70: p_type = "弦脈 (血管硬/壓力大)"
        elif si < 35: p_type = "滑脈 (氣血充盈)"
        else: p_type = "平脈 (正常)"
        
        msgs.append(f"【綜合】：{p_type}")
        if h[2]['ratio'] > 90: msgs.append("● 腎(C2)過高 -> 高壓/緊繃")
        elif h[2]['ratio'] < 35: msgs.append("○ 腎(C2)不足 -> 腎虛/疲勞")
        
        return "\n".join(msgs)

    # --- 視窗一 ---
    def plot_window_1(self, cols):
        rows = len(cols)
        fig, axes = plt.subplots(rows, 2, num=1, figsize=(14, 3*rows), constrained_layout=True)
        if rows==1: axes=np.array([axes])
        fig.canvas.manager.set_window_title("視窗一：訊號細節與 FFT")
        
        for i, col in enumerate(cols):
            clean = self.process_signal(self.signals[col])
            data = self.analyze_channel(clean)
            
            if data:
                base_type = "平脈"
                if data['si'] > 70: base_type = "弦脈"
                elif data['si'] < 35: base_type = "滑脈"
                self.diagnosis_results[col] = {
                    'type': base_type,
                    'hr': data['hr'],
                    'si': data['si']
                }

            name = self.position_map.get(col, col)
            ax1 = axes[i,0]
            limit = min(len(clean), int(6*self.fs))
            ax1.plot(self.time[:limit], clean[:limit], color='tab:blue', lw=1)
            ax1.set_title(f"【{name}】連續波形")
            ax1.grid(True, alpha=0.3)
            ax1.set_ylabel("震幅", fontsize=10)
            ax1.set_xlabel("時間 (s)", fontsize=10)
            
            ax2 = axes[i,1]
            if data:
                ax2.plot(data['freqs'], data['amps'], color='tab:red', lw=1.5)
                labels={1:'肝',2:'腎',3:'脾'}
                for k in range(1,4):
                    h = data['harmonics'][k]
                    ax2.scatter(h['freq'], h['amp'], color='blue', s=15)
                    ax2.text(h['freq'], h['amp'], f"C{k}{labels[k]}", color='blue', fontsize=9, va='bottom', ha='center')
                ax2.set_title(f"'{name}' - 頻譜圖 (FFT)", fontsize=12)
                ax2.set_ylabel("強度 (Magnitude)", fontsize=10)
                ax2.set_xlabel("頻率 (Hz)", fontsize=10)
                ax2.set_xlim(0, 20)
                ax2.grid(True, alpha=0.3)

    # --- 視窗二 (顯示肝腎脾肺胃) ---
    def plot_window_2(self, cols):
        fig = plt.figure(num=2, figsize=(18, 10), constrained_layout=True)
        fig.canvas.manager.set_window_title("視窗二：AI 醫師診斷")
        gs = GridSpec(3, 2, figure=fig)
        plot_positions = [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1)]
        
        for i, col in enumerate(cols):
            if i >= 6: break
            clean = self.process_signal(self.signals[col])
            data = self.analyze_channel(clean)
            name = self.position_map.get(col, col)
            r, c = plot_positions[i]
            ax = fig.add_subplot(gs[r, c])
            
            if data and data['avg_wave'] is not None:
                t = np.linspace(0, 100, 100)
                wav = data['avg_wave']
                ax.plot(t, wav, 'k-', lw=2.5)
                ax.fill_between(t, wav, color='#e0f7fa', alpha=0.5)
                
                idx1 = data['idx_h1']
                idx3 = data['idx_h3']
                ax.scatter(t[idx1], wav[idx1], c='red', s=60, zorder=5)
                ax.text(t[idx1], wav[idx1]+0.05, "$h_1$心", color='red', fontweight='bold', ha='center')
                ax.scatter(t[idx3], wav[idx3], c='blue', s=60, zorder=5)
                ax.text(t[idx3], wav[idx3]+0.05, f"$h_3$硬\n{data['si']:.0f}%", color='blue', fontweight='bold', ha='center')
                
                # --- 柱狀圖修正區 ---
                ax_bar = ax.inset_axes([0.68, 0.55, 0.3, 0.4])
                orgs = [1,2,3,4,5]
                vals = [data['harmonics'][k]['ratio'] for k in orgs]
                ax_bar.bar(range(5), vals, color=['gray','blue','orange','cyan','brown'], alpha=0.8)
                
                # 【修正點】：使用 set_xticks 和 set_xticklabels 顯示文字
                ax_bar.set_xticks(range(5))
                ax_bar.set_xticklabels(['肝','腎','脾','肺','胃'], fontsize=9)
                
                ax_bar.set_title("共振能量(%)", fontsize=9)
                ax_bar.axhline(100, color='red', ls='--')
                # -------------------
                
                txt = self.get_diagnosis_text(data)
                ax.text(5, -0.15, txt, fontsize=11, 
                        bbox=dict(fc='#fff3e0', ec='orange', boxstyle="round,pad=0.3"), 
                        va='top')
                
                ax.set_title(f"【{name}】 HR: {data['hr']:.0f} bpm", fontsize=14, fontweight='bold')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.set_xticks([]) 
                ax.set_yticks([])
                ax.set_ylim(-0.5, 1.5)

    # --- 視窗三 ---
    def plot_window_3(self):
        fig = plt.figure(num=3, figsize=(11, 9))
        fig.canvas.manager.set_window_title("視窗三：脈診診斷報告書")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')

        if not self.diagnosis_results:
            ax.text(0.5, 0.5, "無數據", ha='center', fontsize=20)
            return

        pulse_list = [v['type'] for k, v in self.diagnosis_results.items()]
        hr_list = [v['hr'] for k, v in self.diagnosis_results.items()]
        
        count = Counter(pulse_list)
        total_pulses = len(pulse_list)
        avg_hr = np.mean(hr_list) if hr_list else 0

        if not count: return
        main_pulse_name, main_pulse_count = count.most_common(1)[0]
        percentage = (main_pulse_count / total_pulses) * 100
        
        rate_diag = ""
        is_rapid = False
        if avg_hr > 90: 
            rate_diag = "數脈 (Rapid)"
            is_rapid = True
        elif avg_hr < 60: 
            rate_diag = "遲脈 (Slow)"
        
        final_diagnosis = main_pulse_name
        if is_rapid and main_pulse_name == "弦脈":
            final_diagnosis = "弦數脈 (Wiry-Rapid)"
        elif rate_diag:
            final_diagnosis += f" 兼 {rate_diag}"

        content = []
        content.append({"t": "【中醫脈診診斷報告】", "s": 22, "w": "bold", "c": "black", "y": 0.92})
        content.append({"t": f"診斷結論：{final_diagnosis}", "s": 18, "w": "bold", "c": "crimson", "y": 0.86})
        content.append({"t": "-"*70, "s": 12, "w": "normal", "c": "gray", "y": 0.83})

        y = 0.78
        gap = 0.04
        
        content.append({"t": "一、 整體定性分析 (Overall Assessment)", "s": 14, "w": "bold", "c": "darkblue", "y": y})
        y -= gap
        
        t1 = f"1. {main_pulse_name}出現 {main_pulse_count} 次：佔比 {percentage:.0f}%。"
        t2 = f"   這代表身體的「基礎張力」傾向，血管壁普遍處於{main_pulse_name}狀態。"
        extra_meaning = ""
        if main_pulse_name == "弦脈":
            extra_meaning = "   (意義：高壓力、高張力，血管壁處於緊繃狀態)"
        elif main_pulse_name == "滑脈":
            extra_meaning = "   (意義：氣血充盈、代謝廢物堆積或生理性充血)"
            
        content.append({"t": t1, "s": 12, "w": "normal", "c": "black", "y": y})
        y -= gap
        content.append({"t": t2, "s": 12, "w": "normal", "c": "#333333", "y": y})
        y -= gap
        if extra_meaning:
            content.append({"t": extra_meaning, "s": 11, "w": "normal", "c": "#555555", "y": y})
            y -= gap
        y -= gap/2

        t3 = f"2. 心率分析：平均心率 {avg_hr:.0f} bpm。"
        t4 = ""
        if is_rapid:
            t4 = "   所有數據均顯示心率偏快(>90)，屬於「數脈」。\n   "
            if main_pulse_name == "弦脈":
                t4 += "結論：你的血管又硬(弦)、心跳又快(數)，合稱為「弦數脈」。\n   臨床意義：常見於高血壓前期、精神極度緊張、劇烈疼痛或肝陽上亢。"
            else:
                t4 += "顯示身體處於亢進或發炎發熱狀態。"
        elif avg_hr < 60:
            t4 = "   心率偏慢(<60)，屬於「遲脈」，可能為陽虛體質或運動員心臟。"
        else:
            t4 = "   心率處於正常範圍 (60-90 bpm)，節律平穩。"
            
        content.append({"t": t3, "s": 12, "w": "normal", "c": "black", "y": y})
        y -= gap
        for line in t4.split('\n'):
            content.append({"t": line, "s": 11, "w": "normal", "c": "#333333", "y": y})
            y -= gap
        y -= gap

        content.append({"t": "二、 臟腑定位分析 (Organ Specifics)", "s": 14, "w": "bold", "c": "darkblue", "y": y})
        y -= gap

        left_hand = [self.diagnosis_results.get(k, {}).get('type') for k in ['lcun', 'lguan', 'lchi']]
        if all(p == "弦脈" for p in left_hand):
            txt = "● 左手全弦（心、肝、腎）：\n" \
                  "  左手通常主「血」與「陰」。左三部全是弦脈，代表身體深層的血流阻力大，\n" \
                  "  這通常是「全身性血管緊繃」的鐵證。\n" \
                  "  特別是左尺(腎)也是弦脈，暗示這種緊繃可能源自於長期的壓力或疲勞\n" \
                  "  (腎水不足，無法涵養肝木)。"
            for line in txt.split('\n'):
                content.append({"t": line, "s": 11, "w": "normal", "c": "black", "y": y})
                y -= gap
        y -= gap/2

        rguan_type = self.diagnosis_results.get('rguan', {}).get('type')
        if main_pulse_name == "弦脈" and rguan_type in ["平脈", "滑脈"]:
            txt = f"● 右關獨{rguan_type}（脾/胃）：\n" \
                  f"  這是一個好消息！右關對應脾胃(消化系統)。\n" \
                  f"  顯示為「{rguan_type}」代表你的消化系統功能相對正常，\n" \
                  "  沒有像肝腎系統那麼大的壓力。\n" \
                  "  這說明你的問題主要在「氣血循環/神經系統」的緊繃，\n" \
                  "  而還沒嚴重影響到腸胃吸收。"
            for line in txt.split('\n'):
                content.append({"t": line, "s": 11, "w": "bold", "c": "darkgreen", "y": y})
                y -= gap
        elif rguan_type == "弦脈":
            content.append({"t": "● 右關見弦（脾/胃）：情緒壓力已影響消化(木剋土)，留意胃食道逆流。", "s": 11, "w": "normal", "c": "black", "y": y})
            y -= gap

        for item in content:
            ax.text(0.05, item['y'], item['t'], transform=ax.transAxes, fontsize=item['s'], fontweight=item['w'], color=item['c'], va='top')

    def run(self):
        if self.load_data():
            cols = list(self.signals.keys())
            display_cols = cols[:6]
            print(">>> 繪製圖表中...")
            self.plot_window_1(display_cols)
            self.plot_window_2(display_cols)
            self.plot_window_3()
            plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(title="選擇脈診數據檔案 (CSV/Excel)")
    if path:
        app = PrecisePulseSystem(path)
        app.run()
    else:
        print("未選擇檔案")