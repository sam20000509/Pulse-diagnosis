# streamlit run C:\Users\趙立乘\Desktop\app_Huang.py    
import streamlit as st
import pandas as pd
import numpy as np
import os
from scipy.signal import find_peaks, welch, butter, filtfilt, detrend
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.tools import tool
from langchain_core.documents import Document
from opencc import OpenCC
import joblib

os.environ["HF_TOKEN"] = "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"

# 初始化 OpenCC (簡體轉台灣繁體)
cc = OpenCC('s2twp')

# ==========================================
# 0. 全局設定 & 黃進明臟腑寸關尺映射
# ==========================================
st.set_page_config(
    page_title="AI 中醫脈診智能體",
    page_icon="☯️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 黃進明醫師的寸關尺臟腑對應表
ORGAN_MAP = {
    'Lcun': '左寸 (心)', 'Lguan': '左關 (肝)', 'Lchi': '左尺 (腎陰)',
    'Rcun': '右寸 (肺)', 'Rguan': '右關 (脾)', 'Rchi': '右尺 (腎陽)'
}

st.markdown("""
    <style>
    .main { background-color: #0B0E14; }
    h1, h2, h3 { color: #00D4AA !important; font-family: 'Microsoft JhengHei', sans-serif; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 60px; }
    .stTabs [data-baseweb="tab"] p { 
        font-size: 24px !important; 
        font-weight: bold !important; 
        color: #CCD6F6 !important; 
    }
    .stTabs [data-baseweb="tab"] div {
        font-size: 24px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 1. 資料處理快取層
# ==========================================
@st.cache_data(show_spinner=False)
def load_and_validate_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            return None, "檔案為空，請檢查資料來源。"
        if 'time' not in df.columns:
            return None, "CSV 缺少必要的 'time' 欄位。"
        
        EXPECTED_PULSES = ['Lcun', 'Lguan', 'Lchi', 'Rcun', 'Rguan', 'Rchi']
        valid_pulses = [col for col in EXPECTED_PULSES if col in df.columns and not df[col].isnull().all()]
        
        if not valid_pulses:
            return None, "找不到有效的脈位數據 (Lcun, Lguan, Lchi 等)。"
            
        return df, valid_pulses
    except Exception as e:
        return None, f"檔案解析失敗：{str(e)}"
    


PULSE_THRESHOLDS = {
    "hr_shu": 90,
    "hr_normal_min": 54.300,
    "hr_huan_min": 48.062,
    "cv_jie": 0.670,
    "cv_dai": 0.250,
    "cv_cu": 0.228,
    "amp_da_ratio": 2.221,
    "amp_xiao_ratio": 0.793,
    "cv_bujun_rr": 0.089,
    "amp_cv_bujun": 0.495,
    "amp_cv_se": 0.527,
    "amp_cv_xian": 0.021,
    "amp_cv_hua_max": 0.005,
    "lf_qizhi": 1.095e+00,
    "lf_se_max": 1.269e-01,
    "lf_yinxu": 1.527e-03,
    "hf_qizhi": 0.592,
    "hf_shishi": 0.432,
    "hf_xian": 0.134,
    "hf_hua_max": 0.092,
    "hf_shixu": 0.041,
    "ep_qizhi": 46.392,
    "ep_shishi": 17.573,
    "ep_xian": 14.308,
}

def extract_raw_components(df, valid_pulses):
    raw_components = {}
    FS = 3200.0
    nyq = 0.5 * FS
    # 根據黃進明醫師脈診分析，常使用 50Hz 內訊號，為了保留高頻資訊，濾波器放寬到 50Hz
    b, a = butter(4, 50.0 / nyq, btype='low')
    
    amplitudes = []
    for col in valid_pulses:
        sig = df[col].dropna().values
        if len(sig) > 50:
            amplitudes.append(np.std(detrend(sig)))
    median_amp = np.median(amplitudes) if amplitudes else 1.0

    pulse_name_map = {'Rcun': '右寸', 'Rguan': '右關', 'Rchi': '右尺', 'Lcun': '左寸', 'Lguan': '左關', 'Lchi': '左尺'}
    
    for col in valid_pulses:
        sig = df[col].dropna().values
        if len(sig) < 50:
            continue
            
        # 依據黃進明頻譜分析法：去趨勢 + 漢恩窗 + 能量損失補償 (頻譜校正)
        sig_detrend = detrend(sig, type='linear')
        
        # 時域峰值尋找透過濾波避免極高雜訊干擾
        sig_filt = filtfilt(b, a, sig_detrend)

        window_func = np.hanning(len(sig_filt))
        windowed_sig = sig_filt * window_func
        freqs = np.fft.rfftfreq(len(windowed_sig), d=1.0/FS)
        yf = np.fft.rfft(windowed_sig)
        
        # 振幅校正：Hanning window 造成的振幅損失需乘以 2，FFT 單邊頻譜需再乘 2 (直流與奈奎斯特除外不過近似可忽略)
        fft_vals = (2.0 / len(windowed_sig)) * np.abs(yf) * 2.0
        power = fft_vals ** 2  # 能量光譜
        
        # 尋找心率基頻 f0 (C1)，正常人約 0.8~2.0 Hz (48~120 bpm)
        hr_val = 75.0
        c1_idx = np.where((freqs >= 0.8) & (freqs <= 2.0))[0]
        f0 = 1.2
        if len(c1_idx) > 0:
            f0 = freqs[c1_idx[np.argmax(fft_vals[c1_idx])]]
            hr_val = f0 * 60
            
        # 計算心律不整指標 (CV)
        peaks, _ = find_peaks(sig_filt, distance=int(FS*0.5))
        num_peaks = len(peaks)
        cv = 0.0
        amp_cv = 0.0
        
        if num_peaks > 2:
            rr_intervals = np.diff(peaks)
            cv = np.std(rr_intervals) / np.mean(rr_intervals) if np.mean(rr_intervals) > 0 else 0
            peak_amps = sig_filt[peaks]
            amp_cv = np.std(peak_amps) / np.mean(peak_amps) if np.mean(peak_amps) > 0 else 0
                
        current_amp = np.std(sig_filt)

        # 依據現代脈診圖譜學，擷取各階諧波 (C1到C10以上) 振幅，評估低高頻
        c1_amp = np.max(fft_vals[c1_idx]) if len(c1_idx) > 0 else 1.0
        
        # 低頻指標 (C2~C6)：反映臟腑本體血流充盈度與生理活動
        lf_ratios = 0.0
        for n in range(2, 7):
            idx_n = np.where((freqs >= (n*f0 - 0.2)) & (freqs <= (n*f0 + 0.2)))[0]
            if len(idx_n) > 0:
                lf_ratios += np.max(fft_vals[idx_n]) / c1_amp
                
        # 高頻指標 (C7及以上)：反映末梢血管阻力、弦緊張力與發炎
        hf_ratios = 0.0
        for n in range(7, 15):
            idx_n = np.where((freqs >= (n*f0 - 0.2)) & (freqs <= (n*f0 + 0.2)))[0]
            if len(idx_n) > 0:
                hf_ratios += np.max(fft_vals[idx_n]) / c1_amp
        
        # 頻段總能量計算
        fft_0_10 = np.sum(power[(freqs > 0) & (freqs <= 10.0)])
        fft_10_50 = np.sum(power[(freqs > 10.0) & (freqs <= 50.0)])
        fft_13_50 = np.sum(power[(freqs > 13.0) & (freqs <= 50.0)])
        
        total_energy = fft_0_10 + fft_10_50
        ep_10_50 = (fft_10_50 / total_energy * 100) if total_energy > 0 else 0
        ser_10 = (fft_0_10 / fft_10_50) if fft_10_50 > 0 else 0
        
        raw_components[col] = {
            "hr_val": hr_val, "cv": cv, "amp_cv": amp_cv, "num_peaks": num_peaks,
            "current_amp": current_amp, "median_amp": median_amp,
            "lf_ratios": lf_ratios, "hf_ratios": hf_ratios,
            "fft_0_10": fft_0_10, "fft_10_50": fft_10_50, "fft_13_50": fft_13_50,
            "ep_10_50": ep_10_50, "ser_10": ser_10
        }
    return raw_components

def evaluate_thresholds(raw_comps, params):
    results = {}
    for col, feats in raw_comps.items():
        # 1. 心率
        if feats["hr_val"] > params["hr_shu"]: hr_str = "數"
        elif feats["hr_val"] > params["hr_normal_min"]: hr_str = "正常"
        elif feats["hr_val"] > params["hr_huan_min"]: hr_str = "緩"
        else: hr_str = "遲"
            
        # 2. 心律
        rhythm_str = "正常"
        if feats["num_peaks"] > 2:
            if feats["cv"] > params["cv_jie"]: rhythm_str = "結"
            elif feats["cv"] > params["cv_dai"]: rhythm_str = "代"
            elif feats["cv"] > params["cv_cu"]: rhythm_str = "促"
                
        # 3. 大小 (考量平均基準電流的振幅比)
        if feats["current_amp"] > feats["median_amp"] * params["amp_da_ratio"]: size_str = "大"
        elif feats["current_amp"] < feats["median_amp"] * params["amp_xiao_ratio"]: size_str = "小"
        else: size_str = "適中"

        # 4. 平滑度 & 均勻度
        smooth_str = "平"
        uniform_str = "均"
        if feats["num_peaks"] > 2:
            if feats["amp_cv"] > params["amp_cv_bujun"] or feats["cv"] > params.get("cv_bujun_rr", 0.15): 
                uniform_str = "不均"
                
            if feats["amp_cv"] > params["amp_cv_se"]: smooth_str = "澀"
            elif feats["amp_cv"] > params["amp_cv_xian"]: smooth_str = "弦"
            elif feats["amp_cv"] < params["amp_cv_hua_max"]: smooth_str = "滑"
        elif feats["num_peaks"] < 3: 
            smooth_str = "短"
            
        # 5. 低頻反應 (氣血充盈與耗弱程度)
        lf_str = "平"
        if feats["lf_ratios"] < params["lf_yinxu"]: lf_str = "陰虛"
        elif feats["lf_ratios"] > params["lf_qizhi"]: lf_str = "氣滯"
        elif feats["lf_ratios"] < params["lf_se_max"]: lf_str = "澀"
        
        # 6. 高頻反應 (發炎、疼痛、弦緊張力)
        hf_str = "勢平"
        ep = feats.get("ep_10_50", 0)
        hf_rat = feats["hf_ratios"]
        if hf_rat < params["hf_shixu"]: hf_str = "勢虛"
        elif hf_rat > params["hf_qizhi"] or ep > params.get("ep_qizhi", 100.0): hf_str = "氣滯"
        elif hf_rat > params["hf_shishi"] or ep > params.get("ep_shishi", 100.0): hf_str = "勢實"
        elif hf_rat > params["hf_xian"] or ep > params.get("ep_xian", 100.0): hf_str = "弦"
        elif hf_rat < params["hf_hua_max"]: hf_str = "滑"
        
        # 綜合診斷微調 (相應特徵輔助交叉鑑定)
        if uniform_str == "不均" and (feats["hf_ratios"] > params["hf_qizhi"]/2 or feats["lf_ratios"] > params["lf_qizhi"]):
            smooth_str = "澀"
        if feats["num_peaks"] < 3:
            smooth_str = "短"

        results[col] = {
            "心率": hr_str, "心律": rhythm_str, "大小": size_str,
            "平滑度": smooth_str, "均勻度": uniform_str, "低頻": lf_str, "高頻": hf_str
        }
    return results

@st.cache_resource
def get_vectorstore():
    """快取：全域共用同一個 Embedding 模型與 Chroma 實例，避免每次對話都重新載入。"""
    try:
        embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
        # 由於當前執行路徑可能不同，提供絕對/相對路徑的相容，這裡沿用原設定
        vectorstore = Chroma(persist_directory="./tcm_db", embedding_function=embeddings)
        return vectorstore
    except Exception as e:
        print(f"VectorStore 載入失敗: {e}")
        return None

# ==========================================
# 2. 定義 AI 工具 
# ==========================================
@tool
def search_tcm_knowledge(query: str) -> str:
    """查詢中醫知識或解釋脈象意義。"""
    try:
        vectorstore = get_vectorstore()
        if vectorstore is None:
            return "無法連線至知識庫。"
            
        docs = vectorstore.similarity_search(query, k=1)
        return f"文獻記載：{docs[0].page_content}" if docs else "資料庫中找不到相關文獻。"
    except Exception as e:
        return f"檢索失敗：{str(e)}"

@tool
def teach_tcm_agent(new_knowledge: str) -> str:
    """寫入新知識至記憶庫。"""
    try:
        vectorstore = get_vectorstore()
        if vectorstore is None:
            return "無法連線至知識庫。"
            
        doc = Document(page_content=new_knowledge)
        vectorstore.add_documents([doc])
        return "系統回報：已成功將新知識寫入 ChromaDB 永久記憶庫！"
    except Exception as e:
        return f"寫入記憶失敗：{str(e)}"

@tool
def analyze_pulse_csv(file_reference: str = "current") -> str:
    """分析脈波數據（依據黃進明醫師脈象圖譜）。提取各脈位(寸關尺)的時域波幅與高低頻能量佔比(EP, SER)，並將其分類為標準脈診特徵（心率、心律、大小、平滑度、均勻度、低頻、高頻），供 AI 進行辨證。"""
    if 'current_file_path' not in st.session_state:
        return "系統回報：尚未載入 CSV 檔案。"
    
    file_path = st.session_state['current_file_path']
    file_name = os.path.basename(file_path)

    df, valid_pulses = load_and_validate_csv(file_path)
    if df is None:
        return f"**檔案錯誤**：{valid_pulses}"

    try:
        FS = 3200.0
        duration = len(df) / FS
        report_lines = [
            f"### 現代脈診圖譜分析報告：{file_name}",
            f"- **總訊號時長**：{duration:.2f} 秒",
            "---"
        ]

        raw_comps = extract_raw_components(df, valid_pulses)
        predictions = evaluate_thresholds(raw_comps, PULSE_THRESHOLDS)

        for col in valid_pulses:
            if col in predictions:
                pred = predictions[col]
                rc = raw_comps[col]
                organ = ORGAN_MAP.get(col, col)
                report_lines.append(f"#### 脈位：{organ}")
                report_lines.append(f"- **時域特徵**：心率: {pred['心率']}({rc['hr_val']:.0f} bpm) | 心律: {pred['心律']} | 大小: {pred['大小']} | 平滑度: {pred['平滑度']} | 均勻度: {pred['均勻度']}")
                report_lines.append(f"- **頻域特徵(能譜)**：")
                report_lines.append(f"  - **低頻氣血(0-10Hz)**: {rc['fft_0_10']:.3E} | SER: {rc['ser_10']:.2f} ➡️ 狀態判斷：**{pred['低頻']}** (反映臟腑氣血充盈度落差)")
                report_lines.append(f"  - **高頻弦緊(10-50Hz)**: {rc['fft_10_50']:.3E} | EP佔比: {rc['ep_10_50']:.1f}% ➡️ 狀態判斷：**{pred['高頻']}** (反映血管張力、痛症與發炎現象)\n")

        return "\n".join(report_lines)

    except Exception as e:
        return f"**處理檔案發生錯誤**：\n{str(e)}"

# ==========================================
# 3. 初始化 LLM 與多代理協定 (Multi-agent Protocol)
# ==========================================
from typing import TypedDict, Annotated, Literal
import operator
from langgraph.graph import StateGraph, START, END

class MultiAgentState(TypedDict):
    messages: Annotated[list, operator.add]
    task_type: str
    data_report: str
    research_notes: str
    draft: str
    feedback: str
    final_answer: str

@st.cache_resource
def get_agent():
    llm = ChatOllama(model="qwen2.5:14B", temperature=0.1, top_p=0.85)

    # ---------------------------------------------------------
    # 第一層：任務分派代理 (Task Router Agent)
    # ---------------------------------------------------------
    def task_router(state: MultiAgentState):
        user_msg = state["messages"][-1].content
        if "看診" in user_msg or "分析" in user_msg or "脈" in user_msg:
            return {"task_type": "diagnosis"}
        return {"task_type": "chat"}
        
    def route_after_router(state: MultiAgentState) -> Literal["data_analyst_agent", "chief_doctor_agent"]:
        if state.get("task_type") == "diagnosis":
            return "data_analyst_agent"
        return "chief_doctor_agent"

    # ---------------------------------------------------------
    # 第二層：專業分工代理群 (Specialized Agents & Task Sharing)
    # ---------------------------------------------------------
    # 1. 數據分析代理 (Pulse Data Analyst)
    def data_analyst_agent(state: MultiAgentState):
        report = analyze_pulse_csv.invoke({"file_reference": "current"})
        return {"data_report": f"【數據分析師提取結果】\n{report}"}
        
    # 2. 中醫文獻代理 (TCM Researcher)
    def tcm_researcher_agent(state: MultiAgentState):
        report = state.get("data_report", "")
        query = "中醫脈診"
        if "氣滯" in report or "弦" in report: query = "弦脈 氣滯 臨床表現"
        elif "虛" in report or "澀" in report: query = "虛脈 澀脈 血虛徵兆"
        
        notes = search_tcm_knowledge.invoke({"query": query})
        return {"research_notes": f"【文獻研究員檢索紀錄】\n{notes}"}

    # 3. 主治老中醫 (Chief Doctor)
    def chief_doctor_agent(state: MultiAgentState):
        if state.get("task_type") == "chat":
            sys_msg = SystemMessage(content="你是一位深度融合「專業中醫理論」與「AI 脈象訊號處理技術」的溫暖老中醫。請簡短、友善且具備同理心地回應病患的日常對話。若對方想要看病，請提醒他先從左側面板上傳 CSV 脈象資料，並輸入「開始看診」。請全程使用【台灣繁體中文】。")
            ans = llm.invoke([sys_msg] + state["messages"])
            return {"final_answer": ans.content}
        else:
            sys_msg = SystemMessage(content="""你是一位深度融合「專業中醫理論」與「AI 脈象訊號處理技術」的主治老中醫，精通黃進明醫師的「現代脈診圖譜學」與傳統中醫辨證。請全程使用【台灣繁體中文】與 Markdown 語法。

請綜合【數據分析報告】與【中醫文獻】，為患者撰寫精準的診斷初稿。
【🩺 專業診療引導指南】：
1. **脈象數據摘要**：先用表格或列點，總結各臟腑(左寸心、左關肝等)的突出特例(如高頻過高、大小虛實等)，過濾掉正常的數據。
2. **脈象特徵醫理解讀**：
   - 根據「大小」判斷臟腑【虛、實】。
   - 根據「平滑度(平/澀/弦/滑/短)、均勻度(均/不均)」輔助判斷血流與血管狀態。
   - 根據「低頻(陰虛/氣滯/澀)」判斷氣血充盈度；與「高頻(勢虛/氣滯/勢實/弦/滑)」對應發炎、痛症或神經緊張。
3. **精準問診引導**：針對最異常的 1-2 個臟腑與特徵，向患者拋出 1-2 個與生活作息或病徵高度相關的問題以釐清病情。
⭕ 提出問診後，請立刻結束回覆，等待患者回答，當資訊不足時不可硬猜定論。""")
            context = f"{state.get('data_report', '')}\n\n{state.get('research_notes', '')}"
            prompt_content = f"病患主述：\n{state['messages'][-1].content}\n\n內部專業報告：\n{context}"
            ans = llm.invoke([sys_msg, HumanMessage(content=prompt_content)])
            return {"draft": ans.content}
            
    def route_after_doctor(state: MultiAgentState) -> Literal["reflection_agent", "END"]:
        if state.get("task_type") == "chat":
            return "END"
        return "reflection_agent"

    # ---------------------------------------------------------
    # 第三層：深度反饋與自我修正 (Feedback & Reflection)
    # ---------------------------------------------------------
    # 4. 醫療品質審查代理 (Reflection & Ethics Reviewer)
    def reflection_agent(state: MultiAgentState):
        sys_msg = SystemMessage(content="""你是嚴格的醫療品質與中醫倫理審查員。請檢查主診醫師的草稿：
1. 語氣是否扮演了溫暖、專業的老中醫，具備同理心與關懷特色？
2. 是否詳實根據「脈象特徵醫理解讀(低高頻、虛實等)」進行分析，且並未過度武斷推測病情引起患者恐慌？
3. 是否確實有在結尾包含對患者的「互動問診提問」(1-2個釐清病情的提問)，並在此提問後立刻停止，等待病患回覆？
4. 格式是否使用了 Markdown，以及全篇是否為台灣繁體中文？

如果不合格，請以短短1~2句話給出嚴厲、具體的修改建議 (Feedback)。
如果草稿完美符合上述四點，請回覆大寫的 'APPROVE'。""")
        draft = state["draft"]
        feedback = llm.invoke([sys_msg, HumanMessage(content=f"待審查草稿：\n{draft}")])
        return {"feedback": feedback.content}
        
    def route_after_reflection(state: MultiAgentState) -> Literal["correction_agent", "END"]:
        if "APPROVE" in state.get("feedback", "").upper():
            return "END"
        return "correction_agent"
        
    # 5. 自我修正執行代理 (Self-Correction Executer)
    def correction_agent(state: MultiAgentState):
        sys_msg = SystemMessage(content="你是一位虛心受教的專業主治老中醫。請根據【醫療品質審查員】的嚴格反饋，修改並優化你的初版診斷。保持中醫的溫暖關懷與現代脈診圖譜學的專業度，並確保結尾引導患者回答問診。修改後，請直接輸出你要給病患看的最終完整回覆，不需解釋修正過程。強制使用【台灣繁體中文】與 Markdown。")
        prompt = f"你的初稿：\n{state['draft']}\n\n審查員反饋：\n{state['feedback']}"
        ans = llm.invoke([sys_msg, HumanMessage(content=prompt)])
        return {"final_answer": ans.content}

    # ---------------------------------------------------------
    # 編排 Multi-Agent 工作流圖 (StateGraph)
    # ---------------------------------------------------------
    workflow = StateGraph(MultiAgentState)
    
    workflow.add_node("task_router", task_router)
    workflow.add_node("data_analyst_agent", data_analyst_agent)
    workflow.add_node("tcm_researcher_agent", tcm_researcher_agent)
    workflow.add_node("chief_doctor_agent", chief_doctor_agent)
    workflow.add_node("reflection_agent", reflection_agent)
    workflow.add_node("correction_agent", correction_agent)
    
    workflow.add_edge(START, "task_router")
    workflow.add_conditional_edges("task_router", route_after_router, {"data_analyst_agent": "data_analyst_agent", "chief_doctor_agent": "chief_doctor_agent"})
    workflow.add_edge("data_analyst_agent", "tcm_researcher_agent")
    workflow.add_edge("tcm_researcher_agent", "chief_doctor_agent")
    workflow.add_conditional_edges("chief_doctor_agent", route_after_doctor, {"reflection_agent": "reflection_agent", "END": END})
    workflow.add_conditional_edges("reflection_agent", route_after_reflection, {"correction_agent": "correction_agent", "END": END})
    workflow.add_edge("correction_agent", END)
    
    multi_agent_app = workflow.compile()
    
    # ---------------------------------------------------------
    # 介面橋接器 Wrapper (相容 Streamlit 原有調用方式)
    # ---------------------------------------------------------
    class MultiAgentWrapper:
        def invoke(self, inputs, config=None):
            state = multi_agent_app.invoke(inputs)
            
            # 若經過 self-correction 或 chat 會有 final_answer，否則採用 draft
            final_text = state.get("final_answer")
            if not final_text:
                final_text = state.get("draft", "主治醫師沉思中，請稍後再試。")
                
            # 顯示「多代理協同日誌 (Transparency Log)」於前端
            if state.get("task_type") == "diagnosis":
                fb = state.get("feedback", "APPROVE").split('\n')[0]
                log = f"*( ⚙️ **【Multi-Agent Protocol 執行歷程】**：*\n"
                log += f"*- 🗂️ Task Shared: `Data Analyst` 提取 CSV 頻譜特徵...*\n"
                log += f"*- 📚 Task Shared: `TCM Researcher` 調用 Chroma 檢索中醫文獻...*\n"
                log += f"*- ⚖️ Reflection: 品質審查員認為 `{fb}`...*\n"
                if "APPROVE" not in state.get("feedback", "").upper():
                    log += "*- 🔄 Self-Correction: 主治醫師已根據反饋完成自我修正。 )*\n\n---\n\n"
                else:
                    log += "*- ✅ Approval: 初診草稿符合醫療倫理規範，獲准輸出。 )*\n\n---\n\n"
                final_text = log + final_text
            
            return {"messages": list(inputs["messages"]) + [AIMessage(content=final_text)]}
            
    return MultiAgentWrapper()

agent = get_agent()

if "messages" not in st.session_state:
    st.session_state["messages"] = [AIMessage(content="您好！我是您的**專屬中醫脈診 AI 助理**，隨時為您提供專業的脈診分析！✨\n\n請先於左側面板 📂 **上傳病患的脈波資料 (.csv)**，然後告訴我：「**開始看診**」或「**請分析脈象**」。我將結合「寸關尺」定位與「頻域能量」等特徵，為您提供精準的圖譜解析與辨證建議！")]

# ==========================================
# 4. 介面佈局：側邊欄 (Sidebar) 與主頁籤 (Tabs)
# ==========================================
with st.sidebar:
    st.markdown("""
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center;">
            <img src="https://cdn-icons-png.flaticon.com/512/3063/3063076.png" width="150" style="background-color: white; border-radius: 10px;">
            <h1 style="text-align: center; margin-top: 15px; font-size: 2.2rem;">系統控制面板</h1>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    uploaded_file = st.file_uploader("📂 上傳病患脈波 (.csv)", type=['csv'])
    
    if uploaded_file:
        temp_path = f"temp_{uploaded_file.name}"
        if st.session_state.get('current_file_path') != temp_path:
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            load_and_validate_csv.clear()
            st.session_state['current_file_path'] = temp_path
            st.session_state['current_file_name'] = uploaded_file.name
            st.toast(f"成功載入：{uploaded_file.name}", icon="✅")
        
    if 'current_file_path' in st.session_state:
        file_name = st.session_state.get('current_file_name', 'temp_web_pulse.csv')
        cached_result = load_and_validate_csv(st.session_state['current_file_path'])
        if cached_result[0] is not None:
            mapped_pulses = [ORGAN_MAP.get(p, p) for p in cached_result[1]]
            st.success(f"**🟢 目前載入**：  \n`{file_name}`  \n**包含脈位**：  \n{', '.join(mapped_pulses)}")
        else:
            st.error(f"**🔴 檔案異常**：  \n`{file_name}`")
    
    st.markdown("---")
    if st.button("🧹 清除歷史對話", width='stretch'):
        st.session_state["messages"] = [AIMessage(content="對話已重置。請準備下一位病患的看診。")]
        st.rerun()

st.title("☯️ AI 中醫脈診輔助系統")
tab1, tab2 = st.tabs(["📊 脈波頻譜分析", "🤖 AI 辨證輔助"])

# ---------------- 分頁 1：數據面板 ----------------
with tab1:
    if 'current_file_path' in st.session_state:
        df, valid_pulses = load_and_validate_csv(st.session_state['current_file_path'])
        
        if df is not None:
            st.markdown("### 📌 關鍵分析指標")
            m1, m2, m3 = st.columns(3)
            
            FS = 3200.0
            duration = len(df) / FS
            
            sig_m = df[valid_pulses[0]].dropna().values
            avg_hf_ratio = 0
            if len(sig_m) > 50: 
                nyq = 0.5 * FS
                b, a = butter(4, 50.0 / nyq, btype='low')
                sig_filt_m = filtfilt(b, a, detrend(sig_m))
                
                windowed_sig_m = sig_filt_m * np.hanning(len(sig_filt_m))
                freqs_m = np.fft.rfftfreq(len(windowed_sig_m), 1.0/FS)
                fft_vals_m = (2.0 / len(windowed_sig_m)) * np.abs(np.fft.rfft(windowed_sig_m)) * 2.0
                power_m = fft_vals_m ** 2
                
                lf_power = np.sum(power_m[(freqs_m > 0) & (freqs_m <= 10.0)])
                hf_power = np.sum(power_m[(freqs_m > 10.0) & (freqs_m <= 50.0)])
                avg_hf_ratio = (hf_power / (lf_power + hf_power) * 100) if (lf_power + hf_power) > 0 else 0
            
            m1.metric("⏱️ 總時長", f"{duration:.1f} s")
            m2.metric("📊 總掃描脈位", f"{len(valid_pulses)}")
            
            ep_list = []
            for p in valid_pulses:
                sig_p = df[p].dropna().values
                if len(sig_p) < 50: continue
                sig_f = filtfilt(b, a, detrend(sig_p))
                
                windowed_sig_p = sig_f * np.hanning(len(sig_f))
                f_p = np.fft.rfftfreq(len(windowed_sig_p), 1.0/FS)
                fft_vals_p = (2.0 / len(windowed_sig_p)) * np.abs(np.fft.rfft(windowed_sig_p)) * 2.0
                pw_p = fft_vals_p ** 2

                lf_p = np.sum(pw_p[(f_p > 0) & (f_p <= 10.0)])
                hf_p = np.sum(pw_p[(f_p > 10.0) & (f_p <= 50.0)])
                if (lf_p + hf_p) > 0:
                    ep_list.append(hf_p / (lf_p + hf_p) * 100)
            avg_ep = np.mean(ep_list) if ep_list else 0
            m3.metric("⚡ 系統高頻均值 EP", f"{avg_ep:.1f} %" if avg_ep > 0 else "N/A")
            st.markdown("---")

            col1, col2 = st.columns([5, 5])
            
            with col1:
                c1_a, c1_b = st.columns([2, 1])
                c1_a.subheader("📈 時域波形")
                
                if duration > 0.5:
                    display_sec = c1_b.slider("⏳ 顯示時長", min_value=0.5, max_value=float(duration), value=min(6.0, duration), step=0.5)
                else:
                    display_sec = duration
                    c1_b.warning("⚠️ 訊號過短")
                
                with st.spinner("渲染時域波形中..."):
                    display_points = int(display_sec * FS)
                    display_df = df.head(display_points)
                    num_plots = len(valid_pulses)
                    fig_wave = make_subplots(rows=num_plots, cols=1, shared_xaxes=False, vertical_spacing=0.08)
                    
                    for i, col in enumerate(valid_pulses):
                        organ_name = ORGAN_MAP.get(col, col)
                        fig_wave.add_trace(go.Scatter(x=display_df['time'], y=display_df[col], name=organ_name, line=dict(color='#00D4AA', width=1.5)), row=i+1, col=1)
                        fig_wave.layout[f'yaxis{i+1}'].title = organ_name
                        fig_wave.layout[f'yaxis{i+1}'].tickformat = '.1e'
                        fig_wave.layout[f'yaxis{i+1}'].exponentformat = 'E'
                        
                    fig_wave.update_layout(height=180 * num_plots, template="plotly_dark", showlegend=False, margin=dict(l=20, r=20, t=20, b=20), font=dict(family="Microsoft JhengHei"))
                    fig_wave.update_xaxes(title_text="時間 (秒)")  # 每個子圖都加入 X 軸標籤
                    st.plotly_chart(fig_wave, width='stretch')
            
            with col2:
                c2_a, c2_b = st.columns([2, 1])
                c2_a.subheader("🧲 頻譜圖")
                display_freq = c2_b.slider("📉 顯示頻率上限 (Hz)", min_value=10, max_value=50, value=30, step=10)
                
                with st.spinner("計算頻譜..."):
                    fig_col2 = make_subplots(rows=num_plots, cols=1, shared_xaxes=False, vertical_spacing=0.08)
                    has_data = False

                    nyq = 0.5 * FS
                    b, a = butter(4, 50.0 / nyq, btype='low')

                    for i, col in enumerate(valid_pulses):
                        sig = df[col].dropna().values
                        if len(sig) < 50: continue

                        # 更新為黃進明 50Hz 前處理邏輯
                        sig_detrend = detrend(sig, type='linear')
                        sig_filt = filtfilt(b, a, sig_detrend)
                        
                        windowed_sig = sig_filt * np.hanning(len(sig_filt))
                        freqs = np.fft.rfftfreq(len(windowed_sig), 1.0/FS)
                        yf = np.fft.rfft(windowed_sig)
                        
                        # 振幅校正
                        fft_vals = (2.0 / len(windowed_sig)) * np.abs(yf) * 2.0
                        
                        organ_name = ORGAN_MAP.get(col, col)

                        # 低頻 FFT 曲線 (展示範圍依據拉桿)
                        mask = (freqs >= 0) & (freqs <= display_freq)
                        fig_col2.add_trace(go.Scatter(x=freqs[mask], y=fft_vals[mask], fill='tozeroy', marker_color='#38BDF8', line=dict(width=1.5)), row=i+1, col=1)

                        fig_col2.layout[f'yaxis{i+1}'].title = organ_name
                        fig_col2.layout[f'yaxis{i+1}'].tickformat = '.1e'
                        fig_col2.layout[f'yaxis{i+1}'].exponentformat = 'E'
                        has_data = True

                    if has_data:
                        fig_col2.update_layout(height=180 * num_plots, template="plotly_dark", showlegend=False, margin=dict(l=20, r=20, t=20, b=20), font=dict(family="Microsoft JhengHei"))
                        
                        # 依據拉桿自動調整 x 軸的刻度間距
                        dtick_val = 5 if display_freq <= 30 else 10
                        fig_col2.update_xaxes(range=[0, display_freq], dtick=dtick_val, title_text="頻率 (Hz)")
                        st.plotly_chart(fig_col2, width='stretch')
                    else:
                        st.warning("無法繪製頻譜圖表。")

            # === 脈診特徵分析表 ===
            st.markdown("---")
            st.subheader("📋 脈診特徵分析表")
            

            
            with st.spinner("正在進行多維度脈象特徵擷取與頻譜計算..."):
                pulse_records = []
                spectral_records = []
                
                raw_comps = extract_raw_components(df, valid_pulses)
                predictions = evaluate_thresholds(raw_comps, PULSE_THRESHOLDS)

                for col in valid_pulses:
                    if col not in predictions: continue
                    
                    pred = predictions[col]
                    organ = ORGAN_MAP.get(col, col)
                    
                    # 脈診特徵存入
                    pulse_records.append({
                        "脈位": organ,
                        "心率": pred["心率"], "心律": pred["心律"], "大小": pred["大小"],
                        "平滑度": pred["平滑度"], "均勻度": pred["均勻度"], 
                        "低頻": pred["低頻"], "高頻": pred["高頻"]
                    })
                    
                    # 頻譜能量數值存入 (從 raw_comps 拿，省下重複 FFT)
                    rc = raw_comps[col]
                    spectral_records.append({
                        "脈位": organ,
                        "FFT(0~10Hz)": f"{rc['fft_0_10']:.3E}",
                        "FFT(10~50Hz)": f"{rc['fft_10_50']:.3E}",
                        "FFT(13~50Hz)": f"{rc['fft_13_50']:.3E}",
                        "EP(10~50Hz)%": f"{rc['ep_10_50']:.3f}",
                        "SER(10)": f"{rc['ser_10']:.3f}"
                    })

                if pulse_records:
                    html_table = f"""
                    <style>
                        .pulse-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 18px; font-family: 'Microsoft JhengHei', sans-serif; text-align: center; background-color: #1a1a1a; color: #eee; }}
                        .pulse-table th, .pulse-table td {{ border: 1px solid #444; padding: 10px; }}
                        .pulse-table th {{ background-color: #333; color: #00D4AA; font-weight: bold; }}
                    </style>
                    <table class="pulse-table">
                        <thead>
                            <tr><th rowspan="2">脈位 (臟腑)</th><th colspan="2">脈數</th><th colspan="3">脈形</th><th colspan="2">脈勢</th></tr>
                            <tr><th>心率</th><th>心律</th><th>大小</th><th>平滑度</th><th>均勻度</th><th>低頻</th><th>高頻</th></tr>
                        </thead>
                        <tbody>
                    """
                    for r in pulse_records:
                        html_table += f"<tr><td>{r['脈位']}</td><td>{r['心率']}</td><td>{r['心律']}</td><td>{r['大小']}</td><td>{r['平滑度']}</td><td>{r['均勻度']}</td><td>{r['低頻']}</td><td>{r['高頻']}</td></tr>"
                    html_table += "</tbody></table>"
                    st.markdown(html_table, unsafe_allow_html=True)

            # === 脈波頻譜能量分析表 ===
            st.markdown("---")
            st.subheader("🔬 脈波頻譜能量分析表 (EP / SER)")
            
            if spectral_records:
                html_table_spec = """
                <style>
                    .spec-table { width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 18px; font-family: 'Microsoft JhengHei', sans-serif; text-align: center; background-color: #1a1a1a; color: #eee; }
                    .spec-table th, .spec-table td { border: 1px solid #444; padding: 10px; }
                    .spec-table th { background-color: #333; color: #00D4AA; font-weight: bold; }
                </style>
                <table class="spec-table">
                    <thead>
                        <tr><th>脈位 (臟腑)</th><th>FFT(0~10Hz)</th><th>FFT(10~50Hz)</th><th>FFT(13~50Hz)</th><th>EP(10~50Hz)%</th><th>SER(10)</th></tr>
                    </thead>
                    <tbody>
                """
                for r in spectral_records:
                    html_table_spec += f"<tr><td>{r['脈位']}</td><td>{r['FFT(0~10Hz)']}</td><td>{r['FFT(10~50Hz)']}</td><td>{r['FFT(13~50Hz)']}</td><td>{r['EP(10~50Hz)%']}</td><td>{r['SER(10)']}</td></tr>"
                html_table_spec += "</tbody></table>"
                st.markdown(html_table_spec, unsafe_allow_html=True)
        else:
            st.error(valid_pulses) 
    else:
        st.info("👈 請先從左側面板上傳病患的 CSV 檔案。")

# ---------------- 分頁 2：AI 辨證對話 ----------------
with tab2:
    chat_box = st.container(height=600)
    
    with chat_box:
        for msg in st.session_state["messages"]:
            if isinstance(msg, HumanMessage):
                with st.chat_message("user", avatar="👨‍⚕️"): st.write(msg.content)
            else:
                with st.chat_message("assistant", avatar="☯️"): st.write(cc.convert(msg.content))

    u_input = st.chat_input("輸入對話（例如：請分析這份檔案）...")
    
    if u_input:
        with chat_box:
            with st.chat_message("user", avatar="👨‍⚕️"): st.write(u_input)
        st.session_state["messages"].append(HumanMessage(content=u_input))
        
        with chat_box:
            with st.chat_message("assistant", avatar="☯️"):
                with st.spinner("AI 正在解析頻譜與醫理..."):
                    try:
                        query_messages = st.session_state["messages"][-10:].copy()
                        
                        if "分析" in u_input or "檔案" in u_input:
                            if 'current_file_path' not in st.session_state:
                                ans = "系統目前**尚未載入任何病患 CSV 檔案**。請先從左側面板上傳檔案後，再次告訴我「開始分析」。"
                                st.write(ans)
                                st.session_state["messages"].append(AIMessage(content=ans))
                                st.stop()
                            
                            pulse_report = analyze_pulse_csv.invoke({"file_reference": "doc"})
                            
                            augmented_input = f"{u_input}\n\n【系統傳送的脈波報告】\n{pulse_report}\n\n你的任務：給出明確的【初步脈型判斷】，然後提出 1~2 個問診題目。絕對嚴禁提問檔案格式或上傳。"
                            query_messages[-1] = HumanMessage(content=augmented_input)
                            
                        res = agent.invoke({"messages": query_messages})
                        ans = cc.convert(res['messages'][-1].content)
                        st.write(ans)
                        st.session_state["messages"].append(AIMessage(content=ans))
                    except Exception as e:
                        st.error(f"系統發生例外錯誤：{e}")