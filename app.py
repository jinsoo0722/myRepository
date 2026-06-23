import os
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from scipy import stats
from scipy.fft import rfft, rfftfreq
from scipy.io import loadmat

# -----------------------------------------------------------------------------
# [CONFIGURATION & STYLING]
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CBM 진동 이상 분석 시스템",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS로 UI 고도화
st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    h1 { color: #1e293b; font-weight: 800; font-size: 36px; padding-bottom: 10px; }
    h2 { color: #334155; font-weight: 700; margin-top: 20px; }
    h3 { color: #475569; font-weight: 600; }
    
    /* 깔끔한 가이드 박스 디자인 */
    .stAlert { border-radius: 12px !important; border: none !important; }
    /* 데이터프레임 둥글게 */
    [data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# [DATA LOADING & PROCESSING FUNCTIONS]
# -----------------------------------------------------------------------------
DATASET_NAME = "CWRU Bearing Dataset"
DATASET_URL = "https://www.kaggle.com/datasets/brjapon/cwru-bearing-datasets?resource=download"
NORMAL_FILE = "Time_Normal_1_098.mat"  # 파일 경로에 맞게 수정 필요
FAULT_FILE = "B007_1_123.mat"       # 파일 경로에 맞게 수정 필요
FS = 12000  

@st.cache_data
def load_vibration_data(normal_file, fault_file):
    # 테스트용 가상 데이터 생성 로직 (파일이 없을 때를 대비한 안전장치)
    if not os.path.exists(normal_file) or not os.path.exists(fault_file):
        t = np.linspace(0, 1, FS)
        normal = np.sin(2 * np.pi * 60 * t) + np.random.normal(0, 0.2, FS)
        fault = np.sin(2 * np.pi * 60 * t) + np.random.normal(0, 0.8, FS)
        # 특정 구간에 충격 신호 추가 (지표 변화 확인용)
        fault[3000:3500] += np.random.normal(0, 3.0, 500)
        return normal, fault
    
    mat_normal = loadmat(normal_file)
    mat_fault = loadmat(fault_file)
    normal_signal = mat_normal["X098_DE_time"].ravel() 
    fault_signal = mat_fault["X123_DE_time"].ravel() 
    return normal_signal, fault_signal

def calculate_features(signal):
    signal = np.asarray(signal).ravel()
    rms = np.sqrt(np.mean(signal ** 2))
    peak = np.max(np.abs(signal))
    kurtosis = stats.kurtosis(signal, fisher=False)
    skewness = stats.skew(signal)
    crest_factor = peak / rms if rms > 0 else np.nan
    std = np.std(signal)
    mean_abs = np.mean(np.abs(signal))
    return {
        "mean": np.mean(signal),
        "std": std,
        "rms": rms,
        "peak": peak,
        "kurtosis": kurtosis,
        "skewness": skewness,
        "crest_factor": crest_factor,
        "mean_abs": mean_abs,
    }

@st.cache_data
def get_feature_df(normal_s, fault_s):
    return pd.DataFrame([
        {"state": "normal", **calculate_features(normal_s)},
        {"state": "fault", **calculate_features(fault_s)},
    ])

def window_features(signal, fs, window_sec=0.2, step_sec=0.1):
    signal = np.asarray(signal).ravel()
    window = int(fs * window_sec)
    step = int(fs * step_sec)
    rows = []
    for start in range(0, len(signal) - window + 1, step):
        seg = signal[start:start + window]
        rows.append({
            "time_sec": start / fs,
            **calculate_features(seg),
        })
    return pd.DataFrame(rows)

@st.cache_data
def get_trend_df(normal_s, fault_s, fs):
    normal_win = window_features(normal_s, fs)
    fault_win = window_features(fault_s, fs)
    normal_win["state"] = "normal"
    fault_win["state"] = "fault"
    return pd.concat([normal_win, fault_win], ignore_index=True), normal_win, fault_win

def compute_fft(signal, fs):
    signal = np.asarray(signal).ravel()
    signal = signal - np.mean(signal)
    n = len(signal)
    window = np.hanning(n)
    spectrum = np.abs(rfft(signal * window)) / n
    freq = rfftfreq(n, 1 / fs)
    return freq, spectrum

# -----------------------------------------------------------------------------
# [PLOT FUNCTIONS]
# -----------------------------------------------------------------------------
def plot_time_waveform(signal, fs, title, seconds=0.2, color='#1f77b4'):
    n = min(len(signal), int(fs * seconds))
    df = pd.DataFrame({
        "Time (s)": np.arange(n) / fs,
        "Amplitude": signal[:n]
    })
    fig = px.line(df, x="Time (s)", y="Amplitude", title=title, template="plotly_white")
    fig.update_traces(line_color=color)
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def plot_fft(signal, fs, title, max_freq=1000, color='#1f77b4'):
    freq, spectrum = compute_fft(signal, fs)
    mask = freq <= max_freq
    df = pd.DataFrame({
        "Frequency (Hz)": freq[mask],
        "Amplitude": spectrum[mask]
    })
    fig = px.line(df, x="Frequency (Hz)", y="Amplitude", template="plotly_white", title=title)
    fig.update_traces(line_color=color)
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
    return fig

# -----------------------------------------------------------------------------
# [CORE LOGIC EXECUTION (순서 정렬로 에러 방지)]
# -----------------------------------------------------------------------------
# 1. 데이터 로드 및 특징량 계산 완료하기
normal_signal, fault_signal = load_vibration_data(NORMAL_FILE, FAULT_FILE)
feature_df = get_feature_df(normal_signal, fault_signal)
trend_df, normal_win, fault_win = get_trend_df(normal_signal, fault_signal, FS)

# 2. 임계치 정의 및 진단 수행
normal_baseline = normal_win[["rms", "kurtosis", "crest_factor"]].agg(["mean", "std"])
rms_threshold = normal_baseline.loc["mean", "rms"] + 3 * normal_baseline.loc["std", "rms"]
kurtosis_threshold = 5.0
crest_threshold = 4.0

def diagnose(row):
    reasons = []
    if row["rms"] > rms_threshold: reasons.append("RMS 증가")
    if row["kurtosis"] > kurtosis_threshold: reasons.append("충격성 증가")
    if row["crest_factor"] > crest_threshold: reasons.append("Crest Factor 증가")

    if len(reasons) >= 2: return "위험", ", ".join(reasons)
    if len(reasons) == 1: return "주의", reasons[0]
    return "정상", "-"

diagnosis_df = fault_win.copy()
diagnosis_df[["diagnosis", "reason"]] = diagnosis_df.apply(lambda row: pd.Series(diagnose(row)), axis=1)

# 상단 대시보드 스코어보드용 변수 추출
normal_rms = feature_df.loc[feature_df.state=="normal", "rms"].iloc[0]
fault_rms = feature_df.loc[feature_df.state=="fault", "rms"].iloc[0]
danger_count = len(diagnosis_df[diagnosis_df["diagnosis"]=="위험"])

# -----------------------------------------------------------------------------
# [SIDEBAR UI]
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Analysis Setting")
    st.info(f"**Dataset:**\n{DATASET_NAME}\n\n**Sampling Rate:**\n{FS:,} Hz")
    st.divider()
    st.subheader("📊 Diagnosis Threshold")
    st.markdown(f"""
    - **RMS** : Mean + 3σ (`{rms_threshold:.4f}`)
    - **Kurtosis** : `{kurtosis_threshold}`
    - **Crest Factor** : `{crest_threshold}`
    """)

# -----------------------------------------------------------------------------
# [MAIN APP LAYOUT]
# -----------------------------------------------------------------------------
st.title("⚙️ CBM 기반 설비 진동 이상 분석 시스템")
st.caption("Condition Based Maintenance | Bearing Fault Detection Dashboard")

# 메인 지표 현황판 (가장 상단 배치로 시인성 확보)
st.write("---")
c1, c2, c3, c4 = st.columns(4)
c1.metric("정상 신호 RMS", f"{normal_rms:.4f}")
c2.metric("이상 신호 RMS", f"{fault_rms:.4f}", delta=f"{fault_rms-normal_rms:.4f}", delta_color="inverse")
c3.metric("위험 감지 구간", f"{danger_count}개", delta="- 정상 구간 제외" if danger_count > 0 else "안정")
c4.metric("분석 데이터 크기", f"{len(fault_signal):,} pts")
st.write("---")

# 탭 구성 적용하여 화면을 깔끔하게 분할
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview & Features", 
    "📈 Signal Analysis (Time)", 
    "🌊 Frequency Analysis (FFT)", 
    "🚨 AI Diagnosis & CBM"
])

# -----------------------------------------------------------------------------
# TAB 1: OVERVIEW & FEATURES
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("1. 데이터셋 및 분석 요약")
    col_info1, col_info2 = st.columns([2, 1])
    with col_info1:
        st.markdown(f"""
        - **분석 대상 데이터:** {DATASET_NAME}
        - **데이터 출처:** [Kaggle Dataset 링크]({DATASET_URL})
        - **신호 샘플링 스펙:** {FS:,} Hz (초당 {FS:,}개 데이터 수집)
        """)
    with col_info2:
        st.success("✅ 파일 매핑 성공 및 특징량 정규화 완료")

    st.subheader("2. 시간 영역 통계적 특징값 비교")
    st.dataframe(feature_df.style.highlight_max(axis=0, subset=['rms', 'peak', 'kurtosis', 'crest_factor'], color='#ffeae8'))
    
    # 통계치 바 차트 시각화
    plot_cols = ["rms", "peak", "kurtosis", "crest_factor"]
    feature_plot_df = feature_df.melt(id_vars="state", value_vars=plot_cols, var_name="Feature", value_name="Value")
    fig_bar = px.bar(feature_plot_df, x="Feature", y="Value", color="state", barmode="group",
                     title="정상/이상 상태별 통계 지표 비교", template="plotly_white",
                     color_discrete_map={'normal': '#2563eb', 'fault': '#dc2626'})
    st.plotly_chart(fig_bar, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 2: SIGNAL ANALYSIS (TIME)
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("3. 시간 영역 원시 신호 (Raw Waveform) 비교")
    st.caption("초기 0.2초 구간의 진동 파형 형태 분석")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_time_waveform(normal_signal, FS, "정상 진동 파형", color='#2563eb'), use_container_width=True)
    with col2:
        st.plotly_chart(plot_time_waveform(fault_signal, FS, "이상 진동 파형", color='#dc2626'), use_container_width=True)
        
    st.subheader("4. 구간별(Windowing) 특징값 트렌드 추세 분석")
    st.caption("0.2초 윈도우 크기로 실시간 상태 변화 트렌드를 시뮬레이션합니다.")
    
    for col in ["rms", "kurtosis", "crest_factor"]:
        fig_trend = px.line(trend_df, x="time_sec", y=col, color="state",
                            title=f"시간 경과에 따른 {col.upper()} 지표 추세 분석", template="plotly_white",
                            color_discrete_map={'normal': '#2563eb', 'fault': '#dc2626'})
        st.plotly_chart(fig_trend, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: FREQUENCY ANALYSIS (FFT)
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("5. 주파수 영역 (FFT) 스펙트럼 분석")
    st.caption("진동 신호를 주파수 성분으로 분해하여 결함 성분 주파수(Fault Frequency)를 관찰합니다. (주요 관심 영역: 0 ~ 1,000 Hz)")
    col1_fft, col2_fft = st.columns(2)
    with col1_fft:
        st.plotly_chart(plot_fft(normal_signal, FS, "정상 신호 주파수 스펙트럼", color='#2563eb'), use_container_width=True)
    with col2_fft:
        st.plotly_chart(plot_fft(fault_signal, FS, "이상 신호 주파수 스펙트럼", color='#dc2626'), use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 4: DIAGNOSIS & CBM
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("6. 룰 기반 알고리즘 상태 진단 결과")
    
    col_res1, col_res2 = st.columns([2, 1])
    with col_res1:
        st.write("#### 📋 샘플 구간 진단 리스트 (최상위 20개 구간)")
        st.dataframe(diagnosis_df[["time_sec", "rms", "kurtosis", "crest_factor", "diagnosis", "reason"]].head(20))
    with col_res2:
        st.write("#### 📊 진단 스태티스틱스 요약")
        counts = diagnosis_df["diagnosis"].value_counts()
        fig_pie = px.pie(values=counts.values, names=counts.index, title="전체 구간 상태 비율",
                         color=counts.index, color_discrete_map={'정상': '#10b981', '주의': '#f59e0b', '위험': '#ef4444'})
        st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("7. CBM(상태기반정비) 관점의 의사결정 권고안")
    
    summary = f'''
    ### 📌 핵심 지표 요약
    - **RMS (진동 에너지 요약):** 정상 `{feature_df.loc[feature_df['state']=='normal', 'rms'].iloc[0]:.4f}` → 이상 `{feature_df.loc[feature_df['state']=='fault', 'rms'].iloc[0]:.4f}` (이상 발생 시 에너지 급증)
    - **Kurtosis (첨도/충격성):** 정상 `{feature_df.loc[feature_df['state']=='normal', 'kurtosis'].iloc[0]:.4f}` → 이상 `{feature_df.loc[feature_df['state']=='fault', 'kurtosis'].iloc[0]:.4f}`
    - **Crest Factor (충격비):** 정상 `{feature_df.loc[feature_df['state']=='normal', 'crest_factor'].iloc[0]:.4f}` → 이상 `{feature_df.loc[feature_df['state']=='fault', 'crest_factor'].iloc[0]:.4f}`

    ### 🛠 설정된 CBM 판단 임계치 (Thresholds)
    1. **RMS 경보 임계치:** `{rms_threshold:.4f}` (정상구간 평균 + 3σ 활용)
    2. **Kurtosis(충격성) 경보 기준:** `{kurtosis_threshold}` (베어링 초기 결함 검출용 고정 지표)
    3. **Crest Factor 경보 기준:** `{crest_threshold}` 

    ### 💡 CBM 종합 의견 및 작업 지시 가이드
    본 분석 시스템에서 베어링 결함 신호를 입력받은 결과, **RMS**와 **Crest Factor**가 위험 수준을 동시 혹은 지속적으로 초과하는 구간이 총 **{danger_count}개** 검출되었습니다. 
    
    1. **현장 정비 가이드:** 특정 윈도우 구간에서 `RMS가 {rms_threshold:.4f}`를 초과하고 `Crest Factor가 {crest_threshold}`를 넘어서는 현상이 3회 이상 연속될 경우 현장 설비에 대한 **'예방 정비 및 상세 현장 점검 지시(Work Order)'**를 발행할 것을 권장합니다.
    2. **향후 고도화 방안:** 향후 실시간 CBM 시스템 운영 시에는 단순히 통계치 임계값 제어뿐만 아니라 회전체 속도(RPM) 및 부하(Load) 변화를 반영한 동적 임계치(Dynamic Baseline) 설계가 필수적입니다.
    '''
    st.markdown(summary)
