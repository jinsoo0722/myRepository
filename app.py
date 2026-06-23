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
# [CORE FUNCTIONS]
# -----------------------------------------------------------------------------
def get_vibration_array(mat_data):
    """MAT 파일 내에서 'DE_time'이 포함된 변수를 찾아 자동 추출"""
    for key in mat_data.keys():
        if "DE_time" in key:  
            return mat_data[key].ravel()
    return None

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
# [SIDEBAR UI & FILE UPLOADER]
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("📂 데이터 파일 업로드")
    st.info("CWRU 데이터셋(.mat) 파일을 각각 업로드하세요.")
    
    # 파일 업로더 컴포넌트 추가
    uploaded_normal = st.file_uploader("정상 상태(.mat) 파일 선택", type=["mat"])
    uploaded_fault = st.file_uploader("이상 상태(.mat) 파일 선택", type=["mat"])
    
    st.divider()
    FS = st.number_input("샘플링 주파수 (Hz)", value=12000, step=1000)
    
    st.divider()
    st.subheader("📊 Diagnosis Threshold")
    kurtosis_threshold = st.slider("Kurtosis 임계치", 2.0, 10.0, 5.0)
    crest_threshold = st.slider("Crest Factor 임계치", 2.0, 10.0, 4.0)

# -----------------------------------------------------------------------------
# [MAIN APP LAYOUT & CONTROL FLOW]
# -----------------------------------------------------------------------------
st.title("⚙️ CBM 기반 설비 진동 이상 분석 시스템")
st.caption("Condition Based Maintenance | Bearing Fault Detection Dashboard")

# 두 파일이 모두 업로드 되었을 때만 시각화 및 분석을 수행
if uploaded_normal and uploaded_fault:
    
    # 1. 파일 객체로부터 데이터 로드 및 시그널 추출
    with st.spinner("MAT 파일 읽는 중..."):
        mat_normal = loadmat(uploaded_normal)
        mat_fault = loadmat(uploaded_fault)
        
        normal_signal = get_vibration_array(mat_normal)
        fault_signal = get_vibration_array(mat_fault)
        
    if normal_signal is None or fault_signal is None:
        st.error("❌ 파일 내에서 'DE_time' 변수를 찾을 수 없습니다. 올바른 CWRU 데이터셋 형식인지 확인해 주세요.")
        st.stop()

    # 2. 통계치 및 트렌드 계산
    feature_df = get_feature_df(normal_signal, fault_signal)
    trend_df, normal_win, fault_win = get_trend_df(normal_signal, fault_signal, FS)

    # 3. 규칙 기반 진단 임계치 동적 적용
    normal_baseline = normal_win[["rms", "kurtosis", "crest_factor"]].agg(["mean", "std"])
    rms_threshold = normal_baseline.loc["mean", "rms"] + 3 * normal_baseline.loc["std", "rms"]

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

    # 4. 스코어보드 변수 매핑
    normal_rms = feature_df.loc[feature_df.state=="normal", "rms"].iloc[0]
    fault_rms = feature_df.loc[feature_df.state=="fault", "rms"].iloc[0]
    danger_count = len(diagnosis_df[diagnosis_df["diagnosis"]=="위험"])

    # 5. 상단 실시간 메트릭 표시
    st.write("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("정상 신호 RMS", f"{normal_rms:.4f}")
    c2.metric("이상 신호 RMS", f"{fault_rms:.4f}", delta=f"{fault_rms-normal_rms:.4f}", delta_color="inverse")
    c3.metric("위험 감지 구간", f"{danger_count}개")
    c4.metric("Sampling Rate", f"{FS:,} Hz")
    st.write("---")

    # 6. 탭 레이아웃 렌더링
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview",
        "📈 Signal Analysis",
        "🌊 Frequency",
        "🚨 Diagnosis"
    ])

    # --- TAB 1: OVERVIEW ---
    with tab1:
        st.header("1. 데이터셋 및 분석 요약")
        st.markdown(f"- **업로드된 정상 파일:** `{uploaded_normal.name}` ({len(normal_signal):,} 포인트)")
        st.markdown(f"- **업로드된 이상 파일:** `{uploaded_fault.name}` ({len(fault_signal):,} 포인트)")
        
        st.header("2. 시간 영역 특징값 비교")
        st.dataframe(feature_df.style.highlight_max(axis=0, subset=['rms', 'peak', 'kurtosis', 'crest_factor'], color='#ffeae8'))

        plot_cols = ["rms", "peak", "kurtosis", "crest_factor"]
        feature_plot_df = feature_df.melt(id_vars="state", value_vars=plot_cols, var_name="Feature", value_name="Value")
        fig_bar = px.bar(feature_plot_df, x="Feature", y="Value", color="state", barmode="group",
                         title="정상/이상 특징값 비교", template="plotly_white",
                         color_discrete_map={'normal': '#2563eb', 'fault': '#dc2626'})
        st.plotly_chart(fig_bar, use_container_width=True)

    # --- TAB 2: SIGNAL ANALYSIS ---
    with tab2:
        st.header("3. 시간 영역 파형 비교 (초기 0.2초)")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_time_waveform(normal_signal, FS, "정상 진동 신호", color='#2563eb'), use_container_width=True)
        with col2:
            st.plotly_chart(plot_time_waveform(fault_signal, FS, "이상 진동 신호", color='#dc2626'), use_container_width=True)

        st.header("4. 구간별 특징값 추세 분석")
        for col in ["rms", "kurtosis", "crest_factor"]:
            fig_trend = px.line(trend_df, x="time_sec", y=col, color="state", title=f"구간별 {col.upper()} 추세",
                                template="plotly_white", color_discrete_map={'normal': '#2563eb', 'fault': '#dc2626'})
            st.plotly_chart(fig_trend, use_container_width=True)

    # --- TAB 3: FREQUENCY ---
    with tab3:
        st.header("5. 주파수 영역 분석 (FFT)")
        col1_fft, col2_fft = st.columns(2)
        with col1_fft:
            st.plotly_chart(plot_fft(normal_signal, FS, "정상 신호 FFT", color='#2563eb'), use_container_width=True)
        with col2_fft:
            st.plotly_chart(plot_fft(fault_signal, FS, "이상 신호 FFT", color='#dc2626'), use_container_width=True)

    # --- TAB 4: DIAGNOSIS ---
    with tab4:
        st.header("6. 규칙 기반 상태진단")
        st.subheader("진단 결과 (이상 신호 구간 분석)")
        st.dataframe(diagnosis_df[["time_sec", "rms", "kurtosis", "crest_factor", "diagnosis", "reason"]].head(20))
        
        col_summary1, col_summary2 = st.columns([1, 1])
        with col_summary1:
            st.write("### 📊 진단 요약 수량")
            st.write(diagnosis_df["diagnosis"].value_counts())
        with col_summary2:
            counts = diagnosis_df["diagnosis"].value_counts()
            fig_pie = px.pie(values=counts.values, names=counts.index, title="전체 구간 상태 비율",
                             color=counts.index, color_discrete_map={'정상': '#10b981', '주의': '#f59e0b', '위험': '#ef4444'})
            st.plotly_chart(fig_pie, use_container_width=True)

        st.header("7. CBM 관점의 의사결정")
        summary = f'''
        ### 💡 분석 결과 요약 및 CBM 해석
        - **정상 RMS:** `{feature_df.loc[feature_df['state']=='normal', 'rms'].iloc[0]:.4f}` / **이상 RMS:** `{feature_df.loc[feature_df['state']=='fault', 'rms'].iloc[0]:.4f}`
        - **정상 Kurtosis:** `{feature_df.loc[feature_df['state']=='normal', 'kurtosis'].iloc[0]:.4f}` / **이상 Kurtosis:** `{feature_df.loc[feature_df['state']=='fault', 'kurtosis'].iloc[0]:.4f}`
        - **정상 Crest Factor:** `{feature_df.loc[feature_df['state']=='normal', 'crest_factor'].iloc[0]:.4f}` / **이상 Crest Factor:** `{feature_df.loc[feature_df['state']=='fault', 'crest_factor'].iloc[0]:.4f}`

        ### 📋 적용된 알람 임계치
        - **RMS 주의 임계치 (정상 Mean + 3σ):** `{rms_threshold:.4f}`
        - **Kurtosis 임계치:** `{kurtosis_threshold}` | **Crest Factor 임계치:** `{crest_threshold}`

        ### 🛠 CBM 작업 권고안
        입력된 데이터 분석 결과, 이상 파일에서 전체적인 진동 에너지(RMS)와 급격한 충격 발생 빈도(Kurtosis)가 임계치를 초과하는 구간이 **{danger_count}개** 확인되었습니다.
        현장 운전 중 `RMS 가 {rms_threshold:.4f} 이상`으로 관찰되는 상태가 지속되면 설비 마모 및 손상이 심화되고 있음을 의미하므로 즉각적인 **현장 정비(Work Order)** 작성을 권고합니다.
        '''
        st.markdown(summary)

else:
    # 7. 파일이 업로드되지 않았을 때 표출되는 대기 대시보드 화면
    st.warning("👈 왼쪽 사이드바에서 정상 상태와 이상 상태의 (.mat) 데이터를 업로드해 주세요.")
    
    st.markdown("""
    ### ⚙️ CBM 대시보드 사용 가이드
    본 시스템은 **CWRU 베어링 데이터셋(.mat)**을 직접 업로드하여 설비의 결함 상태를 실시간으로 진단하는 상태기반정비(CBM) 시스템입니다.
    
    1. **사이드바 파일 업로드:** 왼쪽 사이드바에서 정상 데이터와 이상 데이터를 각각 드래그 앤 드롭합니다.
    2. **샘플링 주파수 확인:** 데이터 수집 규격에 맞춰 주파수 값을 입력합니다. (CWRU 기본: 12,000 Hz)
    3. **동적 분석 수행:** 파일이 업로드되면 자동으로 시간/주파수 도메인 특징량을 계산하고 룰 기반 AI 진단 결과를 탭 메뉴로 제공합니다.
    """)
