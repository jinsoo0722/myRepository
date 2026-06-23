import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from scipy import stats
from scipy.fft import rfft, rfftfreq

from scipy.io import loadmat

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

font_path = "fonts/NanumGothic.ttf"

font_prop = fm.FontProperties(fname=font_path)

plt.rcParams["font.family"] = font_prop.get_name()
plt.rcParams["axes.unicode_minus"] = False

# Streamlit 앱 제목 설정
st.set_page_config(layout="wide", page_title="설비 진동 데이터 이상 분석 대시보드")
st.title("설비 진동 데이터 이상 분석 대시보드")

# 0. 라이브러리 불러오기 및 설정 (필요시)
plt.rcParams["axes.unicode_minus"] = False
np.random.seed(42)

# 1. 데이터셋 선택 (하드코딩 또는 사용자 입력)
DATASET_NAME = "CWRU Bearing Dataset"
DATASET_URL = "https://www.kaggle.com/datasets/brjapon/cwru-bearing-datasets?resource=download"
NORMAL_FILE = "Time_Normal_1_098.mat" # Colab 파일 시스템에 있다고 가정
FAULT_FILE = "B007_1_123.mat" # Colab 파일 시스템에 있다고 가정
FS = 12000  # 샘플링 주파수

st.header("1. 데이터셋 정보")
st.write(f"**데이터셋:** {DATASET_NAME}")
st.write(f"**출처:** {DATASET_URL}")
st.write(f"**샘플링 주파수:** {FS} Hz")

# 2. 데이터 불러오기 (MAT 파일)
st.header("2. 데이터 불러오기")
@st.cache_data # 데이터 로딩을 캐싱하여 성능 향상
def load_vibration_data(normal_file, fault_file):
    mat_normal = loadmat(normal_file)
    mat_fault = loadmat(fault_file)
    normal_signal = mat_normal["X098_DE_time"].ravel() # CWRU 특정 변수명
    fault_signal = mat_fault["X123_DE_time"].ravel() # CWRU 특정 변수명
    return normal_signal, fault_signal

normal_signal, fault_signal = load_vibration_data(NORMAL_FILE, FAULT_FILE)
st.write(f"정상 신호 크기: {normal_signal.shape}")
st.write(f"이상 신호 크기: {fault_signal.shape}")

# 3. 시간 영역 파형 비교 함수
def plot_time_waveform(signal, fs, title, seconds=0.2):
    n = min(len(signal), int(fs * seconds))
    x = np.arange(n) / fs
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(x, signal[:n])
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.3)
    return fig

st.header("3. 시간 영역 파형 비교")
col1, col2 = st.columns(2)
with col1:
    st.pyplot(plot_time_waveform(normal_signal, FS, "정상 진동 신호"))
with col2:
    st.pyplot(plot_time_waveform(fault_signal, FS, "이상 진동 신호"))


# 4. 시간 영역 특징값 계산 함수
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

feature_df = get_feature_df(normal_signal, fault_signal)

st.header("4. 시간 영역 특징값 비교")
st.dataframe(feature_df)

plot_cols = ["rms", "peak", "kurtosis", "crest_factor"]
fig_bar, ax_bar = plt.subplots(figsize=(10, 4))
feature_df.set_index("state")[plot_cols].T.plot(kind="bar", figsize=(10, 4), ax=ax_bar)
ax_bar.set_title("정상/이상 특징값 비교")
ax_bar.set_ylabel("Feature value")
ax_bar.tick_params(axis='x', labelrotation=0) # Corrected line: use tick_params for label rotation
ax_bar.grid(axis="y", alpha=0.3)
st.pyplot(fig_bar)


# 5. 주파수 영역 분석 함수
def compute_fft(signal, fs):
    signal = np.asarray(signal).ravel()
    signal = signal - np.mean(signal)
    n = len(signal)
    window = np.hanning(n)
    spectrum = np.abs(rfft(signal * window)) / n
    freq = rfftfreq(n, 1 / fs)
    return freq, spectrum

def plot_fft(signal, fs, title, max_freq=1000):
    freq, spectrum = compute_fft(signal, fs)
    mask = freq <= max_freq
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freq[mask], spectrum[mask])
    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.3)
    return fig

st.header("5. 주파수 영역 분석")
col1_fft, col2_fft = st.columns(2)
with col1_fft:
    st.pyplot(plot_fft(normal_signal, FS, "정상 신호 FFT"))
with col2_fft:
    st.pyplot(plot_fft(fault_signal, FS, "이상 신호 FFT"))


# 6. 구간별 특징값 추세 분석 함수
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

trend_df, normal_win, fault_win = get_trend_df(normal_signal, fault_signal, FS)

st.header("6. 구간별 특징값 추세 분석")
for col in ["rms", "kurtosis", "crest_factor"]:
    fig_trend, ax_trend = plt.subplots(figsize=(10, 3))
    for state, group in trend_df.groupby("state"):
        ax_trend.plot(group["time_sec"], group[col], label=state)
    ax_trend.set_title(f"구간별 {col} 추세")
    ax_trend.set_xlabel("Time (s)")
    ax_trend.set_ylabel(col)
    ax_trend.legend()
    ax_trend.grid(alpha=0.3)
    st.pyplot(fig_trend)


# 7. 규칙 기반 상태진단 기준 제안 및 적용
st.header("7. 규칙 기반 상태진단")
normal_baseline = normal_win[["rms", "kurtosis", "crest_factor"]].agg(["mean", "std"])

rms_threshold = normal_baseline.loc["mean", "rms"] + 3 * normal_baseline.loc["std", "rms"]
kurtosis_threshold = 5.0
crest_threshold = 4.0

def diagnose(row):
    reasons = []
    if row["rms"] > rms_threshold:
        reasons.append("RMS 증가")
    if row["kurtosis"] > kurtosis_threshold:
        reasons.append("충격성 증가")
    if row["crest_factor"] > crest_threshold:
        reasons.append("Crest Factor 증가")

    if len(reasons) >= 2:
        return "위험", ", ".join(reasons)
    if len(reasons) == 1:
        return "주의", reasons[0]
    return "정상", "-"

diagnosis_df = fault_win.copy()
diagnosis_df[["diagnosis", "reason"]] = diagnosis_df.apply(
    lambda row: pd.Series(diagnose(row)),
    axis=1,
)

st.subheader("진단 결과 (이상 신호)")
st.dataframe(diagnosis_df[["time_sec", "rms", "kurtosis", "crest_factor", "diagnosis", "reason"]].head(20))
st.write("### 진단 요약")
st.write(diagnosis_df["diagnosis"].value_counts())


# 8. CBM 관점의 의사결정 요약
st.header("8. CBM 관점의 의사결정")

summary = f'''
- **정상 RMS:** {feature_df.loc[feature_df['state']=='normal', 'rms'].iloc[0]:.4f}
- **이상 RMS:** {feature_df.loc[feature_df['state']=='fault', 'rms'].iloc[0]:.4f}
- **정상 Kurtosis:** {feature_df.loc[feature_df['state']=='normal', 'kurtosis'].iloc[0]:.4f}
- **이상 Kurtosis:** {feature_df.loc[feature_df['state']=='fault', 'kurtosis'].iloc[0]:.4f}
- **정상 Crest Factor:** {feature_df.loc[feature_df['state']=='normal', 'crest_factor'].iloc[0]:.4f}
- **이상 Crest Factor:** {feature_df.loc[feature_df['state']=='fault', 'crest_factor'].iloc[0]:.4f}

### 진단 기준 예시
- **RMS 주의 기준:** 정상 RMS 평균 + 3σ = {rms_threshold:.4f}
- **Kurtosis 주의 기준:** {kurtosis_threshold}
- **Crest Factor 주의 기준:** {crest_threshold}

### CBM 해석
이 분석에서는 **RMS**와 **Crest Factor**가 이상 상태를 잘 설명하는 주요 특징값이었습니다. 특히, 이상 신호에서 RMS가 명확하게 증가하여 전체적인 진동 에너지의 상승을 보여주었습니다.

현장에서는 `RMS가 {rms_threshold:.4f} 이상`, 또는 `Crest Factor가 {crest_threshold} 이상`으로 지속적으로 관찰될 경우 **점검 및 정비 지시**를 고려할 수 있습니다. 예를 들어, 두 지표 중 하나라도 임계값을 초과하고, 시간 추세에서 해당 값의 상승이 확인된다면 추가적인 상세 검사를 진행해야 합니다.

실제 현장 적용 시에는 특정 설비에 대한 **기준값 설정(Baselines)**과 **트렌드 모니터링**, 그리고 다른 진단 기술(예: 오일 분석, 열화상)과의 **복합적인 분석**이 필요합니다. 또한, 진동 데이터뿐만 아니라 설비의 운전 조건(속도, 부하, 온도 등) 데이터도 함께 수집하여 분석에 활용해야 보다 정확하고 신뢰성 있는 진단을 수행할 수 있습니다.
'''
st.subheader("분석 결과 요약 및 CBM 해석")
st.markdown(summary)

