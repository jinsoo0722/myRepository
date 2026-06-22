import streamlit as st
import pandas as pd
import requests
import datetime
import plotly.express as px

# =====================================================================
# [설정] 구글 Apps Script 웹 앱 URL
# =====================================================================
APPS_SCRIPT_URL = "https://script.google.com/macros/s/AKfycbz8lyQqGH_DJQQYILemgI8cERfVWqTI1nhAQp79_Ld4lOhlnnlP_Ne5PhFTjjwp12cP/exec"

st.set_page_config(page_title="팀 예산 관리 대시보드", page_icon="📊", layout="wide")

# --- 데이터 연동 함수 ---
@st.cache_data(ttl=5) 
def load_data():
    if not APPS_SCRIPT_URL.startswith("http"):
        return pd.DataFrame()
    try:
        response = requests.get(APPS_SCRIPT_URL)
        if response.status_code == 200:
            data = response.json()
            if data:
                df = pd.DataFrame(data)
                # 데이터 타입 보정
                if 'amount' in df.columns:
                    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
                return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()

def save_data(data_dict):
    payload = {
        "action": "insert",
        "data": {
            "id": data_dict["id"],
            "member": data_dict["member"],
            "month": data_dict["month"],
            "category": data_dict["category"],
            "amount": data_dict["amount"]
        }
    }
    try:
        response = requests.post(APPS_SCRIPT_URL, json=payload)
        if response.status_code == 200 and response.json().get("status") == "success":
            return True
        else:
            st.error(f"저장 실패: {response.text}")
            return False
    except Exception as e:
        st.error(f"저장 중 오류 발생: {e}")
        return False

# --- UI 레이아웃 ---
st.title("📊 팀 예산 관리 시스템")

tab1, tab2 = st.tabs(["데이터 입력", "전체 대시보드"])

with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📝 내역 입력")
        with st.form("budget_form", clear_on_submit=True):
            member = st.selectbox("팀원 선택", ["부장님", "팀원1", "팀원2", "팀원3", "팀원4"])
            month_str = st.text_input("해당 월 (YYYY-MM)", value=datetime.date.today().strftime("%Y-%m"))
            category = st.selectbox("예산 항목", ["수선유지비", "비품", "개량공사"])
            amount = st.number_input("사용 금액 (원)", min_value=0, step=1000)
            if st.form_submit_button("기록 저장하기", use_container_width=True):
                new_data = {"id": int(datetime.datetime.now().timestamp() * 1000), "month": month_str, "member": member, "category": category, "amount": amount}
                if save_data(new_data):
                    st.success("저장되었습니다.")
                    st.cache_data.clear()
                    st.rerun()

    with col2:
        st.subheader("📂 최근 입력 내역")
        df = load_data()
        if not df.empty:
            # 키값이 'id'이므로 'id'로 정렬
            display_df = df.sort_values(by='id', ascending=False)
            st.dataframe(display_df.style.format({"amount": "{:,.0f}원"}), use_container_width=True, hide_index=True)

with tab2:
    st.subheader("전체 대시보드")
    df = load_data()
    if not df.empty:
        # 요약 지표
        total_amount = df['amount'].sum()
        cat_group = df.groupby('category')['amount'].sum()
        top_cat = cat_group.idxmax() if not cat_group.empty else "-"
        
        c1, c2, c3 = st.columns(3)
        c1.metric("전체 누적 사용액", f"{total_amount:,.0f}원")
        c2.metric("최대 사용 항목", f"{top_cat}")
        c3.metric("총 데이터 건수", f"{len(df)}건")
        
        st.divider()
        
        # 차트 영역
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown("##### 🏠 항목별 예산 분포")
            fig = px.pie(df, values='amount', names='category', hole=0.5)
            st.plotly_chart(fig, use_container_width=True)
        with chart_col2:
            st.markdown("##### 👥 팀원별 누적 사용액")
            mem_sum = df.groupby('member')['amount'].sum().reset_index()
            fig_bar = px.bar(mem_sum, x='member', y='amount')
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("데이터가 없습니다.")
