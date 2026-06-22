import streamlit as st
import pandas as pd
import requests
import datetime
import plotly.express as px

# =====================================================================
# [설정] 복사한 구글 Apps Script 웹 앱 URL을 아래에 입력하세요.
# 예시: "https://script.google.com/macros/s/AKfycb.../exec"
# =====================================================================
APPS_SCRIPT_URL = "https://script.google.com/macros/s/AKfycbz8lyQqGH_DJQQYILemgI8cERfVWqTI1nhAQp79_Ld4lOhlnnlP_Ne5PhFTjjwp12cP/exec"

st.set_page_config(page_title="팀 예산 관리 대시보드", page_icon="📊", layout="wide")

# --- 데이터 연동 함수 ---
@st.cache_data(ttl=5) # 5초마다 데이터 갱신
def load_data():
    if "YOUR_APPS_SCRIPT" in APPS_SCRIPT_URL or not APPS_SCRIPT_URL.startswith("http"):
        return pd.DataFrame(columns=['id', 'month', 'member', 'category', 'amount', 'timestamp'])
    try:
        response = requests.get(APPS_SCRIPT_URL)
        if response.status_code == 200:
            data = response.json()
            if data:
                df = pd.DataFrame(data)
                df['amount'] = pd.to_numeric(df['amount'])
                return df
        return pd.DataFrame(columns=['id', 'month', 'member', 'category', 'amount', 'timestamp'])
    except Exception as e:
        st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
        return pd.DataFrame(columns=['id', 'month', 'member', 'category', 'amount', 'timestamp'])

def save_data(date_str, member, category, amount):
    # 앱스 스크립트가 요구하는 구조로 변경
    payload = {
        "action": "insert",
        "data": {
            "id": int(datetime.datetime.now().timestamp() * 1000),
            "month": date_str,
            "member": member,
            "category": category,
            "amount": amount
        }
    }
    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200 and response.json().get("status") == "success":
            st.success("데이터가 성공적으로 저장되었습니다!")
            st.cache_data.clear()
            return True
        else:
            st.error(f"저장 실패: {response.text}")
            return False
    except Exception as e:
        st.error(f"저장 중 오류 발생: {e}")
        return False

# --- UI 레이아웃 ---
st.title("📊 팀 예산 관리 시스템")
st.markdown("부장님 보고용 월별 예산 취합 및 대시보드 (DB: Google Sheets)")

tab1, tab2 = st.tabs(["데이터 입력", "전체 대시보드"])

# [탭 1] 데이터 입력 폼 및 최근 내역
with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📝 내역 입력")
        with st.form("budget_form", clear_on_submit=True):
            member = st.selectbox("팀원 선택", ["부장님", "팀원1", "팀원2", "팀원3", "팀원4"])
            
            today = datetime.date.today()
            month_str = st.text_input("해당 월 (YYYY-MM 형식)", value=today.strftime("%Y-%m"))
            
            category = st.selectbox("예산 항목", ["수선유지비", "비품", "개량공사"])
            amount = st.number_input("사용 금액 (원)", min_value=0, step=1000)
            
            submitted = st.form_submit_button("기록 저장하기", use_container_width=True)
            
            if submitted:
                new_data = {
                    "id": int(datetime.datetime.now().timestamp() * 1000),
                    "month": month_str,
                    "member": member,
                    "category": category,
                    "amount": amount
                }
                with st.spinner("구글 시트에 저장 중..."):
                    if save_data(new_data):
                        st.success("예산 데이터가 정상적으로 기록되었습니다.")
                        st.cache_data.clear() # 새 데이터 반영을 위해 캐시 삭제
                        st.rerun()

    with col2:
        st.subheader("📂 최근 입력 내역 (구글 시트 연동)")
        df = load_data()
        if not df.empty:
            # 표시용으로 데이터 가공 (id, timestamp 숨김, 최신순 정렬)
            display_df = df.sort_values(by='id', ascending=False).drop(columns=['id', 'timestamp'], errors='ignore')
            # 숫자 포맷 변경
            styled_df = display_df.style.format({"amount": "{:,.0f}원"})
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
        else:
            st.info("등록된 데이터가 없거나 구글 시트에 연결되지 않았습니다.")


# [탭 2] 데이터 분석 및 대시보드
with tab2:
    st.subheader("전체 대시보드")
    df = load_data()
    
    if df.empty:
        st.info("대시보드를 표시할 데이터가 없습니다.")
    else:
        # 요약 지표 (KPI)
        total_amount = df['amount'].sum()
        
        # 이번 달 최대 사용 항목 계산
        cat_group = df.groupby('category')['amount'].sum()
        top_category = cat_group.idxmax() if not cat_group.empty else "-"
        top_category_val = cat_group.max() if not cat_group.empty else 0
        
        data_count = len(df)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("전체 누적 사용액", f"{total_amount:,.0f}원")
        c2.metric("최대 사용 항목", f"{top_category}", f"{top_category_val:,.0f}원", delta_color="off")
        c3.metric("총 데이터 건수", f"{data_count}건")
        
        st.divider()
        
        # 차트 영역
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown("##### 🏠 항목별 예산 분포")
            cat_sum = df.groupby('category')['amount'].sum().reset_index()
            fig_donut = px.pie(cat_sum, values='amount', names='category', hole=0.5, 
                               color_discrete_sequence=['#3b82f6', '#10b981', '#8b5cf6'])
            fig_donut.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_donut, use_container_width=True)
            
        with chart_col2:
            st.markdown("##### 👥 팀원별 누적 사용액")
            mem_sum = df.groupby('member')['amount'].sum().reset_index()
            fig_bar = px.bar(mem_sum, x='member', y='amount', text='amount', 
                             color_discrete_sequence=['#60a5fa'])
            fig_bar.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            fig_bar.update_layout(yaxis_title="사용 금액 (원)", xaxis_title="팀원")
            st.plotly_chart(fig_bar, use_container_width=True)
            
        st.divider()
        
        # 피벗 테이블 영역
        st.markdown("##### 📅 월별/항목별 요약 테이블 (취합본)")
        try:
            # Pandas를 이용해 월별-항목별 피벗 테이블 생성
            pivot_df = pd.pivot_table(df, values='amount', index='month', columns='category', aggfunc='sum', fill_value=0)
            
            # 없는 카테고리 열 추가 보장
            for cat in ["수선유지비", "비품", "개량공사"]:
                if cat not in pivot_df.columns:
                    pivot_df[cat] = 0
                    
            # 행별 총합 추가
            pivot_df['월간 총합'] = pivot_df.sum(axis=1)
            
            # 최신 월이 위로 오도록 정렬
            pivot_df = pivot_df.sort_index(ascending=False)
            
            st.dataframe(
                pivot_df.style.format("{:,.0f}"),
                use_container_width=True
            )
        except Exception as e:
            st.warning("표를 생성하기에 데이터가 충분하지 않습니다.")
