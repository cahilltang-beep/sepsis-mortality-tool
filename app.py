
import streamlit as st
import xgboost as xgb
import numpy as np
import pickle
import pandas as pd

# 1. 标题与说明
st.title("Sepsis 28-Day Mortality Predictor")
st.markdown("### Clinical Tool for EICU/MIMIC-IV Data")
st.write("Please input the patient's clinical features to estimate the 28-day mortality risk.")

# 2. 创建侧边栏输入 (保持特征顺序与模型训练一致)
st.sidebar.header("Input Features")

# 假设特征名对应你模型里的列名
age = st.sidebar.number_input("Age (Years)", 18, 100, 65)
sofa = st.sidebar.slider("SOFA Score", 0, 24, 5)
is_vent = st.sidebar.selectbox("Mechanical Ventilation", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
is_vaso = st.sidebar.selectbox("Vasopressor Use", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
has_cancer = st.sidebar.selectbox("Malignancy History", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
pathogen_count = st.sidebar.number_input("Pathogen Count (mNGS)", 0, 10, 1)
burden_cat = st.sidebar.slider("Burden Category", 0, 3, 1)

# 3. 计算按钮
if st.button("Calculate Mortality Risk"):
    # 构建 DataFrame (保持与训练时的顺序和列名完全一致)
    input_data = pd.DataFrame({
        'anchor_age': [age],
        'pathogen_count': [pathogen_count],
        'burden_category': [burden_cat],
        'SOFA': [sofa],
        'is_vent': [is_vent],
        'is_vaso': [is_vaso],
        'has_cancer': [has_cancer]
    })

    # 这里假设你已经用 pickle 保存了 clf_full
    # 如果没保存，可以在这里暂时用 clf_full 变量 (前提是当前 notebook 没关闭)
    # 建议先运行一次: import pickle; pickle.dump(clf_full, open("sepsis_model.pkl", "wb"))
    try:
        model = pickle.load(open("sepsis_model.pkl", "rb"))
        prob = model.predict_proba(input_data)[:, 1]

        # 显示结果
        st.write("---")
        st.write(f"### Predicted Risk of Mortality: {prob[0]*100:.2f}%")

        # 根据风险给出临床建议
        if prob[0] > 0.5:
            st.error("High Risk: Consider escalated monitoring and ICU care.")
        else:
            st.success("Lower Risk: Standard care.")

    except FileNotFoundError:
        st.error("Model file 'sepsis_model.pkl' not found. Please save your model first!")

