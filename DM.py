import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import matplotlib.font_manager as fm

# 设置中文字体 - 改进版本
try:
    # 尝试使用系统中已安装的中文字体
    font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    chinese_fonts = [f for f in font_list if any(name in f.lower() for name in ['simhei', 'microsoft yahei', 'msyh', 'simsun', 'stsong'])]
    
    if chinese_fonts:
        # 使用找到的第一个中文字体
        plt.rcParams['font.family'] = fm.FontProperties(fname=chinese_fonts[0]).get_name()
    else:
        # 如果没有找到中文字体，使用默认字体
        plt.rcParams['font.family'] = ['sans-serif']
        st.warning("未找到系统中文字体，图表中的中文可能无法正确显示")
except:
    plt.rcParams['font.family'] = ['sans-serif']

plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 页面配置
st.set_page_config(page_title="糖尿病预测", layout="wide")
st.title("🤖 糖尿病风险预测系统")
st.markdown("基于随机森林算法的糖尿病风险评估工具，仅供参考，不替代专业医疗建议")

# 加载数据并训练模型
@st.cache_resource
def get_model_and_data():
    # 读取数据（请确保该路径正确或改为相对路径）
    data = pd.read_csv("diabetes.csv")
    
    # 数据预处理 - 处理异常值
    data = data[(data['Glucose'] > 0) & (data['BloodPressure'] > 0) & 
               (data['BMI'] > 0)]
    
    # 拆分特征和目标变量
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 训练模型
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 计算特征重要性
    feature_importance = pd.DataFrame({
        '特征': X.columns,
        '重要性': model.feature_importances_
    }).sort_values('重要性', ascending=False)
    
    # 计算高危人群和低危人群的平均值
    high_risk_avg = data[data['Outcome'] == 1].mean()
    low_risk_avg = data[data['Outcome'] == 0].mean()
    
    return model, data, feature_importance, high_risk_avg, low_risk_avg

# 加载模型和数据
model, data, feature_importance, high_risk_avg, low_risk_avg = get_model_and_data()

# 创建标签页
tab1, tab2, tab3 = st.tabs(["预测工具", "特征重要性分析", "患者特征对比"])

with tab1:
    # 输入表单
    st.header("输入患者特征")
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("怀孕次数", 0, 20, 0)
        glucose = st.number_input("葡萄糖浓度 (mg/dL)", 0, 200, 100)
        blood_pressure = st.number_input("血压 (mmHg)", 0, 130, 70)
        skin_thickness = st.number_input("皮肤厚度 (mm)", 0, 100, 20)
    
    with col2:
        insulin = st.number_input("胰岛素水平 (μU/ml)", 0, 850, 80)
        bmi = st.number_input("BMI 指数", 0.0, 70.0, 25.0, step=0.1)
        diabetes_pedigree = st.number_input("糖尿病遗传函数", 0.0, 2.5, 0.5, step=0.01)
        age = st.number_input("年龄", 20, 100, 30)
    
    # 存储输入的特征值，供其他标签页使用
    input_features = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree,
        'Age': age
    }
    st.session_state['input_features'] = input_features
    
    # 预测
    if st.button("预测", type="primary"):
        features = np.array([list(input_features.values())])
        
        pred = model.predict(features)[0]
        proba = model.predict_proba(features)[0]
        
        # 显示预测结果
        st.subheader("预测结果")
        if pred == 1:
            st.error(f"⚠️ 有糖尿病风险: {proba[1]*100:.1f}%")
        else:
            st.success(f"✅ 无糖尿病风险: {proba[0]*100:.1f}%")

with tab2:
    st.header("特征重要性分析")
    st.markdown("以下图表展示了各特征对糖尿病预测的影响程度，数值越高表示该特征对预测结果的影响越大")
    
    # 1. 水平条形图展示特征重要性
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='重要性', y='特征', data=feature_importance, palette='viridis')
        plt.title('各特征对糖尿病预测的重要性', fontsize=14)
        plt.xlabel('重要性分数', fontsize=12)
        plt.ylabel('特征名称', fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
    
    # 2. 饼图展示前5个重要特征
    with col2:
        top_features = feature_importance.head(5)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.pie(top_features['重要性'], labels=top_features['特征'], 
               autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
        ax.axis('equal')  # 保证饼图是正圆形
        plt.title('前5位重要特征占比', fontsize=14)
        plt.tight_layout()
        st.pyplot(fig)
    
    # 3. 特征重要性解释
    st.subheader("关键特征解读")
    for i, row in top_features.iterrows():
        st.write(f"• **{row['特征']}**: 重要性分数 {row['重要性']:.3f}，是预测糖尿病的重要指标之一")

with tab3:
    st.header("患者特征与高危人群平均值对比")
    
    # 检查是否有输入特征
    if 'input_features' in st.session_state:
        input_data = st.session_state['input_features']
        
        # 转换为DataFrame以便可视化
        comparison_df = pd.DataFrame({
            '特征': list(input_data.keys()),
            '患者值': list(input_data.values()),
            '高危人群平均值': [high_risk_avg[feat] for feat in input_data.keys()],
            '健康人群平均值': [low_risk_avg[feat] for feat in input_data.keys()]
        })
        
        # 1. 多组条形图对比
        st.subheader("特征值对比 (条形图)")
        fig, ax = plt.subplots(figsize=(12, 8))
        x = np.arange(len(comparison_df))
        width = 0.25
        
        ax.bar(x - width, comparison_df['患者值'], width, label='患者值', color='#4285F4', alpha=0.8)
        ax.bar(x, comparison_df['高危人群平均值'], width, label='高危人群平均值', color='#EA4335', alpha=0.8)
        ax.bar(x + width, comparison_df['健康人群平均值'], width, label='健康人群平均值', color='#34A853', alpha=0.8)
        
        ax.set_xlabel('特征', fontsize=12)
        ax.set_ylabel('数值', fontsize=12)
        ax.set_title('患者与不同风险人群的特征对比', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['特征'], rotation=45)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        
        # 2. 雷达图对比 - 修复版本
        st.subheader("特征分布对比 (雷达图)")
        
        # 选择要显示的特征（避免太多特征导致雷达图过于复杂）
        selected_features = ['Glucose', 'BMI', 'Age', 'BloodPressure', 'Insulin']
        filtered_df = comparison_df[comparison_df['特征'].isin(selected_features)]
        
        # 归一化数据以便雷达图展示
        normalized_df = filtered_df.copy()
        for col in ['患者值', '高危人群平均值']:
            normalized_df[col] = normalized_df[col] / normalized_df[col].max()
        
        # 准备雷达图数据
        labels = normalized_df['特征'].tolist()
        stats_patient = normalized_df['患者值'].tolist()
        stats_high_risk = normalized_df['高危人群平均值'].tolist()
        
        # 闭合雷达图 - 修复长度不一致问题
        num_vars = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        
        # 闭合数据
        stats_patient += stats_patient[:1]
        stats_high_risk += stats_high_risk[:1]
        angles += angles[:1]
        
        # 创建雷达图
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        ax.plot(angles, stats_patient, 'o-', linewidth=2, label='患者值')
        ax.fill(angles, stats_patient, alpha=0.25)
        ax.plot(angles, stats_high_risk, 'o-', linewidth=2, label='高危人群平均值')
        ax.fill(angles, stats_high_risk, alpha=0.25)
        
        # 设置标签
        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        ax.set_title('患者与高危人群的特征分布对比', fontsize=14, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        st.pyplot(fig)
        
        # 3. 数据表格
        st.subheader("详细数据对比")
        st.dataframe(comparison_df, use_container_width=True)
        
    else:
        st.info("请先在'预测工具'标签页输入患者特征，然后才能查看对比分析")

# 页脚信息
st.divider()
st.caption("⚠️ 免责声明：本工具仅作辅助参考，不构成医疗建议，请务必咨询专业医师获取诊断意见")