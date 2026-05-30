"""
车辆总质量预测系统 - Streamlit可视化界面
基于物理估计 + ML校准，带置信度评估和截尾处理
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
from physics_ml_classifier import Config, PhysicsMLPredictor
from visualization import (
    plot_residual_analysis,
    plot_feature_importance,
    plot_cv_results,
    plot_residual_by_region,
    plot_window_comparison,
    plot_confidence_distribution,
    plot_trip_summary,
    generate_mass_time_curve,
    plot_trim_effect, generate_mass_time_curve_with_comparison,
)

warnings.filterwarnings('ignore')

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="车辆质量预测系统",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 样式配置 ====================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem; color: #1E3A8A;
        text-align: center; margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem; color: #3B82F6;
        margin-top: 2rem; margin-bottom: 1rem;
        border-bottom: 2px solid #E5E7EB; padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #F8FAFC; padding: 1rem;
        border-radius: 10px; border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ==================== 初始化session state ====================
def init_session_state():
    defaults = {
        'app_initialized': True,
        'training_data': None,
        'current_file_name': None,
        'predictor': None,
        'model_trained': False,
        'training_metrics': None,
        'training_history': None,
        'model_config': None,
        'prediction_results': None,
        'prediction_file_name': None,
        'confidence_threshold': 0.7,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ==================== 辅助函数 ====================
def load_data(uploaded_file):
    """加载上传的数据"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("不支持的文件格式")
            return None

        column_mapping = {
            'Time': 'time_seconds',
            'MCU_ActMotorTq': 'motor_torque_nm',
            'MCU_ActMotorSpd': 'motor_speed_rpm',
            'IC_CarSpeed': 'speed_kmh',
            'Acceleration': 'acceleration_x',
            'Accelerometer X': 'acceleration_x',
            'Mass': 'mass_kg',
            'Force': 'force_n',
        }

        rename_dict = {}
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                rename_dict[old_name] = new_name

        if rename_dict:
            df = df.rename(columns=rename_dict)

        return df
    except Exception as e:
        st.error(f"数据加载失败: {str(e)}")
        return None


# ==================== Streamlit应用 ====================
def main():
    st.markdown('<h1 class="main-header">🚗 车辆总质量预测系统</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #6B7280;">物理估计 + ML校准 | 截尾降噪 | 置信度评估</p>',
                unsafe_allow_html=True)

    st.sidebar.title("导航菜单")

    with st.sidebar.expander("📊 当前状态", expanded=False):
        if st.session_state.training_data is not None:
            st.success("✅ 已加载训练数据")
        if st.session_state.model_trained:
            st.success("✅ 模型已训练")
        if st.session_state.prediction_results is not None:
            st.success("✅ 有预测结果")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 刷新页面"):
                st.rerun()
        with col2:
            if st.button("🗑️ 清除所有数据"):
                for key in list(st.session_state.keys()):
                    if key != 'app_initialized':
                        del st.session_state[key]
                init_session_state()
                st.success("已清除所有数据")
                st.rerun()

    # 置信度阈值滑块
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 置信度阈值设置")
    confidence_threshold = st.sidebar.slider(
        "置信度阈值", min_value=0.0, max_value=1.0,
        value=st.session_state.confidence_threshold, step=0.05,
        help="低于此阈值的窗口将被过滤"
    )
    st.session_state.confidence_threshold = confidence_threshold
    st.sidebar.info(f"当前阈值: {confidence_threshold:.0%}")

    page = st.sidebar.radio(
        "选择页面",
        ["🏠 首页", "📊 数据探索", "🎯 模型训练", "🔮 预测分析", "💾 模型管理"]
    )

    # ========== 首页 ==========
    if page == "🏠 首页":
        st.markdown('<h2 class="section-header">系统介绍</h2>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>🔬 物理估计层</h3>
                <p>牛顿第二定律 F=ma</p>
                <p>Acc>1, Speed>10, Force>2000N</p>
                <p>3秒二级窗口均值法</p>
                <p>IQR截尾降噪处理</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>🤖 ML校准层</h3>
                <p>16个工况特征</p>
                <p>GBDT学习修正规律</p>
                <p>置信度评估体系</p>
                <p>阈值可控过滤</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>🎯 融合预测</h3>
                <p>截尾→中位数聚合</p>
                <p>窗口→行程逐层收敛</p>
                <p>置信度驱动的过滤</p>
                <p>丰富可视化分析</p>
            </div>
            """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("上传训练数据快速开始", type=['csv', 'xlsx', 'xls'])
        if uploaded_file is not None:
            with st.spinner('正在加载数据...'):
                df = load_data(uploaded_file)
                if df is not None:
                    st.session_state.training_data = df
                    st.session_state.current_file_name = uploaded_file.name
                    st.success(f"✅ 数据加载成功！共 {len(df)} 行")

    # ========== 数据探索 ==========
    elif page == "📊 数据探索":
        st.markdown('<h2 class="section-header">数据探索与分析</h2>', unsafe_allow_html=True)

        if st.session_state.training_data is not None:
            df = st.session_state.training_data

            st.markdown("### 📈 数据概览")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("数据行数", f"{len(df):,}")
            with col2:
                st.metric("数据列数", len(df.columns))
            with col3:
                time_range = 0
                if 'time_seconds' in df.columns:
                    time_vals = pd.to_numeric(df['time_seconds'], errors='coerce')
                    if len(time_vals) >= 2:
                        time_range = time_vals.max() - time_vals.min()
                st.metric("时间范围", f"{time_range:.1f}秒")
            with col4:
                if 'mass_kg' in df.columns:
                    st.metric("质量类别数", df['mass_kg'].nunique())

            with st.expander("📋 数据预览", expanded=True):
                st.dataframe(df.head(20))

        else:
            st.info("👆 请先上传数据文件")
            uploaded_file = st.file_uploader("上传新数据文件", type=['csv', 'xlsx', 'xls'])
            if uploaded_file is not None:
                with st.spinner('正在加载数据...'):
                    df = load_data(uploaded_file)
                    if df is not None:
                        st.session_state.training_data = df
                        st.session_state.current_file_name = uploaded_file.name
                        st.success(f"✅ 数据加载成功！")
                        st.rerun()

    # ========== 模型训练 ==========
    elif page == "🎯 模型训练":
        st.markdown('<h2 class="section-header">模型训练与评估</h2>', unsafe_allow_html=True)

        if st.session_state.training_data is None:
            st.warning("⚠️ 请先上传数据")
            return

        df = st.session_state.training_data

        # 显示已有模型状态
        if st.session_state.model_trained and st.session_state.training_metrics:
            st.success("✅ 模型已训练完成")

            metrics = st.session_state.training_metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("训练MAE", f"{metrics.get('mae_kg', 0):.0f} kg")
            with col2:
                st.metric("训练RMSE", f"{metrics.get('rmse_kg', 0):.0f} kg")
            with col3:
                st.metric("MAPE", f"{metrics.get('mape', 0):.1f}%")
            with col4:
                st.metric("R²", f"{metrics.get('r2', 0):.3f}")

            if 'cv_mae_mean' in metrics:
                st.markdown("### 📊 交叉验证结果")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("CV MAE", f"{metrics.get('cv_mae_mean', 0):.0f} ± {metrics.get('cv_mae_std', 0):.0f} kg")
                with col2:
                    st.metric("CV RMSE",
                              f"{metrics.get('cv_rmse_mean', 0):.0f} ± {metrics.get('cv_rmse_std', 0):.0f} kg")
                with col3:
                    st.metric("CV R²", f"{metrics.get('cv_r2_mean', 0):.3f}")

            # 训练可视化
            if st.session_state.training_history:
                st.markdown("### 📈 残差分析")
                fig_residual = plot_residual_analysis(st.session_state.training_history)
                if fig_residual:
                    st.plotly_chart(fig_residual, use_container_width=True)

                st.markdown("### 🔝 特征重要性")
                fig_importance = plot_feature_importance(st.session_state.training_history)
                if fig_importance:
                    st.plotly_chart(fig_importance, use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 🔄 交叉验证详情")
                    fig_cv = plot_cv_results(st.session_state.training_history)
                    if fig_cv:
                        st.plotly_chart(fig_cv, use_container_width=True)

                with col2:
                    st.markdown("### 📊 分区间残差")
                    fig_region = plot_residual_by_region(st.session_state.training_history)
                    if fig_region:
                        st.plotly_chart(fig_region, use_container_width=True)

                st.markdown("### 🔍 窗口级别质量对比分析")
                fig_window_comparison = plot_window_comparison(st.session_state.training_history)
                if fig_window_comparison:
                    st.plotly_chart(fig_window_comparison, use_container_width=True)

                # 误差统计
                if st.session_state.training_history.get('residuals') is not None:
                    residuals = st.session_state.training_history['residuals']
                    abs_residuals = np.abs(residuals)
                    st.markdown("#### 误差统计")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("平均绝对误差", f"{np.mean(abs_residuals):.0f} kg")
                    with col2:
                        st.metric("中位数绝对误差", f"{np.median(abs_residuals):.0f} kg")
                    with col3:
                        st.metric("最大误差", f"{np.max(abs_residuals):.0f} kg")
                    with col4:
                        within_50 = (abs_residuals < 50).sum()
                        st.metric("误差<50kg占比", f"{within_50 / len(abs_residuals) * 100:.1f}%")

                st.markdown("### ✂️ 截尾效果分析")
                fig_trim = plot_trim_effect(st.session_state.training_history)
                if fig_trim:
                    st.plotly_chart(fig_trim, use_container_width=True)

            st.markdown("---")
            if st.button("🔄 重新训练模型", type="secondary"):
                st.session_state.model_trained = False
                st.session_state.training_metrics = None
                st.session_state.training_history = None
                st.rerun()
            st.markdown("---")

        # 训练配置表单
        st.markdown("### ⚙️ 模型配置")

        # 初始化预设值
        if 'preset_values' not in st.session_state:
            st.session_state.preset_values = {
                'n_estimators': 100,
                'max_depth': 4,
                'learning_rate': 0.05,
                'min_samples_split': 5,
                'min_samples_leaf': 2
            }

        # 快速预设方案（放在表单外部）
        st.markdown("#### 🎯 快速预设方案")
        col_preset1, col_preset2, col_preset3 = st.columns(3)

        with col_preset1:
            if st.button("🛡️ 保守模式", key="preset_conservative", use_container_width=True):
                st.session_state.preset_values = {
                    'n_estimators': 30,
                    'max_depth': 3,
                    'learning_rate': 0.03,
                    'min_samples_split': 10,
                    'min_samples_leaf': 5
                }
                st.success("已切换到保守模式")
                st.rerun()

        with col_preset2:
            if st.button("⚖️ 平衡模式", key="preset_balanced", use_container_width=True):
                st.session_state.preset_values = {
                    'n_estimators': 100,
                    'max_depth': 4,
                    'learning_rate': 0.05,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2
                }
                st.success("已切换到平衡模式")
                st.rerun()

        with col_preset3:
            if st.button("🚀 激进模式", key="preset_aggressive", use_container_width=True):
                st.session_state.preset_values = {
                    'n_estimators': 200,
                    'max_depth': 6,
                    'learning_rate': 0.08,
                    'min_samples_split': 3,
                    'min_samples_leaf': 1
                }
                st.success("已切换到激进模式")
                st.rerun()

        st.info(f"📌 当前预设: 树数量={st.session_state.preset_values['n_estimators']}, "
                f"最大深度={st.session_state.preset_values['max_depth']}, "
                f"学习率={st.session_state.preset_values['learning_rate']}")

        st.markdown("---")

        with st.form("model_training_form"):
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "物理参数", "窗口参数", "分级阈值", "截尾参数", "ML参数"
            ])

            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    min_accel = st.number_input("默认最小加速度 (m/s²)", 0.3, 2.0, 1.0, 0.1,
                                                help="用于首次筛选的加速度阈值")
                    min_speed = st.number_input("最小车速 (km/h)", 5.0, 30.0, 10.0, 5.0)
                with col2:
                    min_force = st.number_input("最小驱动力 (N)", 500, 5000, 2000, 100)
                    roll_resist_coeff = st.number_input("滚阻系数 f", 0.005, 0.030, 0.015, 0.001, format="%.3f")

            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    secondary_window_size = st.number_input("二级窗口大小（秒）", 1.0, 10.0, 3.0, 0.5)
                    min_secondary_points = st.number_input("二级窗口最少点数", 5, 50, 15, 5)
                with col2:
                    min_primary_duration = st.number_input("最小一级窗口时长（秒）", 1.0, 10.0, 3.0, 0.5)
                    stop_duration_trip = st.number_input("行程结束停车时长（秒）", 60, 3600, 600, 60)

            with tab3:
                st.markdown("#### 分级加速度阈值")
                st.markdown("根据物理估计质量，动态调整加速度筛选阈值，增加重载区域样本")
                col1, col2, col3 = st.columns(3)
                with col1:
                    accel_light = st.number_input("轻载阈值 (m/s²)", 0.5, 2.0, 1.0, 0.1,
                                                  help="质量<轻载阈值时使用")
                    light_mass = st.number_input("轻载质量阈值 (kg)", 1500, 3500, 2500, 100,
                                                 help="低于此质量使用轻载加速度阈值")
                with col2:
                    accel_heavy = st.number_input("重载阈值 (m/s²)", 0.3, 1.5, 0.5, 0.1,
                                                  help="质量>重载阈值时使用")
                    heavy_mass = st.number_input("重载质量阈值 (kg)", 2000, 5000, 3000, 100,
                                                 help="高于此质量使用重载加速度阈值")
                with col3:
                    accel_transition = st.number_input("过渡区阈值 (m/s²)", 0.4, 1.5, 0.7, 0.1,
                                                       help="质量在轻载和重载之间时使用")

                st.info(f"轻载(<{light_mass}kg): Acc>{accel_light} | "
                        f"重载(>{heavy_mass}kg): Acc>{accel_heavy} | "
                        f"过渡区: Acc>{accel_transition}")

            with tab4:
                col1, col2 = st.columns(2)
                with col1:
                    trim_method = st.selectbox("截尾方法",
                                               ['iqr', 'percentile', 'adaptive'],
                                               format_func=lambda x: {
                                                   'iqr': 'IQR法',
                                                   'percentile': '百分位法',
                                                   'adaptive': '自适应'
                                               }[x])
                with col2:
                    trim_percentile_low = st.slider("截尾下限(%)", 5, 40, 25, 5)
                    trim_percentile_high = st.slider("截尾上限(%)", 60, 95, 75, 5)
                min_trim_samples = st.slider("截尾后最少样本数", 2, 10, 3, 1)

            with tab5:
                st.markdown("#### 模型类型")
                ml_model_type = st.selectbox("ML模型类型",
                                             ['gbdt', 'rf', 'linear'],
                                             format_func=lambda x: {
                                                 'gbdt': 'GBDT (推荐)',
                                                 'rf': '随机森林',
                                                 'linear': '线性回归'
                                             }[x])

                st.markdown("#### 超参数调整")
                col1, col2, col3 = st.columns(3)
                with col1:
                    n_estimators = st.slider("树的数量", 20, 300,
                                             st.session_state.preset_values['n_estimators'], 10,
                                             help="减少可降低过拟合")
                    max_depth = st.slider("最大深度", 2, 10,
                                          st.session_state.preset_values['max_depth'], 1,
                                          help="减小强制学简单规律")
                with col2:
                    learning_rate = st.slider("学习率 (GBDT)", 0.01, 0.30,
                                              st.session_state.preset_values['learning_rate'], 0.01,
                                              help="减小学得更慢更稳健")
                    min_samples_split = st.slider("最小样本分割", 2, 20,
                                                  st.session_state.preset_values['min_samples_split'], 1,
                                                  help="增大防分裂太细")
                with col3:
                    min_samples_leaf = st.slider("最小叶子样本", 1, 10,
                                                 st.session_state.preset_values['min_samples_leaf'], 1,
                                                 help="增大防记住单个样本")
                    n_folds = st.slider("交叉验证折数", 3, 10, 5, 1,
                                        help="行程少时不宜太多")

                st.markdown("#### 置信度权重")
                col1, col2 = st.columns(2)
                with col1:
                    data_quality_weight = st.slider("数据质量权重", 0.0, 1.0, 0.5, 0.1)
                with col2:
                    model_reliability_weight = st.slider("模型可靠度权重", 0.0, 1.0, 0.5, 0.1)

            train_button = st.form_submit_button("开始训练模型")

        if train_button:
            with st.spinner("正在训练模型..."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    config = Config()
                    config.MIN_ACCEL = min_accel
                    config.MIN_ACCEL_LIGHT = accel_light
                    config.MIN_ACCEL_HEAVY = accel_heavy
                    config.MIN_ACCEL_TRANSITION = accel_transition
                    config.LIGHT_MASS_THRESHOLD = light_mass
                    config.HEAVY_MASS_THRESHOLD = heavy_mass
                    config.MIN_SPEED = min_speed
                    config.MIN_FORCE = min_force
                    config.ROLLING_RESISTANCE_COEFF = roll_resist_coeff
                    config.SECONDARY_WINDOW_SIZE = secondary_window_size
                    config.MIN_SECONDARY_WINDOW_POINTS = min_secondary_points
                    config.MIN_PRIMARY_WINDOW_DURATION = min_primary_duration
                    config.MIN_STOP_DURATION_FOR_TRIP_END = stop_duration_trip
                    config.TRIM_METHOD = trim_method
                    config.TRIM_PERCENTILE_LOW = trim_percentile_low
                    config.TRIM_PERCENTILE_HIGH = trim_percentile_high
                    config.MIN_TRIM_SAMPLES = min_trim_samples
                    config.DATA_QUALITY_WEIGHT = data_quality_weight
                    config.MODEL_RELIABILITY_WEIGHT = model_reliability_weight

                    progress_bar.progress(10)
                    status_text.text("🔄 初始化预测器...")

                    predictor = PhysicsMLPredictor(config)

                    progress_bar.progress(30)
                    status_text.text("🔄 提取物理特征...")

                    predictor.train(
                        df,
                        ml_model_type=ml_model_type,
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=learning_rate if ml_model_type == 'gbdt' else None,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        n_folds=n_folds,
                    )

                    progress_bar.progress(90)
                    status_text.text("🔄 完成训练...")

                    st.session_state.predictor = predictor
                    st.session_state.model_trained = True
                    st.session_state.training_metrics = predictor.training_metrics
                    st.session_state.training_history = predictor.training_history
                    st.session_state.model_config = {
                        'min_accel': min_accel, 'min_speed': min_speed,
                        'min_force': min_force, 'roll_resist_coeff': roll_resist_coeff,
                        'secondary_window_size': secondary_window_size,
                        'min_secondary_points': min_secondary_points,
                        'min_primary_duration': min_primary_duration,
                        'stop_duration_trip': stop_duration_trip,
                        'trim_method': trim_method,
                        'ml_model_type': ml_model_type,
                    }

                    progress_bar.progress(100)
                    status_text.text("✅ 训练完成！")
                    st.success("✅ 模型训练完成！")
                    st.rerun()

                except Exception as e:
                    st.error(f"训练失败: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

    # ========== 预测分析 ==========
    elif page == "🔮 预测分析":
        st.markdown('<h2 class="section-header">数据预测与分析</h2>', unsafe_allow_html=True)

        if not st.session_state.model_trained:
            st.warning("⚠️ 请先训练模型")
            return

        predictor = st.session_state.predictor

        st.info(f"✅ 模型已加载 | 当前置信度阈值: {st.session_state.confidence_threshold:.0%}")

        if st.session_state.training_metrics:
            metrics = st.session_state.training_metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("模型MAE", f"{metrics.get('mae_kg', 0):.0f} kg")
            with col2:
                st.metric("模型MAPE", f"{metrics.get('mape', 0):.1f}%")
            with col3:
                st.metric("R²", f"{metrics.get('r2', 0):.3f}")

        if st.session_state.prediction_results is not None:
            st.success("📊 有之前的预测结果")
            results = st.session_state.prediction_results

            st.markdown("### 📋 预测摘要")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("总数据点", f"{results.get('total_points', 0):,}")
            with col2:
                valid_points = results.get('valid_points', 0)
                total_points = results.get('total_points', 1)
                st.metric("有效预测点", f"{valid_points:,} ({valid_points/total_points*100:.1f}%)")
            with col3:
                if 'trip_results' in results:
                    st.metric("有效行程数", f"{len(results['trip_results'])}")

            st.markdown("### 📈 质量时间曲线")
            if 'predictions_df' in results:
                image_buf = generate_mass_time_curve(results['predictions_df'])
                if image_buf is not None:
                    st.image(image_buf, caption="质量预测时间曲线", use_column_width=True)

            st.markdown("### 📊 置信度分析")
            if 'trip_results' in results:
                fig_conf = plot_confidence_distribution(results['trip_results'])
                if fig_conf:
                    st.plotly_chart(fig_conf, use_container_width=True)

            st.markdown("### 📊 行程汇总")
            if 'trip_results' in results:
                fig_trip = plot_trip_summary(results['trip_results'])
                if fig_trip:
                    st.plotly_chart(fig_trip, use_container_width=True)

            # 修改 streamlit_app.py 中的行程详情显示部分

            # 在预测结果显示部分，将原来的
            results = st.session_state.prediction_results

            # 行程详情部分改为同时显示区间和行程
            st.markdown("### 📊 区间汇总（主要输出）")
            if 'section_results' in results:
                section_detail = []
                for s in results['section_results']:
                    section_detail.append({
                        '区间ID': s['section_idx'],
                        '区间质量(kg)': f"{s['ml_mass']:.0f}",
                        '置信度': f"{s['confidence']:.2f}",
                        '包含行程数': s['n_trips'],
                        '总窗口数': s['n_total_windows'],
                        '开始时间': f"{s['start_time']:.0f}s",
                        '结束时间': f"{s['end_time']:.0f}s",
                    })
                st.dataframe(pd.DataFrame(section_detail), use_container_width=True)

            st.markdown("### 📋 行程详情（辅助参考）")
            if 'trip_results' in results:
                trip_detail = []
                for t in results['trip_results']:
                    trip_detail.append({
                        '行程ID': t['trip_idx'],
                        '所属区间': t.get('section_id', 'N/A'),
                        '质量(kg)': f"{t.get('ml_mass', t.get('mass', 0)):.0f}",
                        '置信度': f"{t['confidence']:.2f}",
                        '有效窗口': t['n_windows'],
                        '开始时间': f"{t['start_time']:.0f}s",
                    })
                st.dataframe(pd.DataFrame(trip_detail), use_container_width=True)

            st.markdown("### 💾 下载预测结果")
            if 'predictions_df' in results:
                csv = results['predictions_df'].to_csv(index=False).encode('utf-8-sig')
                st.download_button(label="下载CSV格式结果", data=csv,
                                 file_name="prediction_results.csv", mime="text/csv")

                # 添加对比分析部分
                st.markdown("### 🔬 物理估计 vs 🤖 ML校准 对比分析")

                # 新增：对比曲线图
                st.markdown("#### 📈 质量时间曲线对比")
                comparison_buf = generate_mass_time_curve_with_comparison(results['predictions_df'])
                if comparison_buf is not None:
                    st.image(comparison_buf, caption="物理估计(蓝) vs ML校准(绿) 对比", use_column_width=True)

                # 新增：散点对比图
                from visualization import plot_physical_vs_ml_comparison, plot_error_decomposition

                if 'trip_results' in results:
                    fig_comparison = plot_physical_vs_ml_comparison(results['trip_results'])
                    if fig_comparison:
                        st.plotly_chart(fig_comparison, use_container_width=True)

                    # 新增：误差分解图
                    st.markdown("### 📊 ML改进效果分析")
                    fig_decomposition = plot_error_decomposition(results['trip_results'])
                    if fig_decomposition:
                        st.plotly_chart(fig_decomposition, use_container_width=True)

                # 新增：行程对比表格
                st.markdown("### 📋 行程对比详情")
                if 'trip_results' in results:
                    comparison_detail = []
                    for t in results['trip_results']:
                        comparison_detail.append({
                            '行程ID': t['trip_idx'],
                            '物理估计质量(kg)': f"{t['physical_mass']:.0f}",
                            'ML校准质量(kg)': f"{t['ml_mass']:.0f}",
                            '差异(kg)': f"{abs(t['physical_mass'] - t['ml_mass']):.0f}",
                            '改进幅度(%)': f"{t['improvement_pct']:.1f}%",
                            '置信度': f"{t['confidence']:.2f}",
                            '有效窗口数': t['n_windows'],
                            '物理估计标准差': f"{t.get('physical_std', 0):.0f}",
                            'ML校准标准差': f"{t.get('ml_std', 0):.0f}",
                        })
                    st.dataframe(pd.DataFrame(comparison_detail), use_container_width=True)

                    # 显示汇总统计
                    total_improvement = np.mean([t['improvement_pct'] for t in results['trip_results']])
                    st.info(f"📊 **汇总**: ML校准平均改进了 {total_improvement:.1f}% 的预测精度")

                    if total_improvement < 5:
                        st.info("💡 **分析**: 改进幅度较小，说明物理模型已经比较准确，数据质量可能是主要限制因素")
                    elif total_improvement < 15:
                        st.info("💡 **分析**: 中等改进，ML模型有效学习了工况修正规律")
                    else:
                        st.success("💡 **分析**: 显著改进！ML模型大幅提升了预测精度，源数据噪声较大但模型有效去噪")

            if st.button("进行新的预测"):
                st.session_state.prediction_results = None
                st.rerun()
            return



        st.markdown("### 📤 上传预测数据")
        pred_file = st.file_uploader("上传需要预测的数据文件",
                                     type=['csv', 'xlsx', 'xls'],
                                     key="prediction_file")

        if pred_file is not None:
            with st.spinner("正在加载预测数据..."):
                pred_df = load_data(pred_file)
                if pred_df is not None:
                    st.success(f"✅ 预测数据加载成功！共 {len(pred_df)} 行")
                    with st.expander("预测数据预览", expanded=False):
                        st.dataframe(pred_df.head(10))

                    # 在 streamlit_app.py 的预测部分，找到并修改：
                    if st.button("开始预测", type="primary"):
                        with st.spinner("正在执行预测..."):
                            try:
                                # 确保使用带对比的预测方法
                                results = predictor.predict_with_comparison(  # 注意这里的方法名
                                    pred_df,
                                    confidence_threshold=st.session_state.confidence_threshold
                                )
                                st.session_state.prediction_results = results
                                st.session_state.prediction_file_name = pred_file.name
                                st.success("✅ 预测完成！")
                                st.rerun()
                            except Exception as e:
                                st.error(f"预测失败: {str(e)}")
                                import traceback
                                st.code(traceback.format_exc())



    # ========== 模型管理 ==========
    elif page == "💾 模型管理":
        st.markdown('<h2 class="section-header">模型管理与部署</h2>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📤 上传已有模型")
            uploaded_model = st.file_uploader("上传训练好的模型文件 (.pkl)",
                                             type=['pkl'], key="model_upload")
            if uploaded_model is not None:
                try:
                    with st.spinner("正在加载模型..."):
                        predictor = PhysicsMLPredictor()
                        with open("temp_model.pkl", "wb") as f:
                            f.write(uploaded_model.getbuffer())
                        predictor.load_model("temp_model.pkl")
                        st.session_state.predictor = predictor
                        st.session_state.model_trained = True
                        st.session_state.training_metrics = predictor.training_metrics
                        st.session_state.training_history = predictor.training_history
                        st.success("✅ 模型加载成功！")

                        st.markdown("#### 模型信息")
                        if predictor.training_metrics:
                            metrics = predictor.training_metrics
                            st.write(f"- 训练样本数: {metrics.get('n_samples', 'N/A')}")
                            st.write(f"- 训练行程数: {metrics.get('n_trips', 'N/A')}")
                            st.write(f"- MAE: {metrics.get('mae_kg', 0):.0f} kg")
                            st.write(f"- MAPE: {metrics.get('mape', 0):.1f}%")
                            st.write(f"- R²: {metrics.get('r2', 0):.3f}")

                except Exception as e:
                    st.error(f"模型加载失败: {str(e)}")

        with col2:
            st.markdown("### 📥 下载当前模型")
            if st.session_state.model_trained:
                model_name = st.text_input("保存模型名称", value="physics_ml_model")
                if st.button("生成模型文件"):
                    filename = f"{model_name}.pkl"
                    st.session_state.predictor.save_model(filename)
                    with open(filename, "rb") as f:
                        bytes_data = f.read()
                    st.download_button(label="下载模型文件", data=bytes_data,
                                     file_name=filename, mime="application/octet-stream")
            else:
                st.info("暂无已训练的模型")

        st.markdown("### 📊 系统状态概览")
        status_cols = st.columns(4)
        with status_cols[0]:
            st.write("✅ 数据已加载" if st.session_state.training_data is not None else "⚠️ 无数据")
        with status_cols[1]:
            st.write("✅ 模型已训练" if st.session_state.model_trained else "⚠️ 未训练")
        with status_cols[2]:
            st.write("✅ 有预测结果" if st.session_state.prediction_results is not None else "📊 无预测")
        with status_cols[3]:
            st.write("✅ 就绪" if st.session_state.model_trained else "⚙️ 待训练")


if __name__ == "__main__":
    main()