"""
车辆总质量预测系统 - Streamlit可视化界面
基于物理估计 F=ma + ML残差校准 (m_final = m_est + Δm)
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
from physics_ml_classifier import Config, PhysicsMLPredictor
from visualization import (
    plot_residual_analysis,
    plot_feature_importance,
    plot_window_comparison,
    plot_mass_time_curve_interactive,
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
    st.markdown('<p style="text-align: center; color: #6B7280;">物理估计 F=ma + ML残差校准 | m_final = m_est + Δm</p>',
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
                <p>3秒固定窗口 F=ma</p>
                <p>行程内窗口截尾取中位数</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>🤖 ML残差校准层</h3>
                <p>学习物理估计偏差 Δm</p>
                <p>m_final = m_est + Δm</p>
                <p>工况特征驱动修正</p>
                <p>修正幅度上限 ±30%</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>🎯 融合预测</h3>
                <p>区间600s / 行程60s</p>
                <p>窗口→行程→区间中位数</p>
                <p>全窗口参与预测</p>
                <p>交互式可视化分析</p>
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
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("ML MAE", f"{metrics.get('mae_kg', 0):.0f} kg")
            with col2:
                st.metric("物理基线 MAE", f"{metrics.get('physics_mae_kg', 0):.0f} kg")
            with col3:
                imp = metrics.get('improvement_over_physics_pct', 0)
                st.metric("相对物理改善", f"{imp:+.1f}%")
            with col4:
                st.metric("MAPE", f"{metrics.get('mape', 0):.1f}%")
            with col5:
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

            # 训练可视化（精简：残差分析 + 特征重要性 + 窗口对比）
            if st.session_state.training_history:
                st.markdown("### 📈 残差分析")
                fig_residual = plot_residual_analysis(st.session_state.training_history)
                if fig_residual:
                    st.plotly_chart(fig_residual, width='stretch')

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 🔝 特征重要性")
                    fig_importance = plot_feature_importance(st.session_state.training_history)
                    if fig_importance:
                        st.plotly_chart(fig_importance, width='stretch')

                with col2:
                    if st.session_state.training_history.get('residuals') is not None:
                        residuals = st.session_state.training_history['residuals']
                        abs_residuals = np.abs(residuals)
                        st.markdown("#### 误差统计")
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("平均绝对误差", f"{np.mean(abs_residuals):.0f} kg")
                            st.metric("中位数绝对误差", f"{np.median(abs_residuals):.0f} kg")
                        with col_b:
                            st.metric("最大误差", f"{np.max(abs_residuals):.0f} kg")
                            within_50 = (abs_residuals < 50).sum()
                            st.metric("误差<50kg占比", f"{within_50 / len(abs_residuals) * 100:.1f}%")

                st.markdown("### 🔍 行程级别质量对比与误差分析")
                fig_window_comparison = plot_window_comparison(st.session_state.training_history)
                if fig_window_comparison:
                    st.plotly_chart(fig_window_comparison, width='stretch')

            st.markdown("---")
            if st.button("🔄 重新训练模型", type="secondary"):
                st.session_state.model_trained = False
                st.session_state.training_metrics = None
                st.session_state.training_history = None
                st.rerun()
            st.markdown("---")

        # 训练配置表单
        st.markdown("### ⚙️ 模型配置")

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
                st.markdown("#### 层级划分（区间 → 行程 → 窗口）")
                col1, col2 = st.columns(2)
                with col1:
                    stop_duration_section = st.number_input(
                        "区间结束停车时长（秒）", 120, 7200, 600, 60,
                        help="停车超过此时间视为装卸货，结束当前装载区间，开始新区间。默认 600s"
                    )
                    stop_duration_trip = st.number_input(
                        "行程结束停车时长（秒）", 10, 600, 60, 10,
                        help="区间内停车超过此时间视为新行程（等灯/休息等）。应远小于区间时长"
                    )
                with col2:
                    window_size = st.number_input(
                        "窗口大小（秒）", 1.0, 10.0, 3.0, 0.5,
                        help="行程内按此时长切窗，每窗直接 F=ma"
                    )
                    min_window_points = st.number_input(
                        "窗口最少有效点数", 5, 50, 15, 5,
                        help="窗口内通过筛选的有效点少于此数则跳过"
                    )
                if stop_duration_trip >= stop_duration_section:
                    st.warning("行程停车时长应小于区间停车时长，否则无法在同一区间内划分多个行程。")

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
                    accel_heavy = st.number_input("重载阈值 (m/s²)", 0.3, 1.5, 1.0, 0.1,
                                                  help="质量>重载阈值时使用")
                    heavy_mass = st.number_input("重载质量阈值 (kg)", 2000, 5000, 3000, 100,
                                                 help="高于此质量使用重载加速度阈值")
                with col3:
                    accel_transition = st.number_input("过渡区阈值 (m/s²)", 0.4, 1.5, 1.0, 0.1,
                                                       help="质量在轻载和重载之间时使用")

                st.info(f"轻载(<{light_mass}kg): Acc>{accel_light} | "
                        f"重载(>{heavy_mass}kg): Acc>{accel_heavy} | "
                        f"过渡区: Acc>{accel_transition}")

            with tab4:
                st.markdown(
                    "**层级：区间 → 行程 → 窗口(F=ma) → ML校准**。"
                    "行程内按固定时长切窗，每窗对有效点直接算 F=ma；"
                    "**截尾仅在行程级**对窗口质量做百分位过滤后取中位数。"
                )
                col1, col2 = st.columns(2)
                with col1:
                    trim_remove_bottom = st.slider(
                        "删除下侧数据点 (%)", 0, 49, 5, 5,
                        help="行程级截尾：去掉该行程内质量最低的前 X% 窗口"
                    )
                with col2:
                    trim_remove_top = st.slider(
                        "删除上侧数据点 (%)", 0, 49, 5, 5,
                        help="行程级截尾：去掉该行程内质量最高的后 X% 窗口"
                    )
                min_trim_samples = st.slider(
                    "截尾后至少保留窗口数", 1, 10, 3, 1,
                    help="截尾后若剩余窗口少于此数，则不截尾，使用全部窗口"
                )
                if trim_remove_bottom + trim_remove_top >= 50:
                    st.warning("下侧+上侧删除比例过大，可能导致有效窗口过少，建议合计不超过 40%。")

            with tab5:
                st.markdown("#### 模型类型")
                ml_model_type = st.selectbox("ML模型类型",
                                             ['gbdt', 'rf', 'linear'],
                                             format_func=lambda x: {
                                                 'gbdt': 'GBDT (推荐)',
                                                 'rf': '随机森林',
                                                 'linear': '线性回归'
                                             }[x])
                st.markdown("#### 超参数设置")
                col_a, col_b = st.columns(2)
                with col_a:
                    n_estimators = st.slider("树的数量 (n_estimators)", 50, 500, 200, 10)
                    max_depth = st.slider("最大深度 (max_depth)", 2, 15, 4, 1)
                    learning_rate = st.slider("学习率 (learning_rate)", 0.01, 0.3, 0.05, 0.01)
                with col_b:
                    min_samples_split = st.slider("最小分割样本 (min_samples_split)", 2, 20, 5, 1)
                    min_samples_leaf = st.slider("最小叶子样本 (min_samples_leaf)", 1, 10, 2, 1)
                    n_folds = st.slider("交叉验证折数 (n_folds)", 3, 10, 5, 1)
                if ml_model_type == 'linear':
                    st.info("线性回归不使用树相关参数，仅 n_folds 生效。")
                else:
                    st.info("GBDT/RF 学习物理估计残差 Δm，最终质量 = m_est + Δm")

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
                    config.WINDOW_SIZE = window_size
                    config.SECONDARY_WINDOW_SIZE = window_size
                    config.MIN_WINDOW_POINTS = min_window_points
                    config.MIN_SECONDARY_WINDOW_POINTS = min_window_points
                    config.MIN_STOP_DURATION_FOR_SECTION_END = stop_duration_section
                    config.MIN_STOP_DURATION_FOR_TRIP_END = stop_duration_trip
                    config.TRIM_REMOVE_BOTTOM_PCT = trim_remove_bottom
                    config.TRIM_REMOVE_TOP_PCT = trim_remove_top
                    config.MIN_TRIM_SAMPLES = min_trim_samples

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
                        learning_rate=learning_rate,
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
                        'window_size': window_size,
                        'min_window_points': min_window_points,
                        'stop_duration_section': stop_duration_section,
                        'stop_duration_trip': stop_duration_trip,
                        'trim_remove_bottom': trim_remove_bottom,
                        'trim_remove_top': trim_remove_top,
                        'ml_model_type': ml_model_type,
                        'n_estimators': n_estimators,
                        'max_depth': max_depth,
                        'learning_rate': learning_rate,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf,
                        'n_folds': n_folds,
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

        st.info("✅ 模型已加载 | 所有有效窗口均参与预测聚合")

        if st.session_state.training_metrics:
            metrics = st.session_state.training_metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("ML MAE", f"{metrics.get('mae_kg', 0):.0f} kg")
            with col2:
                st.metric("物理基线 MAE", f"{metrics.get('physics_mae_kg', 0):.0f} kg")
            with col3:
                st.metric("MAPE", f"{metrics.get('mape', 0):.1f}%")
            with col4:
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

            st.markdown("### 📈 质量预测时间曲线")
            if 'predictions_df' in results:
                fig_curve = plot_mass_time_curve_interactive(
                    results['predictions_df'],
                    trip_results=results.get('trip_results'),
                    section_results=results.get('section_results'),
                )
                if fig_curve is not None:
                    st.plotly_chart(fig_curve, width='stretch', key='mass_time_curve_plotly')
                else:
                    st.warning("无法生成质量时间曲线，请确认预测结果中包含 time_seconds 列。")

            # 行程详情部分
            st.markdown("### 📊 区间汇总（主要输出）")
            if 'section_results' in results:
                section_detail = []
                for s in results['section_results']:
                    section_detail.append({
                        '区间ID': s['section_idx'],
                        '区间质量(kg)': f"{s['ml_mass']:.0f}",
                        '包含行程数': s['n_trips'],
                        '总窗口数': s['n_total_windows'],
                        '开始时间': f"{s['start_time']:.0f}s",
                        '结束时间': f"{s['end_time']:.0f}s",
                    })
                st.dataframe(pd.DataFrame(section_detail), width='stretch')

            st.markdown("### 📋 行程对比详情")
            if 'trip_results' in results:
                comparison_detail = []
                for t in results['trip_results']:
                    comparison_detail.append({
                        '行程ID': t['trip_idx'],
                        '所属区间': t.get('section_id', 'N/A'),
                        '物理估计(kg)': f"{t['physical_mass']:.0f}",
                        'ML校准(kg)': f"{t['ml_mass']:.0f}",
                        '差异(kg)': f"{abs(t['physical_mass'] - t['ml_mass']):.0f}",
                        '窗口总数': t.get('n_primary_windows', '—'),
                    })
                st.dataframe(pd.DataFrame(comparison_detail), width='stretch')

            st.markdown("### 💾 下载预测结果")
            if 'predictions_df' in results:
                csv = results['predictions_df'].to_csv(index=False).encode('utf-8-sig')
                st.download_button(label="下载CSV格式结果", data=csv,
                                 file_name="prediction_results.csv", mime="text/csv")

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
                                results = predictor.predict_with_comparison(pred_df)
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