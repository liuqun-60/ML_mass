"""
可视化函数库
负责所有图表生成
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import io
from typing import Dict, List, Optional, Any

# 解决 matplotlib 中文显示问题
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import io
from typing import Dict, List, Optional, Any


def plot_residual_analysis(training_history: Dict) -> Optional[go.Figure]:
    """绘制残差分析图"""
    if training_history is None or training_history.get('y') is None:
        return None

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('残差分布', '预测值 vs 实际值',
                        '残差 vs 预测值', 'Q-Q图'),
    )

    y_true = training_history['y']
    y_pred = training_history['y_pred']
    residuals = y_true - y_pred

    # 残差分布直方图
    fig.add_trace(
        go.Histogram(x=residuals, nbinsx=50, name='残差',
                     marker_color='#3B82F6'),
        row=1, col=1
    )

    # 预测值 vs 实际值
    fig.add_trace(
        go.Scatter(x=y_pred, y=y_true, mode='markers',
                   marker=dict(size=6, opacity=0.5),
                   name='数据点'),
        row=1, col=2
    )
    min_val = min(y_pred.min(), y_true.min())
    max_val = max(y_pred.max(), y_true.max())
    fig.add_trace(
        go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                   mode='lines', line=dict(dash='dash', color='red'),
                   name='完美预测'),
        row=1, col=2
    )

    # 残差 vs 预测值
    fig.add_trace(
        go.Scatter(x=y_pred, y=residuals, mode='markers',
                   marker=dict(size=6, opacity=0.5),
                   name='残差'),
        row=2, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)

    # Q-Q图
    sorted_residuals = np.sort(residuals)
    theoretical_quantiles = stats.norm.ppf(
        (np.arange(len(sorted_residuals)) + 0.5) / len(sorted_residuals),
        loc=np.mean(sorted_residuals),
        scale=np.std(sorted_residuals)
    )
    fig.add_trace(
        go.Scatter(x=theoretical_quantiles, y=sorted_residuals,
                   mode='markers', marker=dict(size=6, opacity=0.5),
                   name='Q-Q'),
        row=2, col=2
    )
    max_qq = max(abs(theoretical_quantiles).max(), abs(sorted_residuals).max())
    fig.add_trace(
        go.Scatter(x=[-max_qq, max_qq], y=[-max_qq, max_qq],
                   mode='lines', line=dict(dash='dash', color='red'),
                   name='正态参考线'),
        row=2, col=2
    )

    fig.update_layout(height=800, showlegend=False,
                      title_text="残差分析")
    fig.update_xaxes(title_text="残差 (kg)", row=1, col=1)
    fig.update_xaxes(title_text="预测值 (kg)", row=1, col=2)
    fig.update_yaxes(title_text="实际值 (kg)", row=1, col=2)
    fig.update_xaxes(title_text="预测值 (kg)", row=2, col=1)
    fig.update_yaxes(title_text="残差 (kg)", row=2, col=1)
    fig.update_xaxes(title_text="理论分位数", row=2, col=2)
    fig.update_yaxes(title_text="实际分位数", row=2, col=2)

    return fig


def plot_feature_importance(training_history: Dict) -> Optional[go.Figure]:
    """绘制特征重要性图"""
    if training_history.get('feature_importance') is None:
        return None

    importance_df = training_history['feature_importance']

    fig = go.Figure(data=[
        go.Bar(
            x=importance_df['importance'],
            y=importance_df['feature'],
            orientation='h',
            marker_color='#3B82F6',
            text=importance_df['importance'].apply(lambda x: f'{x:.3f}'),
            textposition='outside',
        )
    ])

    fig.update_layout(
        title='特征重要性排名',
        xaxis_title='重要性',
        yaxis_title='特征',
        height=500,
        yaxis=dict(autorange="reversed"),
    )

    return fig


def plot_cv_results(training_history: Dict) -> Optional[go.Figure]:
    """绘制交叉验证结果"""
    if training_history.get('cv_predictions') is None:
        return None

    cv_preds = training_history['cv_predictions']
    cv_actuals = training_history['cv_actuals']
    groups = training_history['groups']

    fig = go.Figure()

    unique_groups = np.unique(groups)
    colors = px.colors.qualitative.Plotly[:len(unique_groups)]

    for i, group in enumerate(unique_groups):
        mask = groups == group
        fig.add_trace(
            go.Scatter(
                x=cv_actuals[mask],
                y=cv_preds[mask],
                mode='markers',
                marker=dict(size=8, color=colors[i % len(colors)]),
                name=f'行程 {group}'
            )
        )

    min_val = min(cv_preds.min(), cv_actuals.min())
    max_val = max(cv_preds.max(), cv_actuals.max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(dash='dash', color='red', width=2),
            name='完美预测'
        )
    )

    fig.update_layout(
        title='交叉验证：预测值 vs 实际值',
        xaxis_title='实际质量 (kg)',
        yaxis_title='CV预测质量 (kg)',
        height=500,
    )

    return fig


def plot_residual_by_region(training_history: Dict) -> Optional[go.Figure]:
    """绘制不同区域的残差分布"""
    if training_history.get('X') is None:
        return None

    X = training_history['X']
    residuals = training_history['residuals']
    m_est_values = X[:, 0]

    bins = [0, 1500, 2000, 2500, 3000, 3500, 4000, 5000]
    labels = ['<1500', '1500-2000', '2000-2500', '2500-3000',
              '3000-3500', '3500-4000', '4000+']

    bin_indices = np.digitize(m_est_values, bins)

    fig = go.Figure()

    for i, label in enumerate(labels):
        mask = bin_indices == i + 1
        if mask.sum() > 0:
            fig.add_trace(
                go.Box(
                    y=np.abs(residuals[mask]),
                    name=f'{label}\n(n={mask.sum()})',
                    boxmean='sd',
                )
            )

    fig.update_layout(
        title='不同质量区间的残差分布',
        yaxis_title='绝对残差 (kg)',
        height=500,
    )

    return fig


def plot_window_comparison(training_history: Dict) -> Optional[go.Figure]:
    """绘制每个窗口的物理估计、ML校准与实际质量的对比"""
    if training_history is None or training_history.get('X') is None:
        return None

    X = training_history['X']
    y_true = training_history['y']
    y_pred = training_history['y_pred']
    groups = training_history['groups']
    residuals = training_history['residuals']

    m_est_values = X[:, 0]
    n_secondary_values = X[:, 2]
    m_est_cv_values = X[:, 1]
    accel_values = X[:, 3]

    window_df = pd.DataFrame({
        'window_id': np.arange(len(y_true)),
        'trip_id': groups,
        'm_physical': m_est_values,
        'm_calibrated': y_pred,
        'm_true': y_true,
        'residual': residuals,
        'abs_residual': np.abs(residuals),
        'n_secondary': n_secondary_values,
        'm_est_cv': m_est_cv_values,
        'accel_mean': accel_values,
    })

    window_df['trip_window_idx'] = window_df.groupby('trip_id').cumcount()

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            '所有窗口：物理估计 vs ML校准 vs 真实质量',
            '按行程：窗口质量对比',
            '误差 vs 物理估计值',
            '误差 vs 窗口内一致性 (m_est_cv)',
            '误差 vs 二级窗口数量',
            '误差 vs 加速度均值'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
    )

    unique_trips = np.unique(groups)
    colors = px.colors.qualitative.Plotly[:len(unique_trips)]
    trip_color_map = {trip: colors[i % len(colors)] for i, trip in enumerate(unique_trips)}

    def hex_to_rgba(hex_color, alpha=0.3):
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 6:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
        else:
            r, g, b = 0, 0, 0
        return f'rgba({r},{g},{b},{alpha})'

    # 子图1：所有窗口的对比
    sorted_idx = np.argsort(window_df['trip_id'].values * 1000 + window_df['trip_window_idx'].values)

    fig.add_trace(
        go.Scatter(x=np.arange(len(y_true)), y=m_est_values[sorted_idx],
                   mode='markers', marker=dict(size=4, opacity=0.5, color='blue'),
                   name='物理估计', legendgroup='physical'),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=np.arange(len(y_true)), y=y_pred[sorted_idx],
                   mode='markers', marker=dict(size=4, opacity=0.5, color='green'),
                   name='ML校准', legendgroup='calibrated'),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=np.arange(len(y_true)), y=y_true[sorted_idx],
                   mode='markers', marker=dict(size=4, opacity=0.5, color='red', symbol='x'),
                   name='真实质量', legendgroup='true'),
        row=1, col=1
    )

    # 子图2：按行程的窗口对比
    for trip_id in unique_trips:
        trip_mask = window_df['trip_id'] == trip_id
        trip_data = window_df[trip_mask].sort_values('trip_window_idx')

        color = trip_color_map[trip_id]
        rgba_color_light = hex_to_rgba(color, 0.2)
        rgba_color_medium = hex_to_rgba(color, 0.5)

        fig.add_trace(
            go.Scatter(x=trip_data['trip_window_idx'], y=trip_data['m_physical'],
                       mode='lines+markers', marker=dict(size=4, color=rgba_color_medium),
                       line=dict(dash='dot', width=1, color=rgba_color_medium),
                       name=f'行程{trip_id}物理', legendgroup=f'trip{trip_id}', showlegend=True),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(x=trip_data['trip_window_idx'], y=trip_data['m_calibrated'],
                       mode='lines+markers', marker=dict(size=4, color=color),
                       line=dict(dash='solid', width=1.5, color=color),
                       name=f'行程{trip_id}ML', legendgroup=f'trip{trip_id}', showlegend=True),
            row=1, col=2
        )

        true_mass = trip_data['m_true'].iloc[0]
        fig.add_hline(y=true_mass, line=dict(dash='solid', width=2, color=rgba_color_light), row=1, col=2)

    # 子图3：误差 vs 物理估计值
    fig.add_trace(
        go.Scatter(x=m_est_values, y=np.abs(residuals), mode='markers',
                   marker=dict(size=8, color=accel_values, colorscale='Viridis',
                               colorbar=dict(title='加速度(m/s²)'), opacity=0.6),
                   text=[f'窗口{i}<br>物理:{m_est_values[i]:.0f}kg<br>误差:{abs(residuals[i]):.0f}kg'
                         for i in range(len(m_est_values))],
                   hoverinfo='text', name='窗口误差'),
        row=2, col=1
    )
    fig.add_hline(y=np.mean(np.abs(residuals)), line=dict(dash='dash', color='rgba(255,0,0,0.7)'),
                  annotation_text=f'平均误差:{np.mean(np.abs(residuals)):.0f}kg', row=2, col=1)

    # 子图4：误差 vs m_est_cv
    fig.add_trace(
        go.Scatter(x=m_est_cv_values, y=np.abs(residuals), mode='markers',
                   marker=dict(size=8, color=n_secondary_values, colorscale='RdYlBu',
                               colorbar=dict(title='二级窗口数'), opacity=0.6),
                   text=[f'窗口{i}<br>CV:{m_est_cv_values[i]:.3f}<br>误差:{abs(residuals[i]):.0f}kg'
                         for i in range(len(m_est_cv_values))],
                   hoverinfo='text', name='窗口误差'),
        row=2, col=2
    )

    # 子图5：误差 vs 二级窗口数量
    fig.add_trace(
        go.Box(x=n_secondary_values, y=np.abs(residuals),
               name='误差分布', marker_color='#3B82F6', boxmean='sd'),
        row=3, col=1
    )

    # 子图6：误差 vs 加速度均值
    fig.add_trace(
        go.Scatter(x=accel_values, y=np.abs(residuals), mode='markers',
                   marker=dict(size=10, color=m_est_cv_values, colorscale='RdYlGn_r',
                               colorbar=dict(title='m_est_cv'), opacity=0.6),
                   text=[f'窗口{i}<br>加速度:{accel_values[i]:.2f}<br>误差:{abs(residuals[i]):.0f}kg'
                         for i in range(len(accel_values))],
                   hoverinfo='text', name='窗口误差'),
        row=3, col=2
    )

    fig.update_layout(height=1200, title_text="窗口级别质量对比与误差分析", hovermode='closest')
    fig.update_xaxes(title_text="窗口序号", row=1, col=1)
    fig.update_yaxes(title_text="质量 (kg)", row=1, col=1)
    fig.update_xaxes(title_text="行程内窗口序号", row=1, col=2)
    fig.update_yaxes(title_text="质量 (kg)", row=1, col=2)
    fig.update_xaxes(title_text="物理估计值 (kg)", row=2, col=1)
    fig.update_yaxes(title_text="绝对误差 (kg)", row=2, col=1)
    fig.update_xaxes(title_text="m_est_cv", row=2, col=2)
    fig.update_yaxes(title_text="绝对误差 (kg)", row=2, col=2)
    fig.update_xaxes(title_text="二级窗口数量", row=3, col=1)
    fig.update_yaxes(title_text="绝对误差 (kg)", row=3, col=1)
    fig.update_xaxes(title_text="加速度均值 (m/s²)", row=3, col=2)
    fig.update_yaxes(title_text="绝对误差 (kg)", row=3, col=2)

    return fig


# 替换 visualization.py 中的 plot_confidence_distribution 函数

def plot_confidence_distribution(trip_results: List[Dict]) -> Optional[go.Figure]:
    """绘制置信度分布"""
    if not trip_results:
        return None

    # 兼容新旧数据结构：优先使用 ml_mass，如果没有则使用 mass
    if 'ml_mass' in trip_results[0]:
        confidences = [t['confidence'] for t in trip_results]
        masses = [t['ml_mass'] for t in trip_results]
        n_windows = [t['n_windows'] for t in trip_results]
        hover_texts = [f"行程{t['trip_idx']}<br>ML质量:{t['ml_mass']:.0f}kg<br>置信度:{t['confidence']:.2f}<br>窗口数:{t['n_windows']}"
                       for t in trip_results]
    else:
        confidences = [t['confidence'] for t in trip_results]
        masses = [t.get('mass', t.get('ml_mass', 0)) for t in trip_results]
        n_windows = [t['n_windows'] for t in trip_results]
        hover_texts = [f"行程{t['trip_idx']}<br>质量:{t.get('mass', t.get('ml_mass', 0)):.0f}kg<br>置信度:{t['confidence']:.2f}<br>窗口数:{t['n_windows']}"
                       for t in trip_results]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('置信度分布', '置信度 vs 窗口数'),
        specs=[[{'type': 'histogram'}, {'type': 'scatter'}]]
    )

    fig.add_trace(
        go.Histogram(x=confidences, nbinsx=20, marker_color='#3B82F6'),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=n_windows, y=confidences, mode='markers',
                   marker=dict(size=10, color=masses, colorscale='Viridis',
                               showscale=True, colorbar=dict(title='质量(kg)')),
                   text=hover_texts,
                   hoverinfo='text'),
        row=1, col=2
    )

    fig.update_layout(height=400, showlegend=False)
    fig.update_xaxes(title_text="置信度", row=1, col=1)
    fig.update_xaxes(title_text="有效窗口数", row=1, col=2)
    fig.update_yaxes(title_text="置信度", row=1, col=2)

    return fig


# 替换 visualization.py 中的 plot_trip_summary 函数

def plot_trip_summary(trip_results: List[Dict]) -> Optional[go.Figure]:
    """绘制行程汇总图"""
    if not trip_results:
        return None

    # 兼容新旧数据结构
    if 'ml_mass' in trip_results[0]:
        # 新数据结构
        trip_labels = [f"行程{t['trip_idx']}" for t in trip_results]
        mass_values = [t['ml_mass'] for t in trip_results]
        confidences = [t['confidence'] for t in trip_results]
        n_windows = [t['n_windows'] for t in trip_results]
        hover_texts = [f"行程{t['trip_idx']}<br>窗口数:{t['n_windows']}" for t in trip_results]
    else:
        # 旧数据结构兼容
        trip_labels = [f"行程{t['trip_idx']}" for t in trip_results]
        mass_values = [t.get('mass', t.get('ml_mass', 0)) for t in trip_results]
        confidences = [t['confidence'] for t in trip_results]
        n_windows = [t['n_windows'] for t in trip_results]
        hover_texts = [f"行程{t['trip_idx']}<br>窗口数:{t['n_windows']}" for t in trip_results]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('行程质量估计', '窗口数 vs 置信度',
                        '质量分布箱线图', '置信度 vs 质量'),
    )

    fig.add_trace(
        go.Bar(x=trip_labels, y=mass_values,
               marker_color=[f'rgba(59, 130, 246, {c})' for c in confidences],
               text=[f'{m:.0f}kg' for m in mass_values],
               textposition='outside'),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=n_windows, y=confidences,
                   mode='markers+text', marker=dict(size=12),
                   text=trip_labels,
                   textposition='top center'),
        row=1, col=2
    )

    fig.add_trace(
        go.Box(y=mass_values, name='质量分布', boxmean='sd'),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(x=mass_values, y=confidences,
                   mode='markers',
                   marker=dict(size=[w * 5 for w in n_windows],
                               color=mass_values,
                               colorscale='Viridis', showscale=True),
                   text=hover_texts,
                   hoverinfo='text'),
        row=2, col=2
    )

    fig.update_layout(height=700, showlegend=False, title_text="行程汇总分析")
    fig.update_xaxes(title_text="行程", row=1, col=1)
    fig.update_yaxes(title_text="质量 (kg)", row=1, col=1)
    fig.update_xaxes(title_text="有效窗口数", row=1, col=2)
    fig.update_yaxes(title_text="置信度", row=1, col=2)
    fig.update_yaxes(title_text="质量 (kg)", row=2, col=1)
    fig.update_xaxes(title_text="质量 (kg)", row=2, col=2)
    fig.update_yaxes(title_text="置信度", row=2, col=2)

    return fig


def generate_mass_time_curve(predictions_df: pd.DataFrame) -> Optional[io.BytesIO]:
    """生成质量时间曲线（matplotlib版本，用于Streamlit展示）"""
    try:
        if 'time_seconds' not in predictions_df.columns or 'predicted_mass' not in predictions_df.columns:
            return None

        total_points = len(predictions_df)
        if total_points > 50000:
            step = max(1, total_points // 50000)
            plot_df = predictions_df.iloc[::step].copy()
        else:
            plot_df = predictions_df.copy()

        fig, ax = plt.subplots(figsize=(15, 6))

        valid_mask = plot_df['is_valid_prediction'] if 'is_valid_prediction' in plot_df.columns else plot_df[
            'predicted_mass'].notna()

        if valid_mask.any():
            valid_data = plot_df[valid_mask]
            time_valid = valid_data['time_seconds'].values
            mass_valid = valid_data['predicted_mass'].values

            if 'prediction_confidence' in valid_data.columns:
                confidences = valid_data['prediction_confidence'].values
                scatter = ax.scatter(time_valid, mass_valid, s=15, alpha=0.8,
                                     c=confidences, cmap='RdYlGn', vmin=0, vmax=1,
                                     edgecolors='none')
                plt.colorbar(scatter, ax=ax, label='置信度')
            else:
                ax.scatter(time_valid, mass_valid, s=15, alpha=0.6,
                           c='#3B82F6', edgecolors='none')

        if 'is_valid_prediction' in plot_df.columns:
            invalid_data = plot_df[~valid_mask & plot_df['predicted_mass'].notna()]
            if len(invalid_data) > 0:
                ax.scatter(invalid_data['time_seconds'].values,
                           invalid_data['predicted_mass'].values,
                           s=5, alpha=0.3, c='gray', edgecolors='none',
                           label='低置信度（继承）')

        ax.set_xlabel('时间 (秒)', fontsize=12)
        ax.set_ylabel('预测质量 (kg)', fontsize=12)
        ax.set_title('质量预测时间曲线', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right')

        stats_text = f"总点数: {total_points:,}\n"
        stats_text += f"有效预测点: {valid_mask.sum():,} ({valid_mask.sum() / total_points * 100:.1f}%)\n"
        if valid_mask.any():
            mass_median = plot_df.loc[valid_mask, 'predicted_mass'].median()
            stats_text += f"中位数质量: {mass_median:.0f} kg"

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return buf
    except Exception as e:
        print(f"生成质量时间曲线失败: {str(e)}")
        return None


def plot_trim_effect(training_history: Dict) -> Optional[go.Figure]:
    """绘制截尾效果对比图"""
    if training_history is None or training_history.get('features_df') is None:
        return None

    features_df = training_history['features_df']

    if 'n_secondary' not in features_df.columns:
        return None

    # 模拟截尾效果（基于CV值）
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('窗口质量CV分布', '截尾效果示意'),
    )

    # CV分布
    if 'm_est_cv' in features_df.columns:
        fig.add_trace(
            go.Histogram(x=features_df['m_est_cv'], nbinsx=50,
                         marker_color='#3B82F6', name='m_est_cv分布'),
            row=1, col=1
        )

    # 截尾效果：显示CV与二级窗口数的关系
    if 'n_secondary' in features_df.columns and 'm_est_cv' in features_df.columns:
        fig.add_trace(
            go.Scatter(x=features_df['n_secondary'], y=features_df['m_est_cv'],
                       mode='markers', marker=dict(size=8, opacity=0.6),
                       text=[
                           f'CV:{features_df["m_est_cv"].iloc[i]:.3f}<br>窗口数:{features_df["n_secondary"].iloc[i]:.0f}'
                           for i in range(len(features_df))],
                       hoverinfo='text', name='窗口'),
            row=1, col=2
        )
        # 添加截尾阈值线
        fig.add_hline(y=0.15, line=dict(dash='dash', color='red'),
                      annotation_text='高截尾阈值(CV>0.15)', row=1, col=2)
        fig.add_hline(y=0.10, line=dict(dash='dash', color='orange'),
                      annotation_text='中截尾阈值(CV>0.10)', row=1, col=2)

    fig.update_layout(height=400, title_text="截尾效果分析")
    fig.update_xaxes(title_text="m_est_cv", row=1, col=1)
    fig.update_xaxes(title_text="二级窗口数", row=1, col=2)
    fig.update_yaxes(title_text="m_est_cv", row=1, col=2)

    return fig

# 添加到 visualization.py 末尾

def plot_physical_vs_ml_comparison(trip_results: List[Dict]) -> Optional[go.Figure]:
    """绘制物理估计 vs ML校准对比图 - 优化版本"""
    if not trip_results:
        return None

    # 提取数据
    physical_masses = [t['physical_mass'] for t in trip_results]
    ml_masses = [t['ml_mass'] for t in trip_results]
    confidences = [t['confidence'] for t in trip_results]
    improvements = [t['improvement_pct'] for t in trip_results]
    trip_ids = [f"行程{t['trip_idx']}" for t in trip_results]
    n_windows = [t['n_windows'] for t in trip_results]

    # 计算差异
    differences = [ml - phy for ml, phy in zip(ml_masses, physical_masses)]
    abs_differences = [abs(d) for d in differences]

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            '物理估计 vs ML校准质量',
            '质量差异分布 (ML - 物理)',
            '改进幅度分布',
            '行程级别质量对比',
            '改进幅度 vs 窗口数',
            '汇总统计'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
        specs=[
            [{'type': 'scatter'}, {'type': 'histogram'}, {'type': 'histogram'}],
            [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'table'}]
        ]
    )

    # ========== 子图1：物理估计 vs ML校准 散点图 ==========
    # 主要散点
    fig.add_trace(
        go.Scatter(
            x=physical_masses,
            y=ml_masses,
            mode='markers+text',
            marker=dict(
                size=18,
                color=differences,
                colorscale='RdBu_r',
                showscale=True,
                colorbar=dict(
                    title='差异(kg)',
                    x=0.32,
                    y=0.82,
                    len=0.35,
                    thickness=15
                ),
                line=dict(width=2, color='white')
            ),
            text=[f"行程{t['trip_idx']}" for t in trip_results],
            textposition='top center',
            textfont=dict(size=8),
            hovertext=[
                f"<b>行程{t['trip_idx']}</b><br>"
                f"物理估计: {t['physical_mass']:.0f} kg<br>"
                f"ML校准: {t['ml_mass']:.0f} kg<br>"
                f"差异: {differences[i]:.0f} kg<br>"
                f"改进幅度: {t['improvement_pct']:.1f}%<br>"
                f"置信度: {t['confidence']:.2f}<br>"
                f"有效窗口: {t['n_windows']}"
                for i, t in enumerate(trip_results)
            ],
            hoverinfo='text',
            name='行程对比'
        ),
        row=1, col=1
    )

    # 对角线（完美一致线）
    min_val = min(min(physical_masses), min(ml_masses)) - 100
    max_val = max(max(physical_masses), max(ml_masses)) + 100
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(dash='dash', color='gray', width=1.5),
            name='完全一致线',
            showlegend=False
        ),
        row=1, col=1
    )

    # ±5% 误差带
    x_range = np.linspace(min_val, max_val, 100)
    fig.add_trace(
        go.Scatter(
            x=list(x_range),
            y=list(x_range * 1.05),
            mode='lines',
            line=dict(dash='dot', color='lightgray', width=0.8),
            fill='tonexty',
            fillcolor='rgba(128,128,128,0.05)',
            name='+5%',
            showlegend=False
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=list(x_range),
            y=list(x_range * 0.95),
            mode='lines',
            line=dict(dash='dot', color='lightgray', width=0.8),
            fill='tonexty',
            fillcolor='rgba(128,128,128,0.05)',
            name='-5%',
            showlegend=False
        ),
        row=1, col=1
    )

    # ========== 子图2：质量差异分布直方图 ==========
    fig.add_trace(
        go.Histogram(
            x=differences,
            nbinsx=20,
            marker_color=['#EF4444' if d < 0 else '#10B981' for d in differences],
            name='差异分布',
            showlegend=False,
            hoverinfo='x+y'
        ),
        row=1, col=2
    )

    # 添加均值线
    mean_diff = np.mean(differences)
    fig.add_vline(
        x=mean_diff,
        line=dict(dash='solid', color='blue', width=2),
        annotation_text=f'均值: {mean_diff:.0f}',
        annotation_position='top',
        row=1, col=2
    )

    # ========== 子图3：改进幅度分布 ==========
    fig.add_trace(
        go.Histogram(
            x=improvements,
            nbinsx=20,
            marker_color='#3B82F6',
            name='改进幅度',
            showlegend=False,
            hoverinfo='x+y'
        ),
        row=1, col=3
    )

    avg_improvement = np.mean(improvements)
    fig.add_vline(
        x=avg_improvement,
        line=dict(dash='solid', color='red', width=2),
        annotation_text=f'均值: {avg_improvement:.1f}%',
        annotation_position='top',
        row=1, col=3
    )

    # ========== 子图4：行程级别质量对比（并排柱状图） ==========
    # 创建并排显示
    x_positions = np.arange(len(trip_results))
    width = 0.35

    fig.add_trace(
        go.Bar(
            x=[f"行程{t['trip_idx']}" for t in trip_results],
            y=physical_masses,
            name='物理估计',
            marker_color='#EF4444',
            opacity=0.8,
            hovertext=[f"物理估计: {p:.0f} kg" for p in physical_masses],
            hoverinfo='text',
            width=width
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Bar(
            x=[f"行程{t['trip_idx']}" for t in trip_results],
            y=ml_masses,
            name='ML校准',
            marker_color='#10B981',
            opacity=0.8,
            hovertext=[f"ML校准: {m:.0f} kg" for m in ml_masses],
            hoverinfo='text',
            width=width
        ),
        row=2, col=1
    )

    # ========== 子图5：改进幅度 vs 窗口数 ==========
    fig.add_trace(
        go.Scatter(
            x=n_windows,
            y=improvements,
            mode='markers',
            marker=dict(
                size=14,
                color=confidences,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(
                    title='置信度',
                    x=0.98,
                    y=0.18,
                    len=0.35,
                    thickness=15
                ),
                line=dict(width=1, color='black')
            ),
            text=trip_ids,
            hovertext=[
                f"<b>{trip_ids[i]}</b><br>"
                f"改进: {improvements[i]:.1f}%<br>"
                f"窗口数: {n_windows[i]}<br>"
                f"置信度: {confidences[i]:.2f}"
                for i in range(len(trip_results))
            ],
            hoverinfo='text',
            name='行程'
        ),
        row=2, col=2
    )

    # 添加趋势线
    if len(n_windows) > 1:
        z = np.polyfit(n_windows, improvements, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(n_windows), max(n_windows), 50)
        fig.add_trace(
            go.Scatter(
                x=x_trend,
                y=p(x_trend),
                mode='lines',
                line=dict(dash='dash', color='orange', width=2),
                name='趋势线',
                showlegend=False
            ),
            row=2, col=2
        )

    # ========== 子图6：汇总统计表格 ==========
    # 计算汇总统计
    stats_data = [
        ['指标', '数值', '说明'],
        ['行程总数', f'{len(trip_results)}', '有效识别的行程数'],
        ['平均物理估计', f'{np.mean(physical_masses):.0f} kg', 'F=ma直接计算结果'],
        ['平均ML校准', f'{np.mean(ml_masses):.0f} kg', '机器学习校准后结果'],
        ['平均差异', f'{np.mean(abs_differences):.0f} kg', '|ML - 物理|的平均值'],
        ['平均改进幅度', f'{avg_improvement:.1f}%', 'ML对物理估计的修正比例'],
        ['中位数改进', f'{np.median(improvements):.1f}%', '改进幅度的中位数'],
        ['ML降低波动',
         f'{(np.std(physical_masses) - np.std(ml_masses)) / max(np.std(physical_masses), 1) * 100:.1f}%',
         '标准差的降低比例'],
        ['高改进行程(>10%)', f'{sum(1 for i in improvements if i > 10)}', 'ML显著改善的行程'],
        ['低改进行程(<3%)', f'{sum(1 for i in improvements if i < 3)}', '物理估计已较准的行程'],
    ]

    fig.add_trace(
        go.Table(
            header=dict(
                values=stats_data[0],
                font=dict(size=11, color='white'),
                fill_color='#3B82F6',
                align='left',
                height=30
            ),
            cells=dict(
                values=[list(col) for col in zip(*stats_data[1:])],
                font=dict(size=10),
                fill_color=[['#F8FAFC', '#E2E8F0'] * 5],
                align='left',
                height=28
            )
        ),
        row=2, col=3
    )

    # ========== 图表布局设置 ==========
    fig.update_layout(
        height=900,
        title=dict(
            text="<b>🔬 物理估计 vs 🤖 ML校准 对比分析</b>",
            font=dict(size=20),
            x=0.5,
            y=0.98
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.75,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        hovermode='closest',
        bargap=0.15,
        bargroupgap=0.1
    )

    # 子图轴标签
    fig.update_xaxes(title_text="物理估计质量 (kg)", row=1, col=1)
    fig.update_yaxes(title_text="ML校准质量 (kg)", row=1, col=1)
    fig.update_xaxes(title_text="质量差异 (kg)", row=1, col=2)
    fig.update_yaxes(title_text="频次", row=1, col=2)
    fig.update_xaxes(title_text="改进幅度 (%)", row=1, col=3)
    fig.update_yaxes(title_text="频次", row=1, col=3)
    fig.update_xaxes(title_text="行程", row=2, col=1)
    fig.update_yaxes(title_text="质量 (kg)", row=2, col=1)
    fig.update_xaxes(title_text="有效窗口数", row=2, col=2)
    fig.update_yaxes(title_text="改进幅度 (%)", row=2, col=2)

    # 添加全局注释
    fig.add_annotation(
        x=0.5, y=1.05, xref='paper', yref='paper',
        text=(
            f"<b>📊 核心发现：</b>"
            f"ML模型平均修正物理估计 <b>{avg_improvement:.1f}%</b> | "
            f"平均质量差异 <b>{np.mean(abs_differences):.0f} kg</b> | "
            f"有效行程 <b>{len(trip_results)}</b> 个"
        ),
        showarrow=False,
        font=dict(size=13, color='#1F2937'),
        bgcolor='#FEF3C7',
        bordercolor='#F59E0B',
        borderwidth=1,
        borderpad=10
    )

    return fig


def generate_mass_time_curve_with_comparison(predictions_df: pd.DataFrame) -> Optional[io.BytesIO]:
    """生成带对比的质量时间曲线（显示物理估计和ML校准两条线）"""
    try:
        required_cols = ['time_seconds', 'physical_mass', 'ml_calibrated_mass']
        if not all(col in predictions_df.columns for col in required_cols):
            return None

        total_points = len(predictions_df)
        if total_points > 50000:
            step = max(1, total_points // 50000)
            plot_df = predictions_df.iloc[::step].copy()
        else:
            plot_df = predictions_df.copy()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # 子图1：两条曲线对比
        valid_mask = plot_df['physical_mass'].notna()

        if valid_mask.any():
            valid_data = plot_df[valid_mask]
            time_valid = valid_data['time_seconds'].values

            # 物理估计曲线
            physical_valid = valid_data['physical_mass'].values
            ax1.plot(time_valid, physical_valid, 'b-', alpha=0.7, linewidth=1.5,
                     label='物理估计 (F=ma)')

            # ML校准曲线
            ml_valid = valid_data['ml_calibrated_mass'].values
            ax1.plot(time_valid, ml_valid, 'g-', alpha=0.7, linewidth=1.5,
                     label='ML校准', marker='o', markersize=2, markevery=max(1, len(time_valid)//100))

        # 填充物理估计和ML校准之间的区域
        if valid_mask.any():
            valid_data = plot_df[valid_mask]
            ax1.fill_between(valid_data['time_seconds'].values,
                            valid_data['physical_mass'].values,
                            valid_data['ml_calibrated_mass'].values,
                            alpha=0.3, color='orange', label='改进区域')

        ax1.set_xlabel('时间 (秒)', fontsize=12)
        ax1.set_ylabel('质量 (kg)', fontsize=12)
        ax1.set_title('物理估计 vs ML校准 质量时间曲线', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3, linestyle='--')

        # 子图2：置信度曲线
        if 'prediction_confidence' in plot_df.columns:
            confidence_valid = plot_df.loc[valid_mask, 'prediction_confidence'].values if valid_mask.any() else []
            if len(confidence_valid) > 0:
                ax2.fill_between(time_valid, 0, confidence_valid, alpha=0.5, color='teal')
                ax2.plot(time_valid, confidence_valid, 'teal-', alpha=0.8, linewidth=1.5)
                ax2.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='阈值线 (0.7)')

        ax2.set_xlabel('时间 (秒)', fontsize=12)
        ax2.set_ylabel('置信度', fontsize=12)
        ax2.set_title('预测置信度时间曲线', fontsize=14, fontweight='bold')
        ax2.set_ylim(0, 1.05)
        ax2.grid(True, alpha=0.3, linestyle='--')

        # 添加统计信息
        improvements = plot_df.loc[valid_mask, 'improvement'].dropna() if 'improvement' in plot_df.columns else pd.Series()
        stats_text = f"总点数: {total_points:,}\n"
        stats_text += f"有效预测点: {valid_mask.sum():,} ({valid_mask.sum()/total_points*100:.1f}%)\n"
        if len(improvements) > 0:
            stats_text += f"平均改进幅度: {improvements.mean():.1f}%\n"
            stats_text += f"最大改进幅度: {improvements.max():.1f}%"
        else:
            if valid_mask.any():
                ml_median = plot_df.loc[valid_mask, 'ml_calibrated_mass'].median()
                physical_median = plot_df.loc[valid_mask, 'physical_mass'].median()
                stats_text += f"ML中位数: {ml_median:.0f} kg\n"
                stats_text += f"物理中位数: {physical_median:.0f} kg\n"
                stats_text += f"差异: {abs(ml_median - physical_median):.0f} kg"

        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return buf
    except Exception as e:
        print(f"生成对比曲线失败: {str(e)}")
        return None


def plot_error_decomposition(trip_results: List[Dict]) -> Optional[go.Figure]:
    """绘制误差分解图 - 分析误差来源"""
    if not trip_results:
        return None

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('误差来源占比', '改进幅度与窗口质量', '改进效果汇总'),
        specs=[[{'type': 'pie'}, {'type': 'scatter'}, {'type': 'bar'}]]
    )

    # 子图1：误差来源分解（基于改进幅度的分布）
    improvements = [t['improvement_pct'] for t in trip_results]

    # 分类
    high_improvement = sum(1 for i in improvements if i > 15)  # 模型显著改善
    medium_improvement = sum(1 for i in improvements if 5 <= i <= 15)  # 模型中等改善
    low_improvement = sum(1 for i in improvements if i < 5)  # 物理估计已经很好

    fig.add_trace(
        go.Pie(
            labels=['ML显著改善 (>15%)', 'ML中等改善 (5-15%)', '物理估计已足够 (<5%)'],
            values=[high_improvement, medium_improvement, low_improvement],
            marker_colors=['#10B981', '#F59E0B', '#6B7280'],
            hole=0.4,
            textinfo='label+percent',
            hoverinfo='label+value+percent'
        ),
        row=1, col=1
    )

    # 子图2：改进幅度与窗口数/置信度的关系
    n_windows = [t['n_windows'] for t in trip_results]
    confidences = [t['confidence'] for t in trip_results]

    fig.add_trace(
        go.Scatter(
            x=n_windows, y=improvements,
            mode='markers',
            marker=dict(
                size=confidences,
                sizeref=2.*max(confidences)/(40.**2),
                sizemin=4,
                color=confidences,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title='置信度', x=0.45),
                line=dict(width=1, color='black')
            ),
            text=[f"行程{t['trip_idx']}<br>改进:{t['improvement_pct']:.1f}%<br>置信度:{t['confidence']:.2f}<br>窗口数:{t['n_windows']}"
                  for t in trip_results],
            hoverinfo='text',
            name='行程'
        ),
        row=1, col=2
    )

    # 添加趋势线
    z = np.polyfit(n_windows, improvements, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(n_windows), max(n_windows), 50)
    fig.add_trace(
        go.Scatter(
            x=x_trend, y=p(x_trend),
            mode='lines', line=dict(dash='solid', color='red', width=2),
            name=f'趋势线'
        ),
        row=1, col=2
    )

    # 子图3：改进效果汇总
    # 计算物理估计的离散度 vs ML的离散度
    physical_stds = [t.get('physical_std', 0) for t in trip_results]
    ml_stds = [t.get('ml_std', 0) for t in trip_results]

    avg_physical_std = np.mean(physical_stds)
    avg_ml_std = np.mean(ml_stds)
    std_reduction = (avg_physical_std - avg_ml_std) / max(avg_physical_std, 1) * 100

    categories = ['物理估计', 'ML校准']
    values = [avg_physical_std, avg_ml_std]

    fig.add_trace(
        go.Bar(
            x=categories, y=values,
            marker_color=['#EF4444', '#10B981'],
            text=[f'{v:.0f}kg' for v in values],
            textposition='outside',
            name='窗口间标准差'
        ),
        row=1, col=3
    )

    # 添加改进标注
    fig.add_annotation(
        x=1, y=avg_ml_std + (avg_physical_std - avg_ml_std) / 2,
        text=f'↓ {std_reduction:.1f}%',
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor='green',
        ax=0.5,
        ay=-30,
        row=1, col=3
    )

    fig.update_layout(
        height=500,
        title_text="📊 误差来源分解与ML改进效果分析",
        showlegend=False
    )
    fig.update_yaxes(title_text="改进幅度 (%)", row=1, col=2)
    fig.update_yaxes(title_text="窗口间标准差 (kg)", row=1, col=3)

    # 添加统计摘要文字
    avg_improvement = np.mean(improvements)
    median_improvement = np.median(improvements)

    fig.add_annotation(
        x=0.5, y=1.08, xref='paper', yref='paper',
        text=f"📈 平均改进: {avg_improvement:.1f}% | 中位数改进: {median_improvement:.1f}% | "
             f"窗口稳定性提升: {std_reduction:.1f}%",
        showarrow=False,
        font=dict(size=12, color='#1F2937'),
        bgcolor='#F3F4F6',
        bordercolor='#D1D5DB',
        borderwidth=1,
        borderpad=4
    )

    return fig