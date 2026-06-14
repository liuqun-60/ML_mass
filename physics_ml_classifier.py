"""
车辆总质量预测系统 - 物理估计 + ML残差校准
基于加速工况的物理模型 F=ma + 机器学习预测残差 (m_true - m_est)
带置信度评估和阈值过滤
"""

import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
from typing import List, Dict, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== 配置类 ====================
class Config:
    """系统配置"""

    # 数据清洗参数
    MIN_SPEED_FOR_TRAINING = 0.1

    # 窗口划分参数
    # 窗口参数（行程内按固定时长切窗，直接 F=ma）
    WINDOW_SIZE = 3.0              # 窗口大小（秒），与 SECONDARY_WINDOW_SIZE 同步
    SECONDARY_WINDOW_SIZE = 3.0    # 兼容旧模型字段名
    MIN_WINDOW_POINTS = 15         # 窗口内至少有效点数
    MIN_SECONDARY_WINDOW_POINTS = 15  # 兼容旧模型字段名
    # 以下字段保留兼容，不再使用
    MIN_PRIMARY_WINDOW_DURATION = 3.0
    MIN_SECONDARY_WINDOWS_PER_PRIMARY = 1
    SAMPLING_RATE = 10.0
    MIN_STOP_DURATION_FOR_TRIP_END = 60.0      # 行程：停车超过此秒数 → 新行程
    MIN_STOP_DURATION_FOR_WINDOW_END = 2.0

    # 加速度筛选参数 - 分级阈值
    MIN_ACCEL = 1.0  # 基础最小加速度阈值 (m/s²)
    MIN_ACCEL_LIGHT = 1.0  # 轻载(<2500kg)的最小加速度
    MIN_ACCEL_HEAVY = 1.0  # 重载(>3000kg)的最小加速度
    MIN_ACCEL_TRANSITION = 1.0  # 过渡区(2500-3000kg)的最小加速度
    LIGHT_MASS_THRESHOLD = 2500  # 轻载阈值(kg)
    HEAVY_MASS_THRESHOLD = 3000  # 重载阈值(kg)
    MIN_SPEED = 10.0
    MIN_FORCE = 2000.0

    # 传动参数
    GEAR_RATIO = 8.0
    TIRE_RADIUS = 0.35
    TRANSMISSION_EFFICIENCY = 0.97

    # 物理模型参数
    ROLLING_RESISTANCE_COEFF = 0.015
    GRAVITY = 9.81

    # 截尾参数（一级窗口百分位截尾：每个行程内去掉两端偏离较大的窗口）
    TRIM_REMOVE_BOTTOM_PCT = 5    # 删除质量最低的下 X% 一级窗口（行程级截尾）
    TRIM_REMOVE_TOP_PCT = 5       # 删除质量最高的上 X% 一级窗口（行程级截尾）
    MIN_TRIM_SAMPLES = 3          # 截尾后至少保留的一级窗口数

    # 区间划分：停车超过此秒数 → 新区间（装卸货），默认 600s
    MIN_STOP_DURATION_FOR_SECTION_END = 600.0

    # ML模型参数
    PREDICTION_MODE = 'residual'
    MAX_RELATIVE_CORRECTION = 0.30
    MIN_MASS_KG = 500
    MAX_MASS_KG = 50000

    RF_N_ESTIMATORS = 200
    RF_MAX_DEPTH = 10
    RF_MIN_SAMPLES_SPLIT = 5
    RF_MIN_SAMPLES_LEAF = 2

    GBDT_N_ESTIMATORS = 200
    GBDT_MAX_DEPTH = 4
    GBDT_LEARNING_RATE = 0.05
    GBDT_MIN_SAMPLES_SPLIT = 5
    GBDT_MIN_SAMPLES_LEAF = 2

    CV_N_FOLDS = 5  # 交叉验证折数

    # 模型参数
    RANDOM_STATE = 42

    # 列名映射
    COLUMN_MAPPING = {
        'Time': 'time_seconds',
        'MCU_ActMotorTq': 'motor_torque_nm',
        'MCU_ActMotorSpd': 'motor_speed_rpm',
        'IC_CarSpeed': 'speed_kmh',
        'Acceleration': 'acceleration_x',
        'Accelerometer X': 'acceleration_x',
        'Mass': 'mass_kg',
        'Force': 'force_n',
    }


# ==================== 数据清洗器 ====================
class DataCleaner:
    """数据清洗器"""

    def __init__(self, config: Config):
        self.config = config

    def clean_data(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """完整数据清洗流程"""
        logger.info("=" * 60)
        logger.info("开始数据清洗")
        logger.info(f"原始数据形状: {df.shape}")

        df_clean = df.copy()

        df_clean = self._rename_columns(df_clean)
        df_clean = self._handle_missing_values(df_clean, is_training)
        df_clean = self._handle_outliers(df_clean)
        df_clean = self._ensure_numeric_types(df_clean)
        df_clean = self._sort_by_time(df_clean)

        df_clean['for_feature_extraction'] = df_clean['speed_kmh'] > 0

        logger.info(f"清洗后数据形状: {df_clean.shape}")
        logger.info("=" * 60)

        return df_clean

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        rename_dict = {}
        for old_name, new_name in self.config.COLUMN_MAPPING.items():
            if old_name in df.columns:
                rename_dict[old_name] = new_name
        if rename_dict:
            df = df.rename(columns=rename_dict)
            logger.info(f"重命名了 {len(rename_dict)} 个列")
        return df

    def _handle_missing_values(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        critical_columns = ['time_seconds', 'motor_torque_nm', 'acceleration_x', 'speed_kmh']
        available_critical = [col for col in critical_columns if col in df.columns]
        if available_critical:
            df = df.dropna(subset=available_critical)
        if is_training and 'mass_kg' in df.columns:
            df = df.dropna(subset=['mass_kg'])
        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        valid_ranges = {
            'motor_torque_nm': (-500, 20000),
            'motor_speed_rpm': (0, 15000),
            'acceleration_x': (-20, 20),
            'speed_kmh': (0, 200),
            'mass_kg': (500, 50000),
            'force_n': (-50000, 50000)
        }
        for col, (min_val, max_val) in valid_ranges.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if col == 'speed_kmh':
                    outlier_mask = (df[col] < 0) | (df[col] > max_val) | df[col].isna()
                else:
                    outlier_mask = (df[col] < min_val) | (df[col] > max_val) | df[col].isna()
                df = df[~outlier_mask].copy()
        return df

    def _ensure_numeric_types(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_columns = ['time_seconds', 'motor_torque_nm', 'motor_speed_rpm',
                          'acceleration_x', 'speed_kmh', 'mass_kg', 'force_n']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    def _sort_by_time(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'time_seconds' in df.columns:
            df = df.sort_values('time_seconds').reset_index(drop=True)
        return df


# ==================== 物理估计器 ====================
class PhysicsEstimator:
    """基于物理模型的质量估计器"""

    def __init__(self, config: Config):
        self.config = config
        self.g = config.GRAVITY
        self.f = config.ROLLING_RESISTANCE_COEFF

    def filter_valid_points(self, df: pd.DataFrame,
                            m_est_previous: float = None) -> pd.DataFrame:
        """筛选有效加速点（支持分级加速度阈值）"""
        df = df.copy()

        required_cols = ['acceleration_x', 'speed_kmh']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"缺少必要列: {col}")

        logger.info(f"加速度范围: {df['acceleration_x'].min():.2f} - {df['acceleration_x'].max():.2f}")
        logger.info(f"车速范围: {df['speed_kmh'].min():.2f} - {df['speed_kmh'].max():.2f}")

        # 驱动力条件
        if 'force_n' in df.columns:
            logger.info(f"驱动力范围: {df['force_n'].min():.2f} - {df['force_n'].max():.2f}")
            force_condition = df['force_n'] > self.config.MIN_FORCE
        elif 'motor_torque_nm' in df.columns:
            logger.info(f"扭矩范围: {df['motor_torque_nm'].min():.2f} - {df['motor_torque_nm'].max():.2f}")
            torque_threshold = self.config.MIN_FORCE * self.config.TIRE_RADIUS / (
                    self.config.GEAR_RATIO * self.config.TRANSMISSION_EFFICIENCY)
            force_condition = df['motor_torque_nm'] > torque_threshold
            logger.info(f"使用扭矩阈值: {torque_threshold:.1f} Nm")
        else:
            raise ValueError("缺少驱动力或扭矩列")

        # 车速条件
        speed_condition = df['speed_kmh'] > self.config.MIN_SPEED

        # 分级加速度阈值 - 根据每个点的质量动态选择
        # 优先使用真实质量（训练时），其次使用传入的估计值，最后使用默认阈值
        if 'mass_kg' in df.columns:
            # 训练模式：有真实质量标签
            masses = df['mass_kg'].values
            accel_thresholds = np.zeros(len(df))

            light_mask = masses < self.config.LIGHT_MASS_THRESHOLD
            heavy_mask = masses > self.config.HEAVY_MASS_THRESHOLD
            transition_mask = ~(light_mask | heavy_mask)

            accel_thresholds[light_mask] = self.config.MIN_ACCEL_LIGHT
            accel_thresholds[heavy_mask] = self.config.MIN_ACCEL_HEAVY
            accel_thresholds[transition_mask] = self.config.MIN_ACCEL_TRANSITION

            accel_condition = df['acceleration_x'] > accel_thresholds

            logger.info(f"使用分级加速度阈值 (基于真实质量): "
                        f"轻载={self.config.MIN_ACCEL_LIGHT:.1f} (质量<{self.config.LIGHT_MASS_THRESHOLD}kg), "
                        f"重载={self.config.MIN_ACCEL_HEAVY:.1f} (质量>{self.config.HEAVY_MASS_THRESHOLD}kg), "
                        f"过渡={self.config.MIN_ACCEL_TRANSITION:.1f} m/s²")

        elif m_est_previous is not None:
            # 预测模式但有上一行程估计值
            if m_est_previous < self.config.LIGHT_MASS_THRESHOLD:
                accel_threshold = self.config.MIN_ACCEL_LIGHT
                logger.info(f"使用轻载加速度阈值: {accel_threshold} m/s² (上一行程质量={m_est_previous:.0f}kg)")
            elif m_est_previous > self.config.HEAVY_MASS_THRESHOLD:
                accel_threshold = self.config.MIN_ACCEL_HEAVY
                logger.info(f"使用重载加速度阈值: {accel_threshold} m/s² (上一行程质量={m_est_previous:.0f}kg)")
            else:
                accel_threshold = self.config.MIN_ACCEL_TRANSITION
                logger.info(f"使用过渡区加速度阈值: {accel_threshold} m/s² (上一行程质量={m_est_previous:.0f}kg)")

            accel_condition = df['acceleration_x'] > accel_threshold

        else:
            # 预测模式：无任何质量信息，使用默认阈值
            accel_threshold = self.config.MIN_ACCEL
            logger.info(f"使用默认加速度阈值: {accel_threshold} m/s² (无质量信息)")
            accel_condition = df['acceleration_x'] > accel_threshold

        # 综合所有条件
        mask = speed_condition & force_condition & accel_condition

        df['is_valid'] = mask.astype(int)
        if 'accel_threshold_used' not in df.columns:
            if 'accel_thresholds' in locals():
                df['accel_threshold_used'] = accel_thresholds
            else:
                df['accel_threshold_used'] = accel_threshold if 'accel_threshold' in locals() else self.config.MIN_ACCEL

        valid_count = mask.sum()
        logger.info(f"逐帧筛选: {valid_count}/{len(df)} 个有效点 ({valid_count / len(df) * 100:.1f}%)")

        return df

    def split_trips(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """按车速0→0，停车>指定时长划分行程"""
        logger.info("开始划分行程...")

        if 'speed_kmh' not in df.columns or 'time_seconds' not in df.columns:
            logger.error("缺少车速或时间列")
            return []

        df = df.sort_values('time_seconds').reset_index(drop=True)
        trips = []
        in_trip = False
        trip_start_idx = 0
        i = 0
        total_len = len(df)

        while i < total_len:
            speed = df.at[i, 'speed_kmh']

            if not in_trip:
                if speed > 0:
                    in_trip = True
                    trip_start_idx = i
                i += 1
            else:
                if speed == 0:
                    zero_start = i
                    j = i + 1
                    while j < total_len and df.at[j, 'speed_kmh'] == 0:
                        j += 1

                    stop_duration = (j - zero_start) / self.config.SAMPLING_RATE

                    if stop_duration >= self.config.MIN_STOP_DURATION_FOR_TRIP_END:
                        trip_df = df.iloc[trip_start_idx:zero_start].copy()
                        if len(trip_df) > 0:
                            trips.append(trip_df)
                            logger.info(f"  行程 {len(trips)}: {trip_df['time_seconds'].iloc[0]:.0f}s - {trip_df['time_seconds'].iloc[-1]:.0f}s ({len(trip_df)} 行)")
                        in_trip = False
                        i = j
                    else:
                        i = j
                else:
                    i += 1

        if in_trip:
            trip_df = df.iloc[trip_start_idx:total_len].copy()
            if len(trip_df) > 0:
                trips.append(trip_df)
                logger.info(f"  行程 {len(trips)}: {trip_df['time_seconds'].iloc[0]:.0f}s - {trip_df['time_seconds'].iloc[-1]:.0f}s ({len(trip_df)} 行)")

        logger.info(f"行程划分完成: {len(trips)} 个行程")
        return trips

    def split_primary_windows(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """
        行程内按固定时长切窗（从行程起点对齐、不重叠）。
        每个窗口含该时段内全部采样点，F=ma 在 estimate_primary_window_mass 中对有效点计算。
        """
        if 'is_valid' not in df.columns:
            raise ValueError("请先执行 filter_valid_points")

        if len(df) == 0:
            return []

        df = df.sort_values('time_seconds').reset_index(drop=True)
        window_size = self.config.WINDOW_SIZE
        trip_start = float(df['time_seconds'].iloc[0])
        trip_end = float(df['time_seconds'].iloc[-1])

        windows = []
        current_start = trip_start
        while current_start + window_size <= trip_end:
            current_end = current_start + window_size
            mask = (df['time_seconds'] >= current_start) & (df['time_seconds'] < current_end)
            window_df = df.loc[mask].copy()
            if len(window_df) > 0:
                windows.append(window_df)
            current_start = current_end

        logger.info(f"行程窗口划分完成: {len(windows)} 个 (每窗 {window_size}s)")
        return windows

    def calculate_window_mass(self, window_df: pd.DataFrame) -> Optional[float]:
        """对单个时间窗口内的有效点计算 F=ma 质量"""
        min_pts = self.config.MIN_WINDOW_POINTS
        if len(window_df) < min_pts:
            return None

        if 'force_n' in window_df.columns:
            force_values = pd.to_numeric(window_df['force_n'], errors='coerce').fillna(0).values
        else:
            return None

        accel_values = pd.to_numeric(window_df['acceleration_x'], errors='coerce').fillna(0).values

        if len(force_values) == 0 or len(accel_values) == 0:
            return None

        F_mean = np.mean(force_values)
        a_mean = np.mean(accel_values)

        denominator = a_mean + self.g * self.f

        if abs(denominator) < 0.01:
            return None

        m_est = F_mean / denominator

        if m_est < 500 or m_est > 50000:
            return None

        return float(m_est)

    calculate_secondary_window_mass = calculate_window_mass  # 兼容旧名称

    def estimate_primary_window_mass(self, window_df: pd.DataFrame) -> Dict[str, Any]:
        """对单个时间窗口：取有效点，直接 F=ma（截尾在行程级进行）"""
        valid_data = window_df[window_df['is_valid'] == 1].copy()
        min_pts = self.config.MIN_WINDOW_POINTS

        if len(valid_data) < min_pts:
            return {'m_est': None, 'n_secondary': len(valid_data), 'm_est_list': [], 'm_est_cv': 0}

        m_est = self.calculate_window_mass(valid_data)
        if m_est is None:
            return {'m_est': None, 'n_secondary': len(valid_data), 'm_est_list': [], 'm_est_cv': 0}

        return {
            'm_est': m_est,
            'n_secondary': len(valid_data),
            'm_est_list': [m_est],
            'm_est_cv': 0.0,
        }

    def get_trim_indices(self, masses: List[float]) -> List[int]:
        """行程级百分位截尾：返回保留的一级窗口下标（仅行程聚合时使用）"""
        n = len(masses)
        if n == 0:
            return []
        if n < 2:
            return list(range(n))

        bottom_n = int(n * self.config.TRIM_REMOVE_BOTTOM_PCT / 100.0)
        top_n = int(n * self.config.TRIM_REMOVE_TOP_PCT / 100.0)

        if bottom_n + top_n >= n:
            return list(range(n))

        sorted_indices = sorted(range(n), key=lambda i: masses[i])
        kept = sorted_indices[bottom_n:n - top_n] if top_n > 0 else sorted_indices[bottom_n:]

        if len(kept) < self.config.MIN_TRIM_SAMPLES:
            return list(range(n))

        return sorted(kept)

    def trim_primary_window_masses(self, masses: List[float]) -> List[float]:
        """对一级窗口质量列表做百分位截尾，保留中间相对集中的窗口"""
        indices = self.get_trim_indices(masses)
        return [masses[i] for i in indices]

    def aggregate_trip_mass_from_primary_windows(
            self, primary_masses: List[float],
            trim_reference: List[float] = None) -> Tuple[float, int, int]:
        """
        行程质量 = 一级窗口在行程级截尾后的中位数。
        trim_reference: 用于决定截尾下标的参考序列（默认与 primary_masses 相同；
                        预测时可用 ML 质量决定截尾，再同步作用于物理质量）。
        返回: (中位数, 截尾后窗口数, 截尾前有效窗口数)
        """
        if len(primary_masses) == 0:
            raise ValueError("一级窗口质量列表为空")
        ref = trim_reference if trim_reference is not None else primary_masses
        indices = self.get_trim_indices(ref)
        trimmed = [primary_masses[i] for i in indices]
        return float(np.median(trimmed)), len(indices), len(primary_masses)

    def extract_primary_window_features(self, primary_window: pd.DataFrame,
                                       m_est_result: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """提取一级窗口的完整特征（16个）"""
        try:
            valid_data = primary_window[primary_window['is_valid'] == 1].copy()

            if len(valid_data) == 0 or m_est_result['m_est'] is None:
                return None

            accel_values = pd.to_numeric(valid_data['acceleration_x'], errors='coerce').fillna(0).values
            speed_values = pd.to_numeric(valid_data['speed_kmh'], errors='coerce').fillna(0).values
            torque_values = pd.to_numeric(valid_data['motor_torque_nm'], errors='coerce').fillna(0).values

            if 'force_n' in valid_data.columns:
                force_values = pd.to_numeric(valid_data['force_n'], errors='coerce').fillna(0).values
            else:
                force_values = np.zeros_like(torque_values)

            # 物理估计特征
            m_est = m_est_result['m_est']
            m_est_cv = m_est_result['m_est_cv']
            n_secondary = m_est_result['n_secondary']

            # 加速度特征
            accel_mean = float(np.mean(accel_values))
            accel_max = float(np.max(accel_values))
            accel_std = float(np.std(accel_values))

            # 车速特征
            speed_mean = float(np.mean(speed_values))
            speed_range = float(np.max(speed_values) - np.min(speed_values))

            # 扭矩/力特征
            tq_mean = float(np.mean(torque_values))
            tq_cv = float(np.std(torque_values) / np.mean(torque_values)) if tq_mean > 0 else 0.0
            force_mean = float(np.mean(force_values))

            # 工况标识特征
            is_strong_accel = 1.0 if accel_mean > 1.5 else 0.0
            accel_quality = self._calculate_accel_quality(
                accel_mean, accel_std, speed_mean, tq_cv, m_est_cv
            )

            # 数据质量特征
            valid_ratio = float(len(valid_data) / len(primary_window)) if len(primary_window) > 0 else 0.0
            window_duration = float(valid_data['time_seconds'].iloc[-1] - valid_data['time_seconds'].iloc[0])

            # 功率特征
            motor_power_mean = 0.0
            if 'motor_speed_rpm' in valid_data.columns:
                rpm_values = pd.to_numeric(valid_data['motor_speed_rpm'], errors='coerce').fillna(0).values
                rpm_mean = np.mean(rpm_values)
                motor_power_mean = float(tq_mean * rpm_mean / 9550.0)

            features = {
                'm_est': m_est,
                'm_est_cv': m_est_cv,
                'n_secondary': float(n_secondary),
                'accel_mean': accel_mean,
                'accel_max': accel_max,
                'accel_std': accel_std,
                'speed_mean': speed_mean,
                'speed_range': speed_range,
                'tq_mean': tq_mean,
                'tq_cv': tq_cv,
                'force_mean': force_mean,
                'is_strong_accel': is_strong_accel,
                'accel_quality': accel_quality,
                'valid_ratio': valid_ratio,
                'window_duration': window_duration,
                'motor_power_mean': motor_power_mean,
            }

            for key, value in features.items():
                if np.isnan(value) or np.isinf(value):
                    features[key] = 0.0

            return features

        except Exception as e:
            logger.warning(f"提取特征失败: {str(e)}")
            return None

    def _calculate_accel_quality(self, accel_mean, accel_std, speed_mean, tq_cv, m_est_cv):
        """计算加速质量评分"""
        score = 1.0

        if accel_std > 0.3:
            score -= 0.3
        elif accel_std > 0.2:
            score -= 0.15

        if speed_mean < 15:
            score -= 0.2
        elif speed_mean > 70:
            score -= 0.2

        if tq_cv > 0.25:
            score -= 0.3
        elif tq_cv > 0.15:
            score -= 0.15

        if m_est_cv > 0.15:
            score -= 0.3
        elif m_est_cv > 0.10:
            score -= 0.15

        return float(max(0.0, min(1.0, score)))

    def split_sections(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """按车速0→0，停车>指定时长划分区间"""
        logger.info("开始划分区间...")

        if 'speed_kmh' not in df.columns or 'time_seconds' not in df.columns:
            logger.error("缺少车速或时间列")
            return []

        df = df.sort_values('time_seconds').reset_index(drop=True)
        sections = []
        in_section = False
        section_start_idx = 0
        i = 0
        total_len = len(df)

        while i < total_len:
            speed = df.at[i, 'speed_kmh']

            if not in_section:
                if speed > 0:
                    in_section = True
                    section_start_idx = i
                i += 1
            else:
                if speed == 0:
                    zero_start = i
                    j = i + 1
                    while j < total_len and df.at[j, 'speed_kmh'] == 0:
                        j += 1

                    stop_duration = (j - zero_start) / self.config.SAMPLING_RATE

                    if stop_duration >= self.config.MIN_STOP_DURATION_FOR_SECTION_END:
                        section_df = df.iloc[section_start_idx:zero_start].copy()
                        if len(section_df) > 0:
                            sections.append(section_df)
                            logger.info(
                                f"  区间 {len(sections)}: {section_df['time_seconds'].iloc[0]:.0f}s - {section_df['time_seconds'].iloc[-1]:.0f}s ({len(section_df)} 行)")
                        in_section = False
                        i = j
                    else:
                        i = j
                else:
                    i += 1

        if in_section:
            section_df = df.iloc[section_start_idx:total_len].copy()
            if len(section_df) > 0:
                sections.append(section_df)
                logger.info(
                    f"  区间 {len(sections)}: {section_df['time_seconds'].iloc[0]:.0f}s - {section_df['time_seconds'].iloc[-1]:.0f}s ({len(section_df)} 行)")

        logger.info(f"区间划分完成: {len(sections)} 个区间")
        return sections

    def split_trips_within_section(self, section_df: pd.DataFrame) -> List[pd.DataFrame]:
        """在区间内划分行程 - 直接复用原有的行程划分逻辑"""
        logger.info(f"  在区间内划分行程...")
        # 直接调用原有的 split_trips 方法
        trips = self.split_trips(section_df)
        logger.info(f"  区间内行程划分完成: {len(trips)} 个行程")
        return trips

# ==================== 物理+ML质量预测器 ====================
class PhysicsMLPredictor:
    """物理估计 + ML校准的质量预测器，带置信度评估和截尾处理"""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        if self.config.MIN_STOP_DURATION_FOR_SECTION_END <= self.config.MIN_STOP_DURATION_FOR_TRIP_END:
            logger.warning(
                f"区间停车阈值({self.config.MIN_STOP_DURATION_FOR_SECTION_END}s) "
                f"≤ 行程阈值({self.config.MIN_STOP_DURATION_FOR_TRIP_END}s)，"
                f"已自动修正区间为 600s"
            )
            self.config.MIN_STOP_DURATION_FOR_SECTION_END = 600.0
        self.cleaner = DataCleaner(self.config)
        self.physics_estimator = PhysicsEstimator(self.config)
        self.scaler = StandardScaler()
        self.ml_model = None

        self.is_trained = False
        self.feature_names = [
            'm_est', 'm_est_cv', 'n_secondary',
            'accel_mean', 'accel_max', 'accel_std',
            'speed_mean', 'speed_range',
            'tq_mean', 'tq_cv', 'force_mean',
            'is_strong_accel', 'accel_quality',
            'valid_ratio', 'window_duration', 'motor_power_mean'
        ]
        self.training_metrics = {}
        self.training_history = {}
        self.residual_stats = {}
        self.prediction_mode = self.config.PREDICTION_MODE

    def _clamp_mass(self, mass) -> float:
        """限制质量在合理范围"""
        return float(np.clip(mass, self.config.MIN_MASS_KG, self.config.MAX_MASS_KG))

    def _clamp_mass_array(self, masses: np.ndarray) -> np.ndarray:
        return np.clip(masses, self.config.MIN_MASS_KG, self.config.MAX_MASS_KG)

    def _apply_residual_correction(self, m_est: float, residual: float) -> float:
        """将 ML 预测的残差叠加到物理估计上，并限制修正幅度"""
        max_corr = max(200.0, abs(m_est) * self.config.MAX_RELATIVE_CORRECTION)
        residual = float(np.clip(residual, -max_corr, max_corr))
        return self._clamp_mass(m_est + residual)

    def _predict_calibrated_mass(self, m_est: float, X_scaled: np.ndarray) -> Tuple[float, float]:
        """
        由特征预测校准质量。
        返回 (校准质量, ML预测残差)
        """
        ml_out = float(self.ml_model.predict(X_scaled.reshape(1, -1) if X_scaled.ndim == 1 else X_scaled)[0])
        if self.prediction_mode == 'residual':
            return self._apply_residual_correction(m_est, ml_out), ml_out
        return self._clamp_mass(ml_out), ml_out - m_est

    def train(self, df: pd.DataFrame, ml_model_type: str = 'gbdt',
              n_estimators: int = None, max_depth: int = None,
              learning_rate: float = None, min_samples_split: int = None,
              min_samples_leaf: int = None, n_folds: int = None):
        """训练模型"""
        logger.info("=" * 60)
        logger.info("开始训练物理+ML残差预测模型（行程窗口百分位截尾）")
        logger.info(f"预测模式: {self.config.PREDICTION_MODE} (目标=m_true-m_est)")
        logger.info(f"ML参数: 类型={ml_model_type}, 树={n_estimators}, 深度={max_depth}, "
                    f"学习率={learning_rate}, 最小分割={min_samples_split}, "
                    f"最小叶子={min_samples_leaf}, CV折数={n_folds}")
        logger.info("=" * 60)

        # 更新模型参数
        if n_estimators is not None:
            self.config.GBDT_N_ESTIMATORS = n_estimators
            self.config.RF_N_ESTIMATORS = n_estimators
        if max_depth is not None:
            self.config.GBDT_MAX_DEPTH = max_depth
            self.config.RF_MAX_DEPTH = max_depth
        if learning_rate is not None:
            self.config.GBDT_LEARNING_RATE = learning_rate
        if min_samples_split is not None:
            self.config.GBDT_MIN_SAMPLES_SPLIT = min_samples_split
            self.config.RF_MIN_SAMPLES_SPLIT = min_samples_split
        if min_samples_leaf is not None:
            self.config.GBDT_MIN_SAMPLES_LEAF = min_samples_leaf
            self.config.RF_MIN_SAMPLES_LEAF = min_samples_leaf
        if n_folds is not None:
            self.config.CV_N_FOLDS = n_folds

        # 1. 数据清洗
        df_clean = self.cleaner.clean_data(df, is_training=True)

        # 2. 筛选有效点（首次用默认阈值）
        df_clean = self.physics_estimator.filter_valid_points(df_clean)

        # 3. 划分行程
        trips = self.physics_estimator.split_trips(df_clean)

        if len(trips) == 0:
            raise ValueError("未找到有效行程")

        # 4. 提取所有行程的窗口特征
        all_features = []
        all_true_masses = []
        all_m_est = []
        all_trip_ids = []

        prev_trip_mass = None  # 用于传递上一行程质量

        for trip_idx, trip_df in enumerate(trips):
            if 'mass_kg' not in trip_df.columns:
                continue

            mass_values = trip_df['mass_kg'].dropna().values
            if len(mass_values) == 0:
                continue

            trip_mass = float(np.median(mass_values))

            # 使用上一行程的质量作为当前行程的参考
            if prev_trip_mass is not None:
                # 重新筛选当前行程的有效点，使用上一行程质量作为参考
                trip_df = self.physics_estimator.filter_valid_points(trip_df, prev_trip_mass)
            else:
                trip_df = self.physics_estimator.filter_valid_points(trip_df)

            primary_windows = self.physics_estimator.split_primary_windows(trip_df)

            logger.info(f"行程 {trip_idx}: 质量={trip_mass:.0f}kg, {len(primary_windows)} 个窗口")

            for primary_window in primary_windows:
                m_est_result = self.physics_estimator.estimate_primary_window_mass(primary_window)

                if m_est_result['m_est'] is None:
                    continue

                features = self.physics_estimator.extract_primary_window_features(
                    primary_window, m_est_result
                )

                if features is None:
                    continue

                all_features.append(features)
                all_true_masses.append(trip_mass)
                all_m_est.append(m_est_result['m_est'])
                all_trip_ids.append(trip_idx)

            prev_trip_mass = trip_mass  # 更新上一行程质量

        if len(all_features) == 0:
            raise ValueError("未提取到有效训练特征")

        # 5. 构建特征矩阵与残差标签
        X = pd.DataFrame(all_features)
        for feature in self.feature_names:
            if feature not in X.columns:
                X[feature] = 0.0

        X = X[self.feature_names].values
        m_est_arr = np.array(all_m_est)
        y_true = np.array(all_true_masses)
        y_residual = y_true - m_est_arr  # ML 学习目标：物理估计偏差
        groups = np.array(all_trip_ids)

        logger.info(f"特征提取完成: {X.shape[0]} 个窗口样本, "
                    f"{X.shape[1]} 个特征, {len(np.unique(groups))} 个行程")
        logger.info(f"残差统计: mean={np.mean(y_residual):.0f}kg, "
                    f"std={np.std(y_residual):.0f}kg, "
                    f"|residual| median={np.median(np.abs(y_residual)):.0f}kg")

        physics_mae = float(mean_absolute_error(y_true, m_est_arr))
        logger.info(f"物理估计基线 MAE: {physics_mae:.0f} kg")

        # 6. 标准化
        X_scaled = self.scaler.fit_transform(X)

        # 7. 交叉验证（在最终质量上评估）
        cv_metrics, cv_predictions, cv_actuals = self._cross_validate(
            X_scaled, y_residual, m_est_arr, y_true, groups, n_folds=self.config.CV_N_FOLDS
        )

        # 8. 训练最终残差模型
        logger.info("训练最终ML残差模型...")
        self.prediction_mode = self.config.PREDICTION_MODE

        if ml_model_type == 'linear':
            self.ml_model = LinearRegression()
        elif ml_model_type == 'rf':
            self.ml_model = RandomForestRegressor(
                n_estimators=self.config.RF_N_ESTIMATORS,
                max_depth=self.config.RF_MAX_DEPTH,
                min_samples_split=self.config.RF_MIN_SAMPLES_SPLIT,
                min_samples_leaf=self.config.RF_MIN_SAMPLES_LEAF,
                random_state=self.config.RANDOM_STATE,
                n_jobs=-1
            )
        else:  # gbdt
            self.ml_model = GradientBoostingRegressor(
                n_estimators=self.config.GBDT_N_ESTIMATORS,
                max_depth=self.config.GBDT_MAX_DEPTH,
                learning_rate=self.config.GBDT_LEARNING_RATE,
                min_samples_split=self.config.GBDT_MIN_SAMPLES_SPLIT,
                min_samples_leaf=self.config.GBDT_MIN_SAMPLES_LEAF,
                random_state=self.config.RANDOM_STATE
            )

        self.ml_model.fit(X_scaled, y_residual)

        # 9. 训练集预测：残差 → 校准质量
        residual_pred = self.ml_model.predict(X_scaled)
        y_pred_mass = self._clamp_mass_array(m_est_arr + residual_pred)
        # 逐点应用修正幅度限制
        for i in range(len(y_pred_mass)):
            y_pred_mass[i] = self._apply_residual_correction(m_est_arr[i], residual_pred[i])

        residuals = y_true - y_pred_mass

        # 10. 计算残差统计（基于最终质量误差，用于置信度）
        self.residual_stats = self._compute_residual_stats(X, residuals)

        # 11. 保存训练历史
        self.training_history = {
            'X': X,
            'y': y_true,
            'm_est': m_est_arr,
            'y_residual': y_residual,
            'residual_pred': residual_pred,
            'y_pred': y_pred_mass,
            'residuals': residuals,
            'groups': groups,
            'cv_predictions': cv_predictions,
            'cv_actuals': cv_actuals,
            'feature_importance': self._get_feature_importance(),
            'features_df': pd.DataFrame(all_features),
            'prediction_mode': self.prediction_mode,
        }

        ml_mae = float(mean_absolute_error(y_true, y_pred_mass))
        improvement = (physics_mae - ml_mae) / max(physics_mae, 1) * 100

        # 12. 计算训练指标
        self.training_metrics = {
            'mae_kg': ml_mae,
            'rmse_kg': float(np.sqrt(mean_squared_error(y_true, y_pred_mass))),
            'mape': float(np.mean(np.abs(residuals / y_true)) * 100),
            'r2': float(r2_score(y_true, y_pred_mass)),
            'physics_mae_kg': physics_mae,
            'improvement_over_physics_pct': float(improvement),
            'residual_mae_kg': float(mean_absolute_error(y_residual, residual_pred)),
            'n_samples': len(y_true),
            'n_trips': len(np.unique(groups)),
            'prediction_mode': self.prediction_mode,
            'cv_mae_mean': cv_metrics['mae_mean'],
            'cv_mae_std': cv_metrics['mae_std'],
            'cv_rmse_mean': cv_metrics['rmse_mean'],
            'cv_rmse_std': cv_metrics['rmse_std'],
            'cv_r2_mean': cv_metrics['r2_mean'],
            'cv_physics_mae_mean': cv_metrics.get('physics_mae_mean', physics_mae),
            'cv_mae_values': cv_metrics.get('mae_values', []),
            'cv_r2_values': cv_metrics.get('r2_values', []),
        }

        self.is_trained = True

        logger.info(f"训练完成: 物理MAE={physics_mae:.0f}kg → ML MAE={ml_mae:.0f}kg "
                    f"(改善{improvement:+.1f}%), R²={self.training_metrics['r2']:.3f}")
        logger.info(f"CV MAE: {cv_metrics['mae_mean']:.0f}±{cv_metrics['mae_std']:.0f}kg, "
                    f"CV R²: {cv_metrics['r2_mean']:.3f}")

        # 过拟合诊断
        if cv_metrics['mae_mean'] > 0 and self.training_metrics['mae_kg'] > 0:
            overfit_ratio = cv_metrics['mae_mean'] / max(self.training_metrics['mae_kg'], 1)
            if overfit_ratio > 5:
                logger.warning(f"⚠️ 严重过拟合! CV MAE/训练MAE = {overfit_ratio:.1f}")
                logger.warning("建议: 减少树的数量和深度，或增加训练样本")
            elif overfit_ratio > 3:
                logger.warning(f"⚠️ 存在过拟合! CV MAE/训练MAE = {overfit_ratio:.1f}")

        logger.info("=" * 60)

    def _cross_validate(self, X, y_residual, m_est, y_true_mass, groups, n_folds=None):
        """行程级别分组交叉验证（残差训练，质量评估）"""
        if n_folds is None:
            n_folds = self.config.CV_N_FOLDS

        n_folds = min(n_folds, len(np.unique(groups)))
        logger.info(f"开始{n_folds}折分组交叉验证（残差模式）...")

        gkf = GroupKFold(n_splits=n_folds)

        mae_list = []
        rmse_list = []
        r2_list = []
        physics_mae_list = []
        all_cv_preds = np.zeros(len(y_true_mass))
        all_cv_actuals = y_true_mass.copy()

        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y_residual, groups)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_res_train = y_residual[train_idx]
            m_est_test = m_est[test_idx]
            y_mass_test = y_true_mass[test_idx]

            model_cv = GradientBoostingRegressor(
                n_estimators=self.config.GBDT_N_ESTIMATORS,
                max_depth=self.config.GBDT_MAX_DEPTH,
                learning_rate=self.config.GBDT_LEARNING_RATE,
                min_samples_split=self.config.GBDT_MIN_SAMPLES_SPLIT,
                min_samples_leaf=self.config.GBDT_MIN_SAMPLES_LEAF,
                random_state=self.config.RANDOM_STATE
            )
            model_cv.fit(X_train, y_res_train)

            res_pred = model_cv.predict(X_test)
            y_pred_mass = np.array([
                self._apply_residual_correction(m_est_test[i], res_pred[i])
                for i in range(len(m_est_test))
            ])
            all_cv_preds[test_idx] = y_pred_mass

            mae = mean_absolute_error(y_mass_test, y_pred_mass)
            rmse = np.sqrt(mean_squared_error(y_mass_test, y_pred_mass))
            r2 = r2_score(y_mass_test, y_pred_mass)
            phy_mae = mean_absolute_error(y_mass_test, m_est_test)

            mae_list.append(mae)
            rmse_list.append(rmse)
            r2_list.append(r2)
            physics_mae_list.append(phy_mae)

            logger.info(f"  第{fold + 1}折: 物理MAE={phy_mae:.0f}kg → ML MAE={mae:.0f}kg, R²={r2:.3f}")

        return {
            'mae_mean': float(np.mean(mae_list)),
            'mae_std': float(np.std(mae_list)),
            'rmse_mean': float(np.mean(rmse_list)),
            'rmse_std': float(np.std(rmse_list)),
            'r2_mean': float(np.mean(r2_list)),
            'physics_mae_mean': float(np.mean(physics_mae_list)),
            'mae_values': [float(x) for x in mae_list],
            'r2_values': [float(x) for x in r2_list],
        }, all_cv_preds, all_cv_actuals

    def _compute_residual_stats(self, X, residuals):
        """计算残差统计"""
        m_est_values = X[:, 0]

        stats = {}
        bins = [0, 1500, 2000, 2500, 3000, 3500, 4000, 5000, np.inf]
        bin_labels = ['<1500', '1500-2000', '2000-2500', '2500-3000',
                     '3000-3500', '3500-4000', '4000-5000', '>5000']

        for i in range(len(bins)-1):
            mask = (m_est_values >= bins[i]) & (m_est_values < bins[i+1])
            if mask.sum() > 0:
                bin_residuals = np.abs(residuals[mask])
                stats[bin_labels[i]] = {
                    'count': int(mask.sum()),
                    'mae': float(np.mean(bin_residuals)),
                    'std': float(np.std(bin_residuals)),
                    'median': float(np.median(bin_residuals)),
                }

        abs_residuals = np.abs(residuals)
        stats['global'] = {
            'mae': float(np.mean(abs_residuals)),
            'std': float(np.std(abs_residuals)),
            'median': float(np.median(abs_residuals)),
            'q25': float(np.percentile(abs_residuals, 25)),
            'q75': float(np.percentile(abs_residuals, 75)),
        }

        return stats

    def _get_feature_importance(self):
        """获取特征重要性"""
        if hasattr(self.ml_model, 'feature_importances_'):
            importance = self.ml_model.feature_importances_
            return pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
        return None

        return None

    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """预测新数据（区间聚合版本，不含对比分析）"""
        logger.info("=" * 60)
        logger.info("开始预测（区间聚合）")
        logger.info("=" * 60)

        if not self.is_trained:
            raise ValueError("模型尚未训练")

        df_clean = self.cleaner.clean_data(df, is_training=False)
        original_df = df_clean.copy()

        temp_df = df_clean.copy()
        if 'force_n' in temp_df.columns:
            temp_force_condition = temp_df['force_n'] > self.config.MIN_FORCE
        elif 'motor_torque_nm' in temp_df.columns:
            torque_threshold = self.config.MIN_FORCE * self.config.TIRE_RADIUS / (
                    self.config.GEAR_RATIO * self.config.TRANSMISSION_EFFICIENCY)
            temp_force_condition = temp_df['motor_torque_nm'] > torque_threshold
        else:
            raise ValueError("缺少驱动力或扭矩列")

        temp_speed_condition = temp_df['speed_kmh'] > self.config.MIN_SPEED
        temp_df['is_valid_temp'] = (temp_speed_condition & temp_force_condition).astype(int)

        sections = self.physics_estimator.split_sections(temp_df)

        if len(sections) == 0:
            return self._create_empty_results(original_df)

        original_df['predicted_mass'] = np.nan
        original_df['trip_id'] = -1
        original_df['section_id'] = -1
        original_df['is_valid_prediction'] = False
        original_df['aggregation_level'] = 'none'

        all_section_results = []
        all_trip_results = []
        global_prev_mass = None
        global_trip_idx = 0

        for section_idx, section_df in enumerate(sections):
            logger.info(f"\n处理区间 {section_idx}")

            section_trips = self.physics_estimator.split_trips_within_section(section_df)

            if len(section_trips) == 0:
                continue

            section_trip_masses = []
            section_trip_window_counts = []

            for trip_df in section_trips:
                if global_prev_mass is not None:
                    trip_df = self.physics_estimator.filter_valid_points(trip_df, global_prev_mass)
                else:
                    trip_df = self.physics_estimator.filter_valid_points(trip_df)

                primary_windows = self.physics_estimator.split_primary_windows(trip_df)

                if len(primary_windows) == 0:
                    trip_mask = (
                            (original_df['time_seconds'] >= trip_df['time_seconds'].iloc[0]) &
                            (original_df['time_seconds'] <= trip_df['time_seconds'].iloc[-1])
                    )
                    original_df.loc[trip_mask, 'trip_id'] = global_trip_idx
                    original_df.loc[trip_mask, 'section_id'] = section_idx
                    global_trip_idx += 1
                    continue

                trip_masses = []

                for primary_window in primary_windows:
                    m_est_result = self.physics_estimator.estimate_primary_window_mass(primary_window)

                    if m_est_result['m_est'] is None:
                        continue

                    features = self.physics_estimator.extract_primary_window_features(
                        primary_window, m_est_result
                    )

                    if features is None:
                        continue

                    features_df = pd.DataFrame([features])
                    for fname in self.feature_names:
                        if fname not in features_df.columns:
                            features_df[fname] = 0.0

                    X_window = features_df[self.feature_names].values
                    X_window_scaled = self.scaler.transform(X_window)
                    m_physical = m_est_result['m_est']
                    m_calibrated, _ = self._predict_calibrated_mass(m_physical, X_window_scaled)
                    trip_masses.append(m_calibrated)

                if len(trip_masses) == 0:
                    trip_mask = (
                            (original_df['time_seconds'] >= trip_df['time_seconds'].iloc[0]) &
                            (original_df['time_seconds'] <= trip_df['time_seconds'].iloc[-1])
                    )
                    original_df.loc[trip_mask, 'trip_id'] = global_trip_idx
                    original_df.loc[trip_mask, 'section_id'] = section_idx
                    global_trip_idx += 1
                    continue

                trip_mass_final, _, _ = self.physics_estimator.aggregate_trip_mass_from_primary_windows(trip_masses)

                all_trip_results.append({
                    'trip_idx': global_trip_idx,
                    'section_id': section_idx,
                    'mass': trip_mass_final,
                    'n_windows': len(trip_masses),
                    'start_time': float(trip_df['time_seconds'].iloc[0]),
                    'end_time': float(trip_df['time_seconds'].iloc[-1]),
                })

                section_trip_masses.append(trip_mass_final)
                section_trip_window_counts.append(len(trip_masses))

                trip_mask = (
                        (original_df['time_seconds'] >= trip_df['time_seconds'].iloc[0]) &
                        (original_df['time_seconds'] <= trip_df['time_seconds'].iloc[-1])
                )
                original_df.loc[trip_mask, 'trip_id'] = global_trip_idx
                original_df.loc[trip_mask, 'section_id'] = section_idx

                global_prev_mass = trip_mass_final
                global_trip_idx += 1

            if len(section_trip_masses) == 0:
                continue

            section_mass = self._aggregate_section_mass(
                section_trip_masses,
                section_trip_window_counts,
                method='median'
            )

            section_result = {
                'section_idx': section_idx,
                'mass': section_mass,
                'n_trips': len(section_trip_masses),
                'n_total_windows': sum(section_trip_window_counts),
                'start_time': float(section_df['time_seconds'].iloc[0]),
                'end_time': float(section_df['time_seconds'].iloc[-1]),
                'trip_masses': section_trip_masses,
            }
            all_section_results.append(section_result)

            section_mask = (
                    (original_df['time_seconds'] >= section_df['time_seconds'].iloc[0]) &
                    (original_df['time_seconds'] <= section_df['time_seconds'].iloc[-1])
            )
            original_df.loc[section_mask, 'predicted_mass'] = section_mass
            original_df.loc[section_mask, 'is_valid_prediction'] = True
            original_df.loc[section_mask, 'aggregation_level'] = 'section'

        original_df = self._fill_gaps_between_sections_simple(original_df, all_section_results)

        valid_points = original_df['is_valid_prediction'].sum()
        total_points = len(original_df)

        logger.info(f"预测完成: {valid_points}/{total_points} 点有预测值")

        return {
            'predictions_df': original_df,
            'section_results': all_section_results,
            'trip_results': all_trip_results,
            'total_points': total_points,
            'valid_points': int(valid_points),
            'valid_percentage': float(valid_points / total_points * 100) if total_points > 0 else 0.0,
        }

    def _fill_gaps_between_sections_simple(self, df: pd.DataFrame,
                                           section_results: List[Dict]) -> pd.DataFrame:
        """简化版的区间间隙填充"""
        if len(section_results) == 0:
            return df

        section_results = sorted(section_results, key=lambda x: x['start_time'])

        first_section = section_results[0]
        if first_section['start_time'] > df['time_seconds'].iloc[0]:
            before_mask = df['time_seconds'] < first_section['start_time']
            df.loc[before_mask, 'predicted_mass'] = first_section['mass']
            df.loc[before_mask, 'is_valid_prediction'] = False
            df.loc[before_mask, 'aggregation_level'] = 'inherited'

        for i in range(len(section_results) - 1):
            curr = section_results[i]
            next_s = section_results[i + 1]

            if curr['end_time'] < next_s['start_time']:
                gap_mask = (
                        (df['time_seconds'] > curr['end_time']) &
                        (df['time_seconds'] < next_s['start_time'])
                )
                df.loc[gap_mask, 'predicted_mass'] = curr['mass']
                df.loc[gap_mask, 'is_valid_prediction'] = False
                df.loc[gap_mask, 'aggregation_level'] = 'inherited'

        last_section = section_results[-1]
        if last_section['end_time'] < df['time_seconds'].iloc[-1]:
            after_mask = df['time_seconds'] > last_section['end_time']
            df.loc[after_mask, 'predicted_mass'] = last_section['mass']
            df.loc[after_mask, 'is_valid_prediction'] = False
            df.loc[after_mask, 'aggregation_level'] = 'inherited'

        return df

    def _create_empty_results(self, df: pd.DataFrame) -> Dict[str, Any]:
        """创建空结果"""
        df['predicted_mass'] = np.nan
        df['trip_id'] = -1
        df['section_id'] = -1
        df['is_valid_prediction'] = False
        df['aggregation_level'] = 'none'
        return {
            'predictions_df': df,
            'section_results': [],
            'trip_results': [],
            'total_points': len(df),
            'valid_points': 0,
            'valid_percentage': 0.0,
        }

    def _split_trips_with_temp_mask(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """使用临时有效标记划分行程（用于预测阶段）"""
        logger.info("开始划分行程...")

        if 'speed_kmh' not in df.columns or 'time_seconds' not in df.columns:
            logger.error("缺少车速或时间列")
            return []

        df = df.sort_values('time_seconds').reset_index(drop=True)
        trips = []
        in_trip = False
        trip_start_idx = 0
        i = 0
        total_len = len(df)

        while i < total_len:
            # 使用临时有效标记或车速>0来判断是否在行程中
            if 'is_valid_temp' in df.columns:
                is_active = df.at[i, 'is_valid_temp'] == 1
            else:
                is_active = df.at[i, 'speed_kmh'] > 0

            if not in_trip:
                if is_active:
                    in_trip = True
                    trip_start_idx = i
                i += 1
            else:
                # 检查是否结束行程：车速为0且停车时间足够长
                if df.at[i, 'speed_kmh'] == 0:
                    zero_start = i
                    j = i + 1
                    while j < total_len and df.at[j, 'speed_kmh'] == 0:
                        j += 1

                    stop_duration = (j - zero_start) / self.config.SAMPLING_RATE

                    if stop_duration >= self.config.MIN_STOP_DURATION_FOR_TRIP_END:
                        trip_df = df.iloc[trip_start_idx:zero_start].copy()
                        if len(trip_df) > 0:
                            trips.append(trip_df)
                            logger.info(
                                f"  行程 {len(trips)}: {trip_df['time_seconds'].iloc[0]:.0f}s - {trip_df['time_seconds'].iloc[-1]:.0f}s ({len(trip_df)} 行)")
                        in_trip = False
                        i = j
                    else:
                        i = j
                else:
                    i += 1

        if in_trip:
            trip_df = df.iloc[trip_start_idx:total_len].copy()
            if len(trip_df) > 0:
                trips.append(trip_df)
                logger.info(
                    f"  行程 {len(trips)}: {trip_df['time_seconds'].iloc[0]:.0f}s - {trip_df['time_seconds'].iloc[-1]:.0f}s ({len(trip_df)} 行)")

        logger.info(f"行程划分完成: {len(trips)} 个行程")
        return trips

    def _fill_gaps_between_trips(self, df: pd.DataFrame,
                                 trip_results: List[Dict]) -> pd.DataFrame:
        """填充行程间的间隙"""
        if len(trip_results) == 0:
            return df

        trip_results = sorted(trip_results, key=lambda x: x['start_time'])

        first_trip = trip_results[0]
        if first_trip['start_time'] > df['time_seconds'].iloc[0]:
            before_mask = df['time_seconds'] < first_trip['start_time']
            df.loc[before_mask, 'predicted_mass'] = first_trip['mass']
            df.loc[before_mask, 'is_valid_prediction'] = False

        for i in range(len(trip_results) - 1):
            curr_trip = trip_results[i]
            next_trip = trip_results[i + 1]

            if curr_trip['end_time'] < next_trip['start_time']:
                gap_mask = (
                    (df['time_seconds'] > curr_trip['end_time']) &
                    (df['time_seconds'] < next_trip['start_time'])
                )
                df.loc[gap_mask, 'predicted_mass'] = curr_trip['mass']
                df.loc[gap_mask, 'is_valid_prediction'] = False

        last_trip = trip_results[-1]
        if last_trip['end_time'] < df['time_seconds'].iloc[-1]:
            after_mask = df['time_seconds'] > last_trip['end_time']
            df.loc[after_mask, 'predicted_mass'] = last_trip['mass']
            df.loc[after_mask, 'is_valid_prediction'] = False

        return df

    def save_model(self, filepath: str = 'physics_ml_model.pkl'):
        """保存模型"""
        model_data = {
            'ml_model': self.ml_model,
            'scaler': self.scaler,
            'config': self.config,
            'is_trained': self.is_trained,
            'training_metrics': self.training_metrics,
            'training_history': self.training_history,
            'residual_stats': self.residual_stats,
            'feature_names': self.feature_names,
            'prediction_mode': self.prediction_mode,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        logger.info(f"模型已保存到: {filepath}")

    def load_model(self, filepath: str = 'physics_ml_model.pkl'):
        """加载模型"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.ml_model = model_data['ml_model']
        self.scaler = model_data['scaler']
        self.config = model_data['config']
        if self.config.MIN_STOP_DURATION_FOR_SECTION_END <= self.config.MIN_STOP_DURATION_FOR_TRIP_END:
            logger.warning(
                f"加载模型：区间停车阈值({self.config.MIN_STOP_DURATION_FOR_SECTION_END}s) "
                f"≤ 行程阈值({self.config.MIN_STOP_DURATION_FOR_TRIP_END}s)，已修正为 600s"
            )
            self.config.MIN_STOP_DURATION_FOR_SECTION_END = 600.0
        if not hasattr(self.config, 'WINDOW_SIZE'):
            self.config.WINDOW_SIZE = getattr(self.config, 'SECONDARY_WINDOW_SIZE', 3.0)
        if not hasattr(self.config, 'MIN_WINDOW_POINTS'):
            self.config.MIN_WINDOW_POINTS = getattr(self.config, 'MIN_SECONDARY_WINDOW_POINTS', 15)
        self.config.SECONDARY_WINDOW_SIZE = self.config.WINDOW_SIZE
        self.config.MIN_SECONDARY_WINDOW_POINTS = self.config.MIN_WINDOW_POINTS
        self.is_trained = model_data['is_trained']
        self.training_metrics = model_data.get('training_metrics', {})
        self.training_history = model_data.get('training_history', {})
        self.residual_stats = model_data.get('residual_stats', {})
        self.feature_names = model_data.get('feature_names', [])
        self.prediction_mode = model_data.get('prediction_mode', 'absolute')
        logger.info(f"模型已从 {filepath} 加载 (预测模式: {self.prediction_mode})")


    def predict_with_comparison(self, df: pd.DataFrame) -> Dict[str, Any]:
        """预测新数据，按区间→行程→窗口层级聚合"""
        logger.info("=" * 60)
        logger.info(f"开始预测（含对比分析，区间聚合，模式={self.prediction_mode}）")
        logger.info("=" * 60)

        if not self.is_trained:
            raise ValueError("模型尚未训练")

        df_clean = self.cleaner.clean_data(df, is_training=False)
        original_df = df_clean.copy()

        temp_df = df_clean.copy()
        if 'force_n' in temp_df.columns:
            temp_force_condition = temp_df['force_n'] > self.config.MIN_FORCE
        elif 'motor_torque_nm' in temp_df.columns:
            torque_threshold = self.config.MIN_FORCE * self.config.TIRE_RADIUS / (
                    self.config.GEAR_RATIO * self.config.TRANSMISSION_EFFICIENCY)
            temp_force_condition = temp_df['motor_torque_nm'] > torque_threshold
        else:
            raise ValueError("缺少驱动力或扭矩列")

        temp_speed_condition = temp_df['speed_kmh'] > self.config.MIN_SPEED
        temp_df['is_valid_temp'] = (temp_speed_condition & temp_force_condition).astype(int)

        sections = self.physics_estimator.split_sections(temp_df)

        if len(sections) == 0:
            return self._create_empty_results_with_comparison(original_df)

        original_df['predicted_mass'] = np.nan
        original_df['physical_mass'] = np.nan
        original_df['ml_calibrated_mass'] = np.nan
        original_df['trip_id'] = -1
        original_df['section_id'] = -1
        original_df['is_valid_prediction'] = False
        original_df['improvement'] = np.nan
        original_df['aggregation_level'] = 'none'

        all_section_results = []
        all_trip_results = []
        all_window_results = []
        global_prev_mass = None
        global_trip_idx = 0

        for section_idx, section_df in enumerate(sections):
            logger.info(f"\n{'=' * 40}")
            logger.info(
                f"处理区间 {section_idx}: {section_df['time_seconds'].iloc[0]:.0f}s - {section_df['time_seconds'].iloc[-1]:.0f}s")
            logger.info(f"{'=' * 40}")

            section_trips = self.physics_estimator.split_trips_within_section(section_df)

            if len(section_trips) == 0:
                logger.info(f"  区间 {section_idx}: 无有效行程")
                continue

            section_trip_masses = []
            section_trip_window_counts = []
            section_trip_physical_masses = []
            section_trip_details = []

            for trip_idx_in_section, trip_df in enumerate(section_trips):
                if global_prev_mass is not None:
                    trip_df = self.physics_estimator.filter_valid_points(trip_df, global_prev_mass)
                else:
                    trip_df = self.physics_estimator.filter_valid_points(trip_df)

                primary_windows = self.physics_estimator.split_primary_windows(trip_df)

                if len(primary_windows) == 0:
                    logger.info(f"    行程 {global_trip_idx}: 无有效窗口")
                    trip_mask = (
                            (original_df['time_seconds'] >= trip_df['time_seconds'].iloc[0]) &
                            (original_df['time_seconds'] <= trip_df['time_seconds'].iloc[-1])
                    )
                    original_df.loc[trip_mask, 'trip_id'] = global_trip_idx
                    original_df.loc[trip_mask, 'section_id'] = section_idx
                    global_trip_idx += 1
                    continue

                trip_physical_masses = []
                trip_calibrated_masses = []
                trip_window_details = []

                for primary_window in primary_windows:
                    m_est_result = self.physics_estimator.estimate_primary_window_mass(primary_window)

                    if m_est_result['m_est'] is None:
                        continue

                    features = self.physics_estimator.extract_primary_window_features(
                        primary_window, m_est_result
                    )

                    if features is None:
                        continue

                    features_df = pd.DataFrame([features])
                    for fname in self.feature_names:
                        if fname not in features_df.columns:
                            features_df[fname] = 0.0

                    X_window = features_df[self.feature_names].values
                    X_window_scaled = self.scaler.transform(X_window)
                    m_physical = m_est_result['m_est']
                    m_calibrated, residual_pred = self._predict_calibrated_mass(m_physical, X_window_scaled)
                    window_start = float(primary_window['time_seconds'].iloc[0])
                    window_end = float(primary_window['time_seconds'].iloc[-1])

                    all_window_results.append({
                        'trip_idx': global_trip_idx,
                        'section_idx': section_idx,
                        'window_start': window_start,
                        'window_end': window_end,
                        'center_time': (window_start + window_end) / 2,
                        'm_physical': m_physical,
                        'm_ml': m_calibrated,
                        'm_residual_pred': residual_pred,
                        'n_secondary': m_est_result['n_secondary'],
                    })

                    trip_physical_masses.append(m_physical)
                    trip_calibrated_masses.append(m_calibrated)
                    trip_window_details.append({
                        'm_physical': m_physical,
                        'm_calibrated': m_calibrated,
                        'n_secondary': m_est_result['n_secondary'],
                        'improvement': abs(m_physical - m_calibrated) / max(m_physical,
                                                                            1) * 100 if m_physical > 0 else 0,
                    })

                if len(trip_calibrated_masses) == 0:
                    logger.info(f"    行程 {global_trip_idx}: 0/{len(primary_windows)} 个有效窗口")
                    trip_mask = (
                            (original_df['time_seconds'] >= trip_df['time_seconds'].iloc[0]) &
                            (original_df['time_seconds'] <= trip_df['time_seconds'].iloc[-1])
                    )
                    original_df.loc[trip_mask, 'trip_id'] = global_trip_idx
                    original_df.loc[trip_mask, 'section_id'] = section_idx
                    global_trip_idx += 1
                    continue

                trip_physical_final, n_after_trim, n_valid_windows = (
                    self.physics_estimator.aggregate_trip_mass_from_primary_windows(
                        trip_physical_masses, trim_reference=trip_calibrated_masses))
                trip_calibrated_final, _, _ = (
                    self.physics_estimator.aggregate_trip_mass_from_primary_windows(
                        trip_calibrated_masses, trim_reference=trip_calibrated_masses))
                improvement_pct = abs(trip_physical_final - trip_calibrated_final) / max(trip_physical_final, 1) * 100

                trip_result = {
                    'trip_idx': global_trip_idx,
                    'section_id': section_idx,
                    'mass': trip_calibrated_final,
                    'physical_mass': trip_physical_final,
                    'ml_mass': trip_calibrated_final,
                    'n_windows': n_valid_windows,
                    'n_windows_after_trim': n_after_trim,
                    'n_primary_windows': len(primary_windows),
                    'improvement_pct': improvement_pct,
                    'physical_std': float(np.std(trip_physical_masses)) if len(trip_physical_masses) > 1 else 0.0,
                    'ml_std': float(np.std(trip_calibrated_masses)) if len(trip_calibrated_masses) > 1 else 0.0,
                    'start_time': float(trip_df['time_seconds'].iloc[0]),
                    'end_time': float(trip_df['time_seconds'].iloc[-1]),
                    'window_details': trip_window_details,
                }
                all_trip_results.append(trip_result)

                section_trip_masses.append(trip_calibrated_final)
                section_trip_window_counts.append(len(trip_calibrated_masses))
                section_trip_physical_masses.append(trip_physical_final)
                section_trip_details.append(trip_result)

                trip_mask = (
                        (original_df['time_seconds'] >= trip_df['time_seconds'].iloc[0]) &
                        (original_df['time_seconds'] <= trip_df['time_seconds'].iloc[-1])
                )
                original_df.loc[trip_mask, 'trip_id'] = global_trip_idx
                original_df.loc[trip_mask, 'section_id'] = section_idx
                original_df.loc[trip_mask, 'predicted_mass'] = trip_calibrated_final
                original_df.loc[trip_mask, 'physical_mass'] = trip_physical_final
                original_df.loc[trip_mask, 'ml_calibrated_mass'] = trip_calibrated_final
                original_df.loc[trip_mask, 'is_valid_prediction'] = True
                original_df.loc[trip_mask, 'improvement'] = improvement_pct
                original_df.loc[trip_mask, 'aggregation_level'] = 'trip'

                global_prev_mass = trip_calibrated_final

                logger.info(
                    f"    行程 {global_trip_idx}: 物理={trip_physical_final:.0f}kg → ML={trip_calibrated_final:.0f}kg, "
                    f"改进={improvement_pct:.1f}%, "
                    f"截尾={n_after_trim}/{n_valid_windows} (窗口共{len(primary_windows)}个)")

                global_trip_idx += 1

            if len(section_trip_masses) == 0:
                logger.info(f"  区间 {section_idx}: 无有效行程，跳过区间聚合")
                continue

            section_mass = self._aggregate_section_mass(
                section_trip_masses,
                section_trip_window_counts,
                method='median'
            )

            section_physical_mass = float(np.median(section_trip_physical_masses))
            section_improvement = abs(section_physical_mass - section_mass) / max(section_physical_mass, 1) * 100

            section_result = {
                'section_idx': section_idx,
                'mass': section_mass,
                'physical_mass': section_physical_mass,
                'ml_mass': section_mass,
                'n_trips': len(section_trip_masses),
                'n_total_windows': sum(section_trip_window_counts),
                'improvement_pct': section_improvement,
                'trip_masses': section_trip_masses,
                'trip_window_counts': section_trip_window_counts,
                'physical_std': float(np.std(section_trip_physical_masses)) if len(
                    section_trip_physical_masses) > 1 else 0.0,
                'ml_std': float(np.std(section_trip_masses)) if len(section_trip_masses) > 1 else 0.0,
                'start_time': float(section_df['time_seconds'].iloc[0]),
                'end_time': float(section_df['time_seconds'].iloc[-1]),
                'trip_details': section_trip_details,
                'aggregation_method': 'median',
            }
            all_section_results.append(section_result)

            section_mask = (
                    (original_df['time_seconds'] >= section_df['time_seconds'].iloc[0]) &
                    (original_df['time_seconds'] <= section_df['time_seconds'].iloc[-1])
            )
            original_df.loc[section_mask, 'predicted_mass'] = section_mass
            original_df.loc[section_mask, 'physical_mass'] = section_physical_mass
            original_df.loc[section_mask, 'ml_calibrated_mass'] = section_mass
            original_df.loc[section_mask, 'is_valid_prediction'] = True
            original_df.loc[section_mask, 'improvement'] = section_improvement
            original_df.loc[section_mask, 'aggregation_level'] = 'section'

            logger.info(f"  区间 {section_idx} 聚合结果: {len(section_trip_masses)}个行程, "
                        f"物理={section_physical_mass:.0f}kg → ML={section_mass:.0f}kg, "
                        f"改进={section_improvement:.1f}%")

            logger.info(f"  区间内行程质量分布: {[f'{m:.0f}' for m in section_trip_masses]}")

        original_df = self._fill_gaps_between_sections(original_df, all_section_results)

        valid_points = original_df['is_valid_prediction'].sum()
        total_points = len(original_df)

        logger.info(f"\n{'=' * 60}")
        logger.info(f"预测完成: {valid_points}/{total_points} 点有预测值 ({valid_points / total_points * 100:.1f}%)")
        logger.info(f"区间总数: {len(all_section_results)}")
        logger.info(f"行程总数: {len(all_trip_results)}")
        logger.info(f"{'=' * 60}")

        return {
            'predictions_df': original_df,
            'section_results': all_section_results,
            'trip_results': all_trip_results,
            'window_results': all_window_results,
            'total_points': total_points,
            'valid_points': int(valid_points),
            'valid_percentage': float(valid_points / total_points * 100) if total_points > 0 else 0.0,
        }

    def _create_empty_results_with_comparison(self, df: pd.DataFrame) -> Dict[str, Any]:
        """创建空结果（带区间字段）"""
        df['predicted_mass'] = np.nan
        df['physical_mass'] = np.nan
        df['ml_calibrated_mass'] = np.nan
        df['trip_id'] = -1
        df['section_id'] = -1
        df['is_valid_prediction'] = False
        df['improvement'] = np.nan
        df['aggregation_level'] = 'none'
        return {
            'predictions_df': df,
            'section_results': [],
            'trip_results': [],
            'window_results': [],
            'total_points': len(df),
            'valid_points': 0,
            'valid_percentage': 0.0,
        }

    def _fill_gaps_between_trips_with_comparison(self, df: pd.DataFrame,
                                                 trip_results: List[Dict]) -> pd.DataFrame:
        """填充行程间的间隙（带对比字段）"""
        if len(trip_results) == 0:
            return df

        trip_results = sorted(trip_results, key=lambda x: x['start_time'])

        first_trip = trip_results[0]
        if first_trip['start_time'] > df['time_seconds'].iloc[0]:
            before_mask = df['time_seconds'] < first_trip['start_time']
            df.loc[before_mask, 'predicted_mass'] = first_trip['ml_mass']
            df.loc[before_mask, 'physical_mass'] = first_trip['physical_mass']
            df.loc[before_mask, 'ml_calibrated_mass'] = first_trip['ml_mass']
            df.loc[before_mask, 'is_valid_prediction'] = False

        for i in range(len(trip_results) - 1):
            curr_trip = trip_results[i]
            next_trip = trip_results[i + 1]

            if curr_trip['end_time'] < next_trip['start_time']:
                gap_mask = (
                        (df['time_seconds'] > curr_trip['end_time']) &
                        (df['time_seconds'] < next_trip['start_time'])
                )
                df.loc[gap_mask, 'predicted_mass'] = curr_trip['ml_mass']
                df.loc[gap_mask, 'physical_mass'] = curr_trip['physical_mass']
                df.loc[gap_mask, 'ml_calibrated_mass'] = curr_trip['ml_mass']
                df.loc[gap_mask, 'is_valid_prediction'] = False

        last_trip = trip_results[-1]
        if last_trip['end_time'] < df['time_seconds'].iloc[-1]:
            after_mask = df['time_seconds'] > last_trip['end_time']
            df.loc[after_mask, 'predicted_mass'] = last_trip['ml_mass']
            df.loc[after_mask, 'physical_mass'] = last_trip['physical_mass']
            df.loc[after_mask, 'ml_calibrated_mass'] = last_trip['ml_mass']
            df.loc[after_mask, 'is_valid_prediction'] = False

        return df

    def _aggregate_section_mass(self, trip_masses: List[float],
                                trip_window_counts: List[int],
                                method: str = 'median') -> float:
        """聚合区间质量"""
        if len(trip_masses) == 0:
            return None

        if method == 'median':
            return float(np.median(trip_masses))

        elif method == 'weighted_median':
            # 按行程的窗口数加权
            if len(trip_masses) == 1:
                return float(trip_masses[0])

            # 创建加权样本
            weighted_samples = []
            for mass, count in zip(trip_masses, trip_window_counts):
                weighted_samples.extend([mass] * max(1, count))

            return float(np.median(weighted_samples))

        elif method == 'mean':
            return float(np.mean(trip_masses))

        elif method == 'trimmed_median':
            # 去掉最高和最低的20%行程后取中位数
            if len(trip_masses) >= 5:
                sorted_masses = sorted(trip_masses)
                trim_n = max(1, int(len(sorted_masses) * 0.2))
                trimmed = sorted_masses[trim_n:-trim_n]
                return float(np.median(trimmed))
            else:
                return float(np.median(trip_masses))

        else:
            return float(np.median(trip_masses))

    def _fill_gaps_between_sections(self, df: pd.DataFrame,
                                    section_results: List[Dict]) -> pd.DataFrame:
        """填充区间间的间隙（长时间停车）"""
        if len(section_results) == 0:
            return df

        # 按开始时间排序
        section_results = sorted(section_results, key=lambda x: x['start_time'])

        # 第一个区间之前的数据
        first_section = section_results[0]
        if first_section['start_time'] > df['time_seconds'].iloc[0]:
            before_mask = df['time_seconds'] < first_section['start_time']
            df.loc[before_mask, 'predicted_mass'] = first_section['ml_mass']
            df.loc[before_mask, 'physical_mass'] = first_section['physical_mass']
            df.loc[before_mask, 'ml_calibrated_mass'] = first_section['ml_mass']
            df.loc[before_mask, 'is_valid_prediction'] = False
            df.loc[before_mask, 'aggregation_level'] = 'inherited'
            df.loc[before_mask, 'section_id'] = -1

        # 区间之间的间隙
        for i in range(len(section_results) - 1):
            curr_section = section_results[i]
            next_section = section_results[i + 1]

            if curr_section['end_time'] < next_section['start_time']:
                gap_mask = (
                        (df['time_seconds'] > curr_section['end_time']) &
                        (df['time_seconds'] < next_section['start_time'])
                )
                df.loc[gap_mask, 'predicted_mass'] = curr_section['ml_mass']
                df.loc[gap_mask, 'physical_mass'] = curr_section['physical_mass']
                df.loc[gap_mask, 'ml_calibrated_mass'] = curr_section['ml_mass']
                df.loc[gap_mask, 'is_valid_prediction'] = False
                df.loc[gap_mask, 'aggregation_level'] = 'inherited'
                df.loc[gap_mask, 'section_id'] = -1

        # 最后一个区间之后的数据
        last_section = section_results[-1]
        if last_section['end_time'] < df['time_seconds'].iloc[-1]:
            after_mask = df['time_seconds'] > last_section['end_time']
            df.loc[after_mask, 'predicted_mass'] = last_section['ml_mass']
            df.loc[after_mask, 'physical_mass'] = last_section['physical_mass']
            df.loc[after_mask, 'ml_calibrated_mass'] = last_section['ml_mass']
            df.loc[after_mask, 'is_valid_prediction'] = False
            df.loc[after_mask, 'aggregation_level'] = 'inherited'
            df.loc[after_mask, 'section_id'] = -1

        return df
