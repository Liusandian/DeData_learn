"""
Pocket4 Sensor 线性噪声模型标定脚本
====================================
针对 14-bit Raw 数据，完成从 BLC 标定到线性噪声模型参数输出的完整流程。

噪声模型: σ²(y) = α · μ(y) + β
  - α: 系统增益 (DN/e⁻)，对应散粒噪声
  - β: 读出噪声方差 (DN²)

数据采集矩阵:
  - 5 个增益档位: 1 / 2 / 4 / 8 / 1600
  - 9 个亮度级: dark / 12.5% / 25% / 37.5% / 50% / 62.5% / 75% / 87.5% / 100%
  - 场景: 4×6 标准色卡，位于画面九宫格中心 1/9 区域

用法:
    # 使用真实数据标定
    python pocket4_noise_calibration.py --data_root /path/to/raw_data --raw_shape 3024 4032

    # 使用模拟数据验证流程
    python pocket4_noise_calibration.py --simulate

    # 指定 Bayer pattern 和输出目录
    python pocket4_noise_calibration.py --data_root /path/to/raw_data --raw_shape 3024 4032 \\
        --bayer RGGB --output_dir ./calibration_results
"""

import numpy as np
import os
import json
import argparse
from datetime import datetime
from scipy import stats as sp_stats
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适用于服务器/无GUI环境
import matplotlib.pyplot as plt

# ─── 尝试导入可选依赖 ───
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import rawpy
    HAS_RAWPY = True
except ImportError:
    HAS_RAWPY = False


# ============================================================
# 全局配置常量
# ============================================================

BIT_DEPTH = 14                        # Raw 位深度
MAX_DN = (2 ** BIT_DEPTH) - 1         # 14-bit 最大 DN 值 = 16383
BAYER_PATTERNS = {                    # Bayer 拆分行列偏移 (row_offset, col_offset)
    'RGGB': {'R': (0, 0), 'Gr': (0, 1), 'Gb': (1, 0), 'B': (1, 1)},
    'BGGR': {'B': (0, 0), 'Gb': (0, 1), 'Gr': (1, 0), 'R': (1, 1)},
    'GRBG': {'Gr': (0, 0), 'R': (0, 1), 'B': (1, 0), 'Gb': (1, 1)},
    'GBRG': {'Gb': (0, 0), 'B': (0, 1), 'R': (1, 0), 'Gr': (1, 1)},
}
GAIN_LIST = [1, 2, 4, 8, 1600]       # 增益档位列表
BRIGHTNESS_LIST = [                    # 亮度级列表（不含黑帧）
    12.5, 25, 37.5, 50, 62.5, 75, 87.5, 100
]
CHANNEL_NAMES = ['R', 'Gr', 'Gb', 'B']
CHANNEL_COLORS = {'R': 'red', 'Gr': 'green', 'Gb': 'limegreen', 'B': 'blue'}

# 色卡布局: 4 行 × 6 列，标准 24 色色卡
CHART_ROWS = 4
CHART_COLS = 6
# 灰阶 patch 索引（底行第 19~24 号，0-indexed 为 18~23）
GRAY_PATCH_INDICES = [18, 19, 20, 21, 22, 23]

# ROI 边缘裁切比例：取 patch 中心 60% 区域（上下左右各裁 20%）
ROI_MARGIN = 0.20

# 饱和阈值：超过满量程 90% 的数据点视为饱和，不参与拟合
SATURATION_RATIO = 0.90


# ============================================================
# 配置 Matplotlib 中文字体
# ============================================================
def _configure_chinese_fonts():
    """配置 Matplotlib 中文字体，防止标题/标签乱码。"""
    from matplotlib import font_manager
    preferred = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
        "Source Han Sans CN", "PingFang SC", "WenQuanYi Micro Hei",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False

_configure_chinese_fonts()


# ============================================================
# 第一部分：Raw 数据加载
# ============================================================

def load_single_raw(filepath, raw_shape, bit_depth=14):
    """
    加载单帧 Raw 文件，返回 float64 数组（DN 值，未归一化）。

    支持格式:
      - .npy: numpy 数组直接加载
      - .raw / .RAW: 纯二进制文件，按 uint16 读取并 reshape
      - .dng / .DNG 等: 通过 rawpy 读取

    参数:
        filepath:  Raw 文件路径
        raw_shape: (H, W) 元组，仅 .raw 格式需要
        bit_depth: 位深度，默认 14

    返回:
        frame: np.ndarray, shape=(H, W), dtype=float64, 单位 DN
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.npy':
        # ── numpy 格式直接加载 ──
        frame = np.load(filepath).astype(np.float64)

    elif ext == '.raw':
        # ── 纯二进制 Raw，按 uint16 读入 ──
        if raw_shape is None:
            raise ValueError(f"加载 .raw 文件需要指定 --raw_shape，文件: {filepath}")
        # 14-bit 数据通常存储为 uint16
        dtype = np.uint16 if bit_depth <= 16 else np.uint32
        frame = np.fromfile(filepath, dtype=dtype).astype(np.float64)
        frame = frame.reshape(raw_shape)

    elif ext in ('.dng', '.arw', '.cr2', '.nef'):
        # ── 通过 rawpy 读取相机 Raw 格式 ──
        if not HAS_RAWPY:
            raise ImportError("加载 DNG/ARW 等格式需要 rawpy: pip install rawpy")
        with rawpy.imread(filepath) as raw:
            frame = raw.raw_image_visible.astype(np.float64)
    else:
        raise ValueError(f"不支持的文件格式: {ext}")

    return frame


def load_multi_frames(folder_path, raw_shape, bit_depth=14, max_frames=None):
    """
    加载指定文件夹下的所有 Raw 帧，返回多帧数组。

    参数:
        folder_path: 存放同一条件下多帧 Raw 的文件夹
        raw_shape:   (H, W) 元组
        bit_depth:   位深度
        max_frames:  最多加载帧数，None 表示全部

    返回:
        frames: np.ndarray, shape=(N, H, W), dtype=float64, 单位 DN
    """
    supported_ext = ('.npy', '.raw', '.dng', '.arw', '.cr2', '.nef')
    # 获取文件列表并排序，保证帧顺序一致
    files = sorted([
        f for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in supported_ext
    ])
    if max_frames is not None:
        files = files[:max_frames]
    if len(files) == 0:
        raise FileNotFoundError(f"文件夹 {folder_path} 中未找到 Raw 文件")

    frames = []
    for fname in files:
        fpath = os.path.join(folder_path, fname)
        frame = load_single_raw(fpath, raw_shape, bit_depth)
        frames.append(frame)

    frames = np.array(frames)
    print(f"  [加载] {folder_path}: {len(frames)} 帧, "
          f"shape={frames[0].shape}, 范围=[{frames.min():.1f}, {frames.max():.1f}] DN")
    return frames


# ============================================================
# 第二部分：Bayer 通道拆分
# ============================================================

def split_bayer_channels(raw_frame, bayer_pattern='RGGB'):
    """
    将完整 Raw 帧拆分为 R/Gr/Gb/B 四个子通道。

    原理:
      Bayer CFA 排列中，相邻 2×2 像素块包含 R/Gr/Gb/B 各一个。
      按行列奇偶关系提取各通道，得到 4 个 (H/2, W/2) 的子图像。

    参数:
        raw_frame:     np.ndarray, shape=(H, W), 完整 Raw 帧
        bayer_pattern: 字符串，Bayer 排列模式

    返回:
        channels: dict, {'R': array, 'Gr': array, 'Gb': array, 'B': array}
                  每个 array shape = (H/2, W/2)
    """
    if bayer_pattern not in BAYER_PATTERNS:
        raise ValueError(f"不支持的 Bayer pattern: {bayer_pattern}，"
                         f"支持: {list(BAYER_PATTERNS.keys())}")

    offsets = BAYER_PATTERNS[bayer_pattern]
    channels = {}
    for ch_name, (row_off, col_off) in offsets.items():
        # 从 (row_off, col_off) 开始，每隔 2 行/列取一个像素
        channels[ch_name] = raw_frame[row_off::2, col_off::2].copy()

    return channels


def split_bayer_multi_frames(frames, bayer_pattern='RGGB'):
    """
    对多帧 Raw 数据批量做 Bayer 通道拆分。

    参数:
        frames:        np.ndarray, shape=(N, H, W)
        bayer_pattern: Bayer 排列模式

    返回:
        channels_all: dict, {'R': (N, H/2, W/2), 'Gr': ..., 'Gb': ..., 'B': ...}
    """
    channels_all = {ch: [] for ch in CHANNEL_NAMES}
    for i in range(frames.shape[0]):
        ch_dict = split_bayer_channels(frames[i], bayer_pattern)
        for ch_name in CHANNEL_NAMES:
            channels_all[ch_name].append(ch_dict[ch_name])

    # 转成 numpy 数组: (N, H/2, W/2)
    for ch_name in CHANNEL_NAMES:
        channels_all[ch_name] = np.array(channels_all[ch_name])

    return channels_all


# ============================================================
# 第三部分：BLC（黑电平）逐通道标定
# ============================================================

def calibrate_blc_per_channel(dark_frames, bayer_pattern='RGGB'):
    """
    从黑帧数据中标定每个 Bayer 通道的黑电平值（Black Level）。

    原理:
      黑帧 = 遮光条件下采集的图像，像素值 = 黑电平 + 暗电流噪声。
      取多帧黑帧的逐通道均值作为该通道的黑电平估计。
      同时计算黑帧方差，用于后续与拟合得到的 β 做交叉验证。

    关键点:
      - 必须逐增益标定，不可跨增益共享（黑电平随增益变化）
      - 必须逐 Bayer 通道标定（R/Gr/Gb/B 的读出电路不同，BL 不同）
      - 黑帧方差 ≈ 读出噪声方差 β（可用于验证拟合结果）

    参数:
        dark_frames:   np.ndarray, shape=(N, H, W), 多帧黑帧数据（DN 值）
        bayer_pattern: Bayer 排列模式

    返回:
        blc_dict:     dict, {'R': float, 'Gr': float, 'Gb': float, 'B': float}
                      每个通道的黑电平均值 (DN)
        dark_var_dict: dict, 每个通道的黑帧方差 (DN²)，用于交叉验证
    """
    # 先拆通道，再统计 —— 这是正确的顺序
    channels = split_bayer_multi_frames(dark_frames, bayer_pattern)

    blc_dict = {}
    dark_var_dict = {}
    print(f"  [BLC 标定] Bayer={bayer_pattern}")
    for ch_name in CHANNEL_NAMES:
        ch_data = channels[ch_name]  # shape: (N, H/2, W/2)
        # 所有帧、所有像素的均值 → 黑电平
        blc_val = np.mean(ch_data)
        # 时域方差的空域均值 → 暗噪声方差（用于验证 β）
        # 先逐像素算时域方差，再取空域均值
        if ch_data.shape[0] >= 2:
            temporal_var = np.var(ch_data, axis=0, ddof=1)  # (H/2, W/2)
            var_val = np.mean(temporal_var)
        else:
            # 只有 1 帧时，只能用空域方差（包含 DSNU，精度较低）
            var_val = np.var(ch_data)

        blc_dict[ch_name] = blc_val
        dark_var_dict[ch_name] = var_val
        print(f"    {ch_name}: BLC = {blc_val:.2f} DN, 暗噪声方差 = {var_val:.4f} DN²")

    return blc_dict, dark_var_dict


# ============================================================
# 第四部分：ROI 选取（色卡 patch 定位）
# ============================================================

def compute_patch_rois(image_h, image_w, bayer_pattern='RGGB',
                       chart_rows=CHART_ROWS, chart_cols=CHART_COLS,
                       margin=ROI_MARGIN):
    """
    计算色卡各 patch 在 Bayer 子通道坐标系中的 ROI 坐标。

    场景假设:
      - 色卡位于画面九宫格中心 1/9 区域
      - 色卡为 4 行 × 6 列标准色卡（如 X-Rite ColorChecker）
      - 每个 patch 取中心 60% 区域（margin=0.20），避开 patch 边缘串扰

    坐标映射:
      Raw 坐标 → Bayer 子通道坐标需要除以 2（因为 2×2 下采样）
      所以这里直接在子通道的 (H/2, W/2) 空间上计算 ROI

    参数:
        image_h, image_w: 原始 Raw 的高、宽（全分辨率）
        bayer_pattern:     Bayer 模式（此函数中暂未使用，预留）
        chart_rows:        色卡行数，默认 4
        chart_cols:        色卡列数，默认 6
        margin:            边缘裁切比例，默认 0.20（取中心 60%）

    返回:
        rois: list of dict, 每个 dict 包含:
              {'patch_idx': int, 'row': int, 'col': int,
               'y1': int, 'y2': int, 'x1': int, 'x2': int,
               'is_gray': bool}
              坐标是 Bayer 子通道空间的像素坐标
    """
    # ── 子通道尺寸 = 原始尺寸 / 2 ──
    sub_h = image_h // 2
    sub_w = image_w // 2

    # ── 色卡在中心 1/9 区域 ──
    chart_y1 = sub_h // 3
    chart_y2 = 2 * sub_h // 3
    chart_x1 = sub_w // 3
    chart_x2 = 2 * sub_w // 3

    chart_h = chart_y2 - chart_y1  # 色卡区域高度（子通道像素）
    chart_w = chart_x2 - chart_x1  # 色卡区域宽度（子通道像素）

    # ── 每个 patch 的尺寸 ──
    patch_h = chart_h / chart_rows
    patch_w = chart_w / chart_cols

    rois = []
    patch_idx = 0
    for row in range(chart_rows):
        for col in range(chart_cols):
            # patch 中心坐标
            cy = chart_y1 + (row + 0.5) * patch_h
            cx = chart_x1 + (col + 0.5) * patch_w

            # 取中心 (1 - 2*margin) 区域
            roi_h = patch_h * (1 - 2 * margin)
            roi_w = patch_w * (1 - 2 * margin)

            y1 = int(cy - roi_h / 2)
            y2 = int(cy + roi_h / 2)
            x1 = int(cx - roi_w / 2)
            x2 = int(cx + roi_w / 2)

            # 边界保护
            y1 = max(0, y1)
            y2 = min(sub_h, y2)
            x1 = max(0, x1)
            x2 = min(sub_w, x2)

            # 标记是否是灰阶 patch（底行 19~24 号）
            is_gray = patch_idx in GRAY_PATCH_INDICES

            rois.append({
                'patch_idx': patch_idx,
                'row': row,
                'col': col,
                'y1': y1, 'y2': y2,
                'x1': x1, 'x2': x2,
                'is_gray': is_gray,
            })
            patch_idx += 1

    # 统计信息
    n_gray = sum(1 for r in rois if r['is_gray'])
    print(f"  [ROI] 共 {len(rois)} 个 patch, 其中 {n_gray} 个灰阶 patch, "
          f"ROI 尺寸约 {int(patch_h*(1-2*margin))}×{int(patch_w*(1-2*margin))} 像素(子通道)")
    return rois


# ============================================================
# 第五部分：均值-方差计算（时域统计）
# ============================================================

def compute_mean_variance_temporal(channel_frames, blc_value, rois,
                                   saturation_dn=None):
    """
    对单个 Bayer 通道，使用时域统计计算各 ROI 的均值和方差。

    原理:
      时域统计（temporal statistics）= 同一像素在多帧间的统计
        μ_pixel = mean(frames, axis=0)   → 逐像素时域均值
        σ²_pixel = var(frames, axis=0)   → 逐像素时域方差
      然后对 ROI 内的 μ 和 σ² 分别取空域均值。

    关键点:
      - 时域方差天然不包含 PRNU（像素响应非均匀性），无需额外修正
      - 必须先减去 BLC 再计算均值（方差不受常数偏移影响，但均值必须准确）
      - 饱和像素（> 90% 满量程）的 ROI 需标记并在拟合时剔除

    参数:
        channel_frames: np.ndarray, shape=(N, H/2, W/2), 单通道多帧数据（DN 值）
        blc_value:      float, 该通道的黑电平值 (DN)
        rois:           list of dict, ROI 坐标列表
        saturation_dn:  float, 饱和 DN 阈值，None 则自动设为 MAX_DN * 0.9

    返回:
        results: list of dict, 每个 dict 包含:
                 {'patch_idx': int, 'mean': float, 'var': float,
                  'is_saturated': bool, 'is_gray': bool, 'n_pixels': int}
    """
    if saturation_dn is None:
        saturation_dn = MAX_DN * SATURATION_RATIO

    n_frames = channel_frames.shape[0]
    if n_frames < 2:
        raise ValueError(f"时域统计至少需要 2 帧，当前只有 {n_frames} 帧。"
                         f"如果只有单帧，请使用帧差法。")

    # ── 逐像素时域均值和方差 ──
    # 注意: 先在 DN 空间计算，再减 BLC
    pixel_mean = np.mean(channel_frames, axis=0)    # (H/2, W/2), DN 域
    pixel_var = np.var(channel_frames, axis=0, ddof=1)  # (H/2, W/2), 无偏估计

    results = []
    for roi in rois:
        y1, y2, x1, x2 = roi['y1'], roi['y2'], roi['x1'], roi['x2']

        # 提取 ROI 区域
        roi_mean = pixel_mean[y1:y2, x1:x2]
        roi_var = pixel_var[y1:y2, x1:x2]

        # 计算 ROI 的统计量
        # 均值: 先取空域均值，再减 BLC → 纯信号均值
        mu = np.mean(roi_mean) - blc_value
        # 方差: 取 ROI 内时域方差的空域均值
        var = np.mean(roi_var)

        # 判断是否饱和: ROI 内均值（未减 BLC）超过饱和阈值
        is_saturated = np.mean(roi_mean) > saturation_dn

        n_pixels = (y2 - y1) * (x2 - x1)

        results.append({
            'patch_idx': roi['patch_idx'],
            'mean': mu,          # 信号均值 (DN), 已减 BLC
            'var': var,          # 时域方差 (DN²)
            'is_saturated': is_saturated,
            'is_gray': roi['is_gray'],
            'n_pixels': n_pixels,
        })

    return results


def compute_mean_variance_frame_diff(channel_frames, blc_value, rois,
                                     saturation_dn=None):
    """
    帧差法计算均值和方差（适用于只有 2 帧的场景）。

    原理:
      两帧相减: diff = frame1 - frame2
      Var(diff) = Var(frame1) + Var(frame2) = 2·σ²
      所以: σ² = Var(diff) / 2

    此方法同样能消除 PRNU（PRNU 是帧间固定的，相减后抵消）。

    参数:
        channel_frames: np.ndarray, shape=(N, H/2, W/2), 至少 2 帧
        blc_value:      float, 黑电平 (DN)
        rois:           list of dict
        saturation_dn:  float, 饱和阈值

    返回:
        results: list of dict (同 compute_mean_variance_temporal)
    """
    if saturation_dn is None:
        saturation_dn = MAX_DN * SATURATION_RATIO

    if channel_frames.shape[0] < 2:
        raise ValueError("帧差法至少需要 2 帧")

    # ── 取前两帧做差 ──
    frame1 = channel_frames[0].astype(np.float64)
    frame2 = channel_frames[1].astype(np.float64)
    diff = frame1 - frame2

    # 均值用两帧平均
    pixel_mean = (frame1 + frame2) / 2.0

    results = []
    for roi in rois:
        y1, y2, x1, x2 = roi['y1'], roi['y2'], roi['x1'], roi['x2']

        roi_diff = diff[y1:y2, x1:x2]
        roi_mean = pixel_mean[y1:y2, x1:x2]

        # 帧差方差除以 2 即为单帧时域方差
        var = np.var(roi_diff) / 2.0
        mu = np.mean(roi_mean) - blc_value

        is_saturated = np.mean(roi_mean) > saturation_dn

        results.append({
            'patch_idx': roi['patch_idx'],
            'mean': mu,
            'var': var,
            'is_saturated': is_saturated,
            'is_gray': roi['is_gray'],
            'n_pixels': (y2 - y1) * (x2 - x1),
        })

    return results


# ============================================================
# 第六部分：线性回归拟合
# ============================================================

def fit_linear_noise_model(data_points, use_gray_only=False):
    """
    对一组 (均值, 方差) 数据点做线性回归，拟合噪声模型 σ² = α·μ + β。

    数据清洗规则:
      1. 剔除饱和点 (is_saturated=True)
      2. 剔除 μ < 0 的点（减 BLC 后信号为负，说明暗于黑电平，无效）
      3. 剔除 3σ 离群点（基于初次拟合的残差）
      4. 可选: 仅使用灰阶 patch (use_gray_only=True)

    参数:
        data_points:   list of dict, 每个 dict 包含 'mean', 'var',
                       'is_saturated', 'is_gray'
        use_gray_only: bool, 是否仅使用灰阶 patch

    返回:
        result: dict 包含:
                'alpha': float, 斜率 (DN/e⁻)
                'beta':  float, 截距 (DN²), 读出噪声方差
                'r_squared': float, R² 拟合优度
                'n_points':  int, 参与拟合的有效数据点数
                'mu_list':   list, 有效数据点的均值列表
                'var_list':  list, 有效数据点的方差列表
                'outlier_count': int, 被剔除的离群点数
    """
    # ── 数据清洗: 过滤饱和 & 负信号 ──
    filtered = []
    for dp in data_points:
        if dp['is_saturated']:
            continue
        if dp['mean'] <= 0:
            continue
        if use_gray_only and not dp['is_gray']:
            continue
        filtered.append(dp)

    if len(filtered) < 3:
        print(f"    [警告] 有效数据点不足 ({len(filtered)} < 3), 无法拟合")
        return None

    mu_arr = np.array([dp['mean'] for dp in filtered])
    var_arr = np.array([dp['var'] for dp in filtered])

    # ── 第一轮拟合（含离群点）──
    slope, intercept, r_value, p_value, std_err = sp_stats.linregress(mu_arr, var_arr)

    # ── 剔除 3σ 离群点 ──
    predicted = slope * mu_arr + intercept
    residuals = var_arr - predicted
    res_std = np.std(residuals)
    if res_std > 0:
        inlier_mask = np.abs(residuals) < 3 * res_std
    else:
        inlier_mask = np.ones(len(mu_arr), dtype=bool)

    outlier_count = int(np.sum(~inlier_mask))
    mu_clean = mu_arr[inlier_mask]
    var_clean = var_arr[inlier_mask]

    if len(mu_clean) < 3:
        print(f"    [警告] 剔除离群点后数据不足 ({len(mu_clean)} < 3)")
        return None

    # ── 第二轮拟合（剔除离群点后）──
    slope, intercept, r_value, p_value, std_err = sp_stats.linregress(mu_clean, var_clean)

    result = {
        'alpha': slope,
        'beta': intercept,
        'r_squared': r_value ** 2,
        'n_points': len(mu_clean),
        'mu_list': mu_clean.tolist(),
        'var_list': var_clean.tolist(),
        'outlier_count': outlier_count,
        'std_err': std_err,
    }
    return result


# ============================================================
# 第七部分：参数验证
# ============================================================

def validate_calibration(all_results, dark_var_all):
    """
    验证标定结果的合理性，输出验证报告。

    验证项:
      1. R² 检查: 应 > 0.99
      2. α 比例检查: α 应随增益线性增长 (α_g ≈ g/g0 * α_g0)
      3. β 比例检查: β 应随增益平方增长 (β_g ≈ (g/g0)² * β_g0，模拟增益)
      4. β 与黑帧方差交叉验证: β ≈ dark_var (偏差 < 20%)

    参数:
        all_results:  dict, {gain: {channel: fit_result}}
        dark_var_all: dict, {gain: {channel: float}}, 黑帧方差

    返回:
        report: list of str, 验证报告文本行
    """
    report = []
    report.append("=" * 70)
    report.append("标定结果验证报告")
    report.append("=" * 70)

    # ── 1. R² 检查 ──
    report.append("\n[1] R² 拟合优度检查 (阈值 > 0.99):")
    for gain in GAIN_LIST:
        if gain not in all_results:
            continue
        for ch in CHANNEL_NAMES:
            res = all_results[gain].get(ch)
            if res is None:
                report.append(f"  Gain={gain:>5}, {ch:>2}: 拟合失败")
                continue
            r2 = res['r_squared']
            status = "PASS" if r2 > 0.99 else ("WARN" if r2 > 0.95 else "FAIL")
            report.append(f"  Gain={gain:>5}, {ch:>2}: R² = {r2:.6f}  [{status}]")

    # ── 2. α 比例检查 ──
    report.append("\n[2] α 增益比例检查 (α_g / α_1 ≈ g):")
    base_gain = GAIN_LIST[0]
    if base_gain in all_results:
        for ch in CHANNEL_NAMES:
            base_res = all_results[base_gain].get(ch)
            if base_res is None:
                continue
            alpha_base = base_res['alpha']
            if alpha_base <= 0:
                continue
            report.append(f"  通道 {ch}: α(Gain=1) = {alpha_base:.6f}")
            for gain in GAIN_LIST[1:]:
                res = all_results.get(gain, {}).get(ch)
                if res is None:
                    continue
                ratio = res['alpha'] / alpha_base
                expected = gain / base_gain
                deviation = abs(ratio - expected) / expected * 100
                status = "OK" if deviation < 20 else "偏差较大"
                report.append(
                    f"    Gain={gain:>5}: α={res['alpha']:.6f}, "
                    f"比值={ratio:.2f} (预期={expected:.0f}, 偏差={deviation:.1f}%) [{status}]"
                )

    # ── 3. β 与黑帧方差交叉验证 ──
    report.append("\n[3] β 与黑帧方差交叉验证 (偏差 < 20%):")
    for gain in GAIN_LIST:
        if gain not in all_results:
            continue
        for ch in CHANNEL_NAMES:
            res = all_results[gain].get(ch)
            if res is None:
                continue
            dark_v = dark_var_all.get(gain, {}).get(ch, None)
            if dark_v is None:
                continue
            beta = res['beta']
            if dark_v > 0:
                deviation = abs(beta - dark_v) / dark_v * 100
            else:
                deviation = float('inf')
            status = "PASS" if deviation < 20 else "WARN"
            report.append(
                f"  Gain={gain:>5}, {ch:>2}: β={beta:.4f}, "
                f"暗帧方差={dark_v:.4f}, 偏差={deviation:.1f}% [{status}]"
            )

    return report


# ============================================================
# 第八部分：可视化（PTC 曲线绘制）
# ============================================================

def plot_ptc_curves(all_results, output_dir):
    """
    绘制光子转移曲线 (Photon Transfer Curve, PTC)。

    每个增益档位一张子图，X 轴为信号均值 μ (DN)，Y 轴为方差 σ² (DN²)，
    散点为实测数据，直线为拟合结果。四个 Bayer 通道分别用不同颜色绘制。

    参数:
        all_results: dict, {gain: {channel: fit_result}}
        output_dir:  str, 图像保存目录
    """
    # 确定有数据的增益档位
    valid_gains = [g for g in GAIN_LIST if g in all_results]
    n_gains = len(valid_gains)
    if n_gains == 0:
        print("  [跳过] 无有效数据，不绘制 PTC 曲线")
        return

    fig, axes = plt.subplots(1, n_gains, figsize=(6 * n_gains, 5), squeeze=False)
    axes = axes[0]

    for i, gain in enumerate(valid_gains):
        ax = axes[i]
        for ch in CHANNEL_NAMES:
            res = all_results[gain].get(ch)
            if res is None:
                continue
            color = CHANNEL_COLORS[ch]
            mu = np.array(res['mu_list'])
            var = np.array(res['var_list'])

            # 散点: 实测数据
            ax.scatter(mu, var, c=color, s=15, alpha=0.6, label=f'{ch}')

            # 拟合直线
            x_fit = np.linspace(0, max(mu) * 1.05, 100)
            y_fit = res['alpha'] * x_fit + res['beta']
            ax.plot(x_fit, y_fit, c=color, linewidth=1.5, linestyle='--')

        ax.set_xlabel('Mean (DN)')
        ax.set_ylabel('Variance (DN²)')
        ax.set_title(f'Gain={gain}\n'
                     f'R²={all_results[gain].get("R", {}).get("r_squared", 0):.4f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Photon Transfer Curve (PTC) - Pocket4 Sensor', fontsize=14)
    plt.tight_layout()

    save_path = os.path.join(output_dir, 'ptc_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [保存] PTC 曲线 → {save_path}")


def plot_noise_model_summary(all_results, output_dir):
    """
    绘制噪声模型参数汇总图: α 和 β 随增益变化的趋势。

    左图: α vs Gain（预期线性增长）
    右图: β vs Gain（预期二次增长，模拟增益场景）

    参数:
        all_results: dict, {gain: {channel: fit_result}}
        output_dir:  str, 图像保存目录
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ch in CHANNEL_NAMES:
        gains = []
        alphas = []
        betas = []
        for gain in GAIN_LIST:
            res = all_results.get(gain, {}).get(ch)
            if res is None:
                continue
            gains.append(gain)
            alphas.append(res['alpha'])
            betas.append(res['beta'])

        color = CHANNEL_COLORS[ch]
        ax1.plot(gains, alphas, 'o-', color=color, label=ch, markersize=6)
        ax2.plot(gains, betas, 's-', color=color, label=ch, markersize=6)

    ax1.set_xlabel('Gain')
    ax1.set_ylabel('alpha (DN/e-)')
    ax1.set_title('alpha vs Gain')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')

    ax2.set_xlabel('Gain')
    ax2.set_ylabel('beta (DN²)')
    ax2.set_title('beta (Read Noise Var) vs Gain')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    plt.suptitle('Noise Model Parameters vs Gain - Pocket4', fontsize=13)
    plt.tight_layout()

    save_path = os.path.join(output_dir, 'noise_model_summary.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [保存] 参数趋势图 → {save_path}")


# ============================================================
# 第九部分：结果导出
# ============================================================

def export_results_json(all_results, blc_all, dark_var_all, output_dir,
                        bayer_pattern='RGGB'):
    """
    将标定结果导出为 JSON 文件。

    输出格式与 ISP 参数配置对齐，可直接用于 noiseMap 模块。

    参数:
        all_results:  dict, {gain: {channel: fit_result}}
        blc_all:      dict, {gain: {channel: float}}
        dark_var_all: dict, {gain: {channel: float}}
        output_dir:   str, 输出目录
        bayer_pattern: str

    返回:
        output: dict, 完整标定结果
    """
    output = OrderedDict()
    output['sensor_model'] = 'Pocket4'
    output['bit_depth'] = BIT_DEPTH
    output['max_dn'] = MAX_DN
    output['bayer_pattern'] = bayer_pattern
    output['noise_model'] = 'linear: var = alpha * mu + beta'
    output['calibration_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    output['unit'] = {'alpha': 'DN/e-', 'beta': 'DN^2', 'black_level': 'DN'}
    output['params'] = OrderedDict()

    for gain in GAIN_LIST:
        gain_key = f'gain_{gain}'
        output['params'][gain_key] = OrderedDict()
        for ch in CHANNEL_NAMES:
            res = all_results.get(gain, {}).get(ch)
            blc = blc_all.get(gain, {}).get(ch, 0)
            dark_v = dark_var_all.get(gain, {}).get(ch, 0)
            if res is not None:
                # 读出噪声 (e⁻) = sqrt(beta) / alpha
                # β < 0 说明拟合截距为负（数据点波动导致），取绝对值计算参考值
                beta_abs = max(res['beta'], 0)
                read_noise_e = (np.sqrt(beta_abs) / res['alpha']
                                if res['alpha'] > 0 else float('inf'))
                # 满阱容量 (e⁻) = (饱和DN - BLC) / alpha
                fwc = (MAX_DN - blc) / res['alpha'] if res['alpha'] > 0 else 0
                # 动态范围 (dB)
                dr = (20 * np.log10(fwc / read_noise_e)
                      if read_noise_e > 0 and fwc > 0 else 0)

                output['params'][gain_key][ch] = OrderedDict([
                    ('alpha', round(res['alpha'], 8)),
                    ('beta', round(res['beta'], 6)),
                    ('black_level', round(blc, 2)),
                    ('r_squared', round(res['r_squared'], 6)),
                    ('n_points', res['n_points']),
                    ('read_noise_e', round(read_noise_e, 4)),
                    ('full_well_capacity_e', round(fwc, 1)),
                    ('dynamic_range_dB', round(dr, 2)),
                    ('dark_frame_var', round(dark_v, 6)),
                ])
            else:
                output['params'][gain_key][ch] = {'error': '拟合失败'}

    # 保存 JSON
    save_path = os.path.join(output_dir, 'pocket4_noise_model.json')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  [保存] 标定结果 JSON → {save_path}")

    return output


def export_results_excel(all_results, blc_all, dark_var_all, output_dir):
    """
    将标定结果导出为 Excel 表格（需要 pandas + openpyxl）。

    Sheet 1: 参数总表（α, β, BLC, R², 读出噪声, 满阱, 动态范围）
    Sheet 2: 验证数据（β vs 暗帧方差对比）

    参数:
        all_results, blc_all, dark_var_all: 标定结果字典
        output_dir: 输出目录
    """
    if not HAS_PANDAS:
        print("  [跳过] 未安装 pandas/openpyxl，跳过 Excel 导出")
        return

    # ── Sheet 1: 参数总表 ──
    rows = []
    for gain in GAIN_LIST:
        for ch in CHANNEL_NAMES:
            res = all_results.get(gain, {}).get(ch)
            blc = blc_all.get(gain, {}).get(ch, 0)
            dark_v = dark_var_all.get(gain, {}).get(ch, 0)
            if res is not None:
                beta_abs = max(res['beta'], 0)
                read_noise_e = (np.sqrt(beta_abs) / res['alpha']
                                if res['alpha'] > 0 else float('inf'))
                fwc = (MAX_DN - blc) / res['alpha'] if res['alpha'] > 0 else 0
                dr = (20 * np.log10(fwc / read_noise_e)
                      if read_noise_e > 0 and fwc > 0 else 0)
                rows.append({
                    'Gain': gain,
                    'Channel': ch,
                    'alpha (DN/e-)': res['alpha'],
                    'beta (DN^2)': res['beta'],
                    'Black Level (DN)': blc,
                    'R²': res['r_squared'],
                    'N_points': res['n_points'],
                    'Read Noise (e-)': read_noise_e,
                    'FWC (e-)': fwc,
                    'DR (dB)': dr,
                    'Dark Var (DN^2)': dark_v,
                    'beta-DarkVar Diff%': (abs(res['beta'] - dark_v) / dark_v * 100
                                           if dark_v > 0 else float('nan')),
                })
            else:
                rows.append({'Gain': gain, 'Channel': ch, 'alpha (DN/e-)': 'FAIL'})

    df = pd.DataFrame(rows)
    save_path = os.path.join(output_dir, 'pocket4_noise_model.xlsx')
    try:
        with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='NoiseModel', index=False)
        print(f"  [保存] 标定结果 Excel → {save_path}")
    except ImportError:
        # openpyxl 未安装时，降级为 CSV 导出
        csv_path = os.path.join(output_dir, 'pocket4_noise_model.csv')
        df.to_csv(csv_path, index=False)
        print(f"  [保存] openpyxl 未安装，降级为 CSV → {csv_path}")


# ============================================================
# 第十部分：打印结果总表
# ============================================================

def print_summary_table(all_results, blc_all, dark_var_all):
    """
    在终端打印标定结果汇总表，一目了然。

    参数:
        all_results:  dict, {gain: {channel: fit_result}}
        blc_all:      dict, {gain: {channel: float}}
        dark_var_all: dict, {gain: {channel: float}}
    """
    print("\n" + "=" * 100)
    print("Pocket4 线性噪声模型标定结果")
    print("模型: σ²(y) = α · μ(y) + β")
    print("=" * 100)
    header = (f"{'Gain':>6} | {'Ch':>3} | {'α (DN/e⁻)':>12} | {'β (DN²)':>12} | "
              f"{'BLC (DN)':>10} | {'R²':>10} | {'ReadNoise(e⁻)':>14} | "
              f"{'FWC(e⁻)':>10} | {'DR(dB)':>8}")
    print(header)
    print("-" * 100)

    for gain in GAIN_LIST:
        for ch in CHANNEL_NAMES:
            res = all_results.get(gain, {}).get(ch)
            blc = blc_all.get(gain, {}).get(ch, 0)
            if res is not None:
                alpha = res['alpha']
                beta = res['beta']
                r2 = res['r_squared']
                beta_abs = max(beta, 0)
                rn_e = np.sqrt(beta_abs) / alpha if alpha > 0 else float('inf')
                fwc = (MAX_DN - blc) / alpha if alpha > 0 else 0
                dr = 20 * np.log10(fwc / rn_e) if rn_e > 0 and fwc > 0 else 0
                print(f"{gain:>6} | {ch:>3} | {alpha:>12.6f} | {beta:>12.4f} | "
                      f"{blc:>10.2f} | {r2:>10.6f} | {rn_e:>14.4f} | "
                      f"{fwc:>10.1f} | {dr:>8.2f}")
            else:
                print(f"{gain:>6} | {ch:>3} | {'FAIL':>12} | {'---':>12} | "
                      f"{blc:>10.2f} | {'---':>10} | {'---':>14} | "
                      f"{'---':>10} | {'---':>8}")
        print("-" * 100)


# ============================================================
# 第十一部分：模拟数据生成（用于无真实数据时验证流程）
# ============================================================

def generate_simulated_data(output_dir='/tmp/sim_pocket4', raw_shape=(3024, 4032)):
    """
    生成一套模拟 Raw 数据用于验证标定流程。

    模拟策略:
      - 对每个增益 g，噪声模型为: σ² = (α0·g)·μ + (β0·g²)
        其中 α0 是基础增益的 α，β0 是基础增益的 β
      - 色卡区域（中心 1/9）填充不同灰度值，模拟 24 个 patch
      - 其他区域填充中灰

    参数:
        output_dir: 模拟数据保存目录
        raw_shape:  (H, W) 模拟图像尺寸

    返回:
        data_root: str, 生成的数据根目录路径
        true_params: dict, 真实参数（用于验证标定准确性）
    """
    H, W = raw_shape
    np.random.seed(2024)

    # ── 真实噪声参数（ground truth）──
    alpha0 = 0.5    # Gain=1 时的 α (DN/e⁻)
    beta0 = 2.0     # Gain=1 时的 β (DN²)
    blc0 = 1024.0   # 基础黑电平 (14-bit Raw 中常见值)

    true_params = {}

    os.makedirs(output_dir, exist_ok=True)

    for gain in GAIN_LIST:
        # 该增益下的真实参数
        # 注意: Gain=1600 可能是混合增益，这里简化为线性关系
        alpha_g = alpha0 * gain       # α 与增益成正比
        beta_g = beta0 * (gain ** 2)  # β 与增益平方成正比（模拟增益）
        blc_g = blc0 + gain * 0.5     # 黑电平随增益略有增加

        true_params[gain] = {
            'alpha': alpha_g, 'beta': beta_g, 'black_level': blc_g
        }

        gain_dir = os.path.join(output_dir, f'gain_{gain}')

        # ── 生成黑帧 ──
        dark_dir = os.path.join(gain_dir, 'dark')
        os.makedirs(dark_dir, exist_ok=True)
        for frame_i in range(5):
            # 黑帧 = 黑电平 + 读出噪声
            dark = np.full((H, W), blc_g, dtype=np.float64)
            dark += np.random.randn(H, W) * np.sqrt(beta_g)
            dark = np.clip(dark, 0, MAX_DN).astype(np.uint16)
            dark.tofile(os.path.join(dark_dir, f'frame_{frame_i:03d}.raw'))

        # ── 生成各亮度级的带噪图像 ──
        for brightness in BRIGHTNESS_LIST:
            br_dir = os.path.join(gain_dir, f'brightness_{brightness}')
            os.makedirs(br_dir, exist_ok=True)

            # 色卡区域信号值（中心 1/9）
            chart_y1, chart_y2 = H // 3, 2 * H // 3
            chart_x1, chart_x2 = W // 3, 2 * W // 3

            for frame_i in range(10):
                # 初始化为中灰背景
                signal = np.full((H, W), blc_g + brightness / 100.0 * (MAX_DN - blc_g) * 0.5,
                                 dtype=np.float64)

                # 色卡区域: 24 个 patch 填充不同灰度
                patch_h = (chart_y2 - chart_y1) // CHART_ROWS
                patch_w = (chart_x2 - chart_x1) // CHART_COLS
                for pr in range(CHART_ROWS):
                    for pc in range(CHART_COLS):
                        # 每个 patch 的信号值随 patch 编号变化
                        patch_idx = pr * CHART_COLS + pc
                        # 信号值: 亮度百分比 × patch 灰度因子
                        patch_factor = (patch_idx + 1) / 24.0  # 0.04 ~ 1.0
                        patch_signal = blc_g + brightness / 100.0 * (MAX_DN - blc_g) * patch_factor
                        py1 = chart_y1 + pr * patch_h
                        py2 = chart_y1 + (pr + 1) * patch_h
                        px1 = chart_x1 + pc * patch_w
                        px2 = chart_x1 + (pc + 1) * patch_w
                        signal[py1:py2, px1:px2] = patch_signal

                # 添加噪声: σ² = α·(signal - BLC) + β
                net_signal = signal - blc_g
                net_signal = np.maximum(net_signal, 0)
                variance_map = alpha_g * net_signal + beta_g
                variance_map = np.maximum(variance_map, 1e-10)
                noise = np.random.randn(H, W) * np.sqrt(variance_map)

                noisy = signal + noise
                noisy = np.clip(noisy, 0, MAX_DN).astype(np.uint16)
                noisy.tofile(os.path.join(br_dir, f'frame_{frame_i:03d}.raw'))

    # 保存真实参数供验证
    true_path = os.path.join(output_dir, 'true_params.json')
    # 转换 key 为字符串以支持 JSON
    true_save = {str(k): v for k, v in true_params.items()}
    with open(true_path, 'w') as f:
        json.dump(true_save, f, indent=2)

    print(f"[模拟数据] 生成完成 → {output_dir}")
    print(f"  增益: {GAIN_LIST}")
    print(f"  亮度: {BRIGHTNESS_LIST}")
    print(f"  每增益每亮度: 10 帧, 黑帧: 5 帧")
    print(f"  图像尺寸: {H}×{W}, 14-bit")
    print(f"  真实参数: α0={alpha0}, β0={beta0}, BLC0={blc0}")

    return output_dir, true_params


# ============================================================
# 第十二部分：自动发现目录结构
# ============================================================

def discover_data_structure(data_root):
    """
    自动发现数据目录结构，返回组织好的路径字典。

    支持两种目录命名风格:
      风格 A (推荐):
        data_root/gain_1/dark/
        data_root/gain_1/brightness_12.5/
        ...
      风格 B:
        data_root/ISO100/dark/
        data_root/ISO100/12.5/
        ...

    参数:
        data_root: 数据根目录

    返回:
        structure: dict, {gain: {'dark': path, brightness: path, ...}}
    """
    structure = {}

    for gain in GAIN_LIST:
        # 尝试多种目录命名
        candidates = [
            f'gain_{gain}', f'Gain_{gain}', f'GAIN_{gain}',
            f'gain{gain}', f'ISO{gain}', f'iso{gain}',
            f'analog_gain_{gain}', f'AG_{gain}',
        ]
        gain_dir = None
        for c in candidates:
            path = os.path.join(data_root, c)
            if os.path.isdir(path):
                gain_dir = path
                break

        if gain_dir is None:
            print(f"  [警告] 未找到 Gain={gain} 的数据目录，跳过")
            continue

        structure[gain] = {}

        # 查找黑帧目录
        dark_candidates = ['dark', 'Dark', 'DARK', 'black', 'Black',
                           'blackframe', 'dark_frame', 'BLC']
        for dc in dark_candidates:
            dark_path = os.path.join(gain_dir, dc)
            if os.path.isdir(dark_path):
                structure[gain]['dark'] = dark_path
                break

        # 查找各亮度级目录
        for br in BRIGHTNESS_LIST:
            br_candidates = [
                f'brightness_{br}', f'Brightness_{br}',
                f'{br}', f'{br}%',
                f'lux_{br}', f'level_{br}',
            ]
            for bc in br_candidates:
                br_path = os.path.join(gain_dir, bc)
                if os.path.isdir(br_path):
                    structure[gain][br] = br_path
                    break

    # 打印发现结果
    print(f"\n[目录扫描] data_root = {data_root}")
    for gain in GAIN_LIST:
        if gain in structure:
            n_br = len([k for k in structure[gain] if k != 'dark'])
            has_dark = 'dark' in structure[gain]
            print(f"  Gain={gain:>5}: {'有黑帧' if has_dark else '无黑帧'}, "
                  f"{n_br}/{len(BRIGHTNESS_LIST)} 个亮度级")

    return structure


# ============================================================
# 主流程函数
# ============================================================

def run_calibration(data_root, raw_shape, bayer_pattern='RGGB',
                    output_dir='./calibration_results',
                    use_gray_only=False, max_frames=None):
    """
    执行完整的线性噪声模型标定流程。

    流程:
      Step 0: 数据目录扫描与加载
      Step 1: 黑帧 BLC 逐通道标定
      Step 2: ROI 选取（色卡 patch 定位）
      Step 3: 逐增益、逐亮度、逐通道计算均值-方差
      Step 4: 线性回归拟合 σ² = α·μ + β
      Step 5: 参数验证 & 可视化
      Step 6: 结果导出（JSON + Excel + PNG）

    参数:
        data_root:      数据根目录
        raw_shape:      (H, W) 原始 Raw 尺寸
        bayer_pattern:  Bayer 排列模式
        output_dir:     输出目录
        use_gray_only:  是否仅使用灰阶 patch 拟合
        max_frames:     每组最多加载帧数

    返回:
        all_results: dict, 完整标定结果
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("Pocket4 Sensor 线性噪声模型标定")
    print(f"  Raw 尺寸: {raw_shape[0]}×{raw_shape[1]}, {BIT_DEPTH}-bit")
    print(f"  Bayer: {bayer_pattern}")
    print(f"  输出目录: {output_dir}")
    print("=" * 70)

    # ────────────────────────────────────────────────────────────
    # Step 0: 发现数据目录结构
    # ────────────────────────────────────────────────────────────
    print("\n>>> Step 0: 扫描数据目录结构")
    data_struct = discover_data_structure(data_root)

    if len(data_struct) == 0:
        raise FileNotFoundError(f"在 {data_root} 下未找到任何增益档位数据")

    # ────────────────────────────────────────────────────────────
    # Step 1: 黑帧 BLC 逐通道标定
    # ────────────────────────────────────────────────────────────
    print("\n>>> Step 1: 黑电平 (BLC) 逐通道标定")
    blc_all = {}       # {gain: {channel: blc_value}}
    dark_var_all = {}   # {gain: {channel: dark_var}}

    for gain in GAIN_LIST:
        if gain not in data_struct:
            continue
        if 'dark' not in data_struct[gain]:
            print(f"  [警告] Gain={gain} 无黑帧数据，使用默认 BLC=0")
            blc_all[gain] = {ch: 0.0 for ch in CHANNEL_NAMES}
            dark_var_all[gain] = {ch: 0.0 for ch in CHANNEL_NAMES}
            continue

        print(f"\n  Gain={gain}:")
        dark_frames = load_multi_frames(
            data_struct[gain]['dark'], raw_shape, BIT_DEPTH, max_frames
        )
        blc_dict, dv_dict = calibrate_blc_per_channel(dark_frames, bayer_pattern)
        blc_all[gain] = blc_dict
        dark_var_all[gain] = dv_dict

    # ────────────────────────────────────────────────────────────
    # Step 2: ROI 选取
    # ────────────────────────────────────────────────────────────
    print("\n>>> Step 2: 色卡 ROI 选取")
    rois = compute_patch_rois(raw_shape[0], raw_shape[1], bayer_pattern)

    # ────────────────────────────────────────────────────────────
    # Step 3 & 4: 逐增益、逐亮度计算均值-方差，然后拟合
    # ────────────────────────────────────────────────────────────
    print("\n>>> Step 3-4: 均值-方差计算 & 线性拟合")
    all_results = {}  # {gain: {channel: fit_result}}

    for gain in GAIN_LIST:
        if gain not in data_struct:
            continue
        print(f"\n  ──── Gain={gain} ────")

        # 收集该增益下所有亮度级的数据点
        # 结构: {channel: [data_point, data_point, ...]}
        channel_data_points = {ch: [] for ch in CHANNEL_NAMES}

        for brightness in BRIGHTNESS_LIST:
            if brightness not in data_struct[gain]:
                print(f"    亮度 {brightness}%: 数据缺失，跳过")
                continue

            print(f"    亮度 {brightness}%:")

            # 加载该条件下的多帧 Raw
            frames = load_multi_frames(
                data_struct[gain][brightness], raw_shape, BIT_DEPTH, max_frames
            )

            # 按 Bayer 通道拆分
            channels = split_bayer_multi_frames(frames, bayer_pattern)

            # 逐通道计算均值-方差
            for ch_name in CHANNEL_NAMES:
                ch_frames = channels[ch_name]  # (N, H/2, W/2)
                blc_val = blc_all.get(gain, {}).get(ch_name, 0)

                # 选择统计方法: ≥ 3 帧用时域统计，2 帧用帧差法
                n_frames = ch_frames.shape[0]
                if n_frames >= 3:
                    mv_results = compute_mean_variance_temporal(
                        ch_frames, blc_val, rois
                    )
                elif n_frames == 2:
                    mv_results = compute_mean_variance_frame_diff(
                        ch_frames, blc_val, rois
                    )
                else:
                    print(f"      {ch_name}: 只有 1 帧，跳过")
                    continue

                channel_data_points[ch_name].extend(mv_results)

        # 对每个通道做线性拟合
        all_results[gain] = {}
        for ch_name in CHANNEL_NAMES:
            dps = channel_data_points[ch_name]
            if len(dps) == 0:
                print(f"    {ch_name}: 无有效数据")
                continue

            fit_res = fit_linear_noise_model(dps, use_gray_only=use_gray_only)
            if fit_res is not None:
                all_results[gain][ch_name] = fit_res
                print(f"    {ch_name}: α={fit_res['alpha']:.6f}, "
                      f"β={fit_res['beta']:.4f}, R²={fit_res['r_squared']:.6f}, "
                      f"N={fit_res['n_points']}, 离群={fit_res['outlier_count']}")
            else:
                print(f"    {ch_name}: 拟合失败")

    # ────────────────────────────────────────────────────────────
    # Step 5: 参数验证
    # ────────────────────────────────────────────────────────────
    print("\n>>> Step 5: 参数验证")
    report = validate_calibration(all_results, dark_var_all)
    for line in report:
        print(line)

    # 保存验证报告
    report_path = os.path.join(output_dir, 'validation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    print(f"  [保存] 验证报告 → {report_path}")

    # ────────────────────────────────────────────────────────────
    # Step 6: 可视化 & 导出
    # ────────────────────────────────────────────────────────────
    print("\n>>> Step 6: 可视化 & 结果导出")

    # 打印终端总表
    print_summary_table(all_results, blc_all, dark_var_all)

    # PTC 曲线
    plot_ptc_curves(all_results, output_dir)

    # 参数趋势图
    plot_noise_model_summary(all_results, output_dir)

    # JSON 导出
    export_results_json(all_results, blc_all, dark_var_all, output_dir, bayer_pattern)

    # Excel 导出
    export_results_excel(all_results, blc_all, dark_var_all, output_dir)

    print("\n" + "=" * 70)
    print("标定完成!")
    print("=" * 70)

    return all_results


# ============================================================
# 模拟数据验证函数
# ============================================================

def run_simulation_validation():
    """
    使用模拟数据运行标定流程并验证准确性。

    流程:
      1. 生成已知参数的模拟数据
      2. 运行标定流程
      3. 对比标定结果与真实参数

    无需真实 Raw 数据即可验证整个标定管线是否正确。
    """
    print("\n" + "#" * 70)
    print("# 模拟模式: 生成模拟数据并验证标定准确性")
    print("#" * 70)

    # 使用较小尺寸加速模拟
    sim_shape = (1200, 1600)
    sim_dir = '/tmp/sim_pocket4'

    # 生成模拟数据
    data_root, true_params = generate_simulated_data(sim_dir, sim_shape)

    # 运行标定
    output_dir = os.path.join(sim_dir, 'calibration_output')
    all_results = run_calibration(
        data_root, sim_shape,
        bayer_pattern='RGGB',
        output_dir=output_dir,
        use_gray_only=False,
        max_frames=10,
    )

    # ── 对比标定结果与真实参数 ──
    print("\n" + "=" * 70)
    print("标定精度验证: 标定值 vs 真实值")
    print("=" * 70)
    print(f"{'Gain':>6} | {'Ch':>3} | {'α_true':>10} | {'α_cal':>10} | {'α_err%':>8} | "
          f"{'β_true':>10} | {'β_cal':>10} | {'β_err%':>8}")
    print("-" * 80)

    for gain in GAIN_LIST:
        if gain not in all_results:
            continue
        tp = true_params[gain]
        for ch in CHANNEL_NAMES:
            res = all_results[gain].get(ch)
            if res is None:
                continue
            a_err = abs(res['alpha'] - tp['alpha']) / tp['alpha'] * 100
            b_err = abs(res['beta'] - tp['beta']) / tp['beta'] * 100
            print(f"{gain:>6} | {ch:>3} | {tp['alpha']:>10.4f} | {res['alpha']:>10.4f} | "
                  f"{a_err:>7.2f}% | {tp['beta']:>10.4f} | {res['beta']:>10.4f} | "
                  f"{b_err:>7.2f}%")
        print("-" * 80)


# ============================================================
# CLI 入口
# ============================================================

def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description='Pocket4 Sensor 线性噪声模型标定工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 模拟数据测试
  python pocket4_noise_calibration.py --simulate

  # 真实数据标定
  python pocket4_noise_calibration.py --data_root /path/to/raw_data --raw_shape 3024 4032

  # 指定 Bayer 和输出目录
  python pocket4_noise_calibration.py --data_root /data --raw_shape 3024 4032 \\
      --bayer GRBG --output_dir ./results --gray_only
        """
    )

    parser.add_argument('--data_root', type=str, default=None,
                        help='Raw 数据根目录')
    parser.add_argument('--raw_shape', type=int, nargs=2, default=None,
                        metavar=('H', 'W'),
                        help='Raw 图像尺寸, 如: 3024 4032')
    parser.add_argument('--bayer', type=str, default='RGGB',
                        choices=['RGGB', 'BGGR', 'GRBG', 'GBRG'],
                        help='Bayer pattern (默认 RGGB)')
    parser.add_argument('--output_dir', type=str, default='./calibration_results',
                        help='标定结果输出目录 (默认 ./calibration_results)')
    parser.add_argument('--simulate', action='store_true',
                        help='使用模拟数据运行标定流程（验证管线正确性）')
    parser.add_argument('--gray_only', action='store_true',
                        help='仅使用灰阶 patch 拟合（更准确但数据点更少）')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='每组最多加载帧数，None 表示全部')
    parser.add_argument('--bit_depth', type=int, default=14,
                        help='Raw 数据位深度 (默认 14)')

    return parser.parse_args()


def main():
    """程序入口: 根据命令行参数选择模拟模式或真实标定模式。"""
    args = parse_args()

    # 更新全局位深度
    global BIT_DEPTH, MAX_DN
    BIT_DEPTH = args.bit_depth
    MAX_DN = (2 ** BIT_DEPTH) - 1

    if args.simulate:
        # ── 模拟模式 ──
        run_simulation_validation()
    else:
        # ── 真实数据标定模式 ──
        if args.data_root is None:
            print("[错误] 请指定 --data_root 或使用 --simulate 模式")
            return
        if args.raw_shape is None:
            print("[错误] 请指定 --raw_shape，例如: --raw_shape 3024 4032")
            return

        run_calibration(
            data_root=args.data_root,
            raw_shape=tuple(args.raw_shape),
            bayer_pattern=args.bayer,
            output_dir=args.output_dir,
            use_gray_only=args.gray_only,
            max_frames=args.max_frames,
        )


if __name__ == '__main__':
    main()
