"""
KL散度计算脚本：评估线性噪声模型拟合质量
===========================================
流程：
1. 真实数据raw采集（支持多ISO，各ISO有对应黑帧BLC数据）
2. 同ISO多帧stack堆叠计算平均，得到干净信号GT
3. real noise = raw - GT，得到多帧噪声样本集合
4. 线性噪声模型 σ²=k·I+b 构建fake noise
5. 基于real noise和fake noise构建概率分布（直方图拟合）
6. 计算KL散度
7. 对每个亮度档位循环，得到多组KL散度结果并统计分析
8. 输出结果图像和Excel表格

使用方法：
    # 使用真实raw数据，指定BLC黑帧目录
    python kl_count_function.py --data_root /path/to/raw_data --blc_root /path/to/blc_data --iso 200

    # 使用模拟数据测试流程
    python kl_count_function.py --simulate --iso 200
    
    # 指定输出Excel文件名
    python kl_count_function.py --data_root /path/to/raw_data --blc_root /path/to/blc --output_excel kl_results.xlsx
"""

import numpy as np
import os
import argparse
from scipy.stats import entropy
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，服务器环境可用
import matplotlib.pyplot as plt

# 尝试导入pandas和openpyxl用于Excel输出
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("[警告] 未安装pandas，Excel输出功能不可用。请运行: pip install pandas openpyxl")


# ============================================================
# 2套线性噪声模型参数 (k, b)
# 线性噪声模型: variance = k * I + b
# 其中 I 是归一化信号强度 [0, 1]
# ============================================================
# 参数集1：例如来自标定方法A或论文参数
NOISE_PARAMS_1 = {
    'k': 0.0020,   # shot noise系数（信号相关）
    'b': 0.00001,  # read noise方差（信号无关）
    'label': '模型1 (k1=0.0020, b1=1e-5)'
}

# 参数集2：例如来自标定方法B或不同拟合
NOISE_PARAMS_2 = {
    'k': 0.0025,   # shot noise系数
    'b': 0.000005, # read noise方差
    'label': '模型2 (k2=0.0025, b2=5e-6)'
}


def load_blc_data(blc_root, iso, bit_depth=14):
    """
    加载指定ISO的黑帧BLC（Black Level Correction）数据
    
    支持的BLC数据格式：
    1. 单个npy文件：blc_root/ISO{iso}/blc.npy 或 blc_root/iso{iso}_blc.npy
    2. 多帧黑帧取平均：blc_root/ISO{iso}/ 目录下的多个raw/npy文件
    3. 直接提供黑电平值的txt/json文件
    
    参数:
        blc_root: BLC数据根目录
        iso: ISO值
        bit_depth: raw数据的位深度（默认14-bit）
    
    返回:
        blc_value: float 或 ndarray, 黑电平值（归一化前的原始值）
        如果是单一值，返回float；如果是空间变化的，返回2D数组
    """
    if blc_root is None:
        return None
    
    max_value = (2 ** bit_depth) - 1
    
    # 尝试多种命名方式查找BLC数据
    possible_paths = [
        os.path.join(blc_root, f"ISO{iso}", "blc.npy"),
        os.path.join(blc_root, f"ISO{iso}", "black_level.npy"),
        os.path.join(blc_root, f"iso{iso}_blc.npy"),
        os.path.join(blc_root, f"ISO{iso}_blc.npy"),
        os.path.join(blc_root, f"blc_iso{iso}.npy"),
        os.path.join(blc_root, f"ISO{iso}", "blc.txt"),
        os.path.join(blc_root, f"iso{iso}_blc.txt"),
    ]
    
    # 方式1：查找预处理的BLC npy文件
    for path in possible_paths:
        if path.endswith('.npy') and os.path.exists(path):
            blc_data = np.load(path)
            print(f"  加载BLC数据: {path}")
            if blc_data.ndim == 0:  # 标量
                return float(blc_data)
            elif blc_data.ndim == 1 and len(blc_data) <= 4:  # 每通道黑电平
                return np.mean(blc_data)  # 取平均
            else:  # 2D空间变化的黑电平
                return blc_data
        elif path.endswith('.txt') and os.path.exists(path):
            with open(path, 'r') as f:
                content = f.read().strip()
                # 尝试解析为数值
                try:
                    values = [float(v) for v in content.split()]
                    if len(values) == 1:
                        print(f"  加载BLC数据: {path}, 值={values[0]}")
                        return values[0]
                    else:
                        print(f"  加载BLC数据: {path}, 值={values}")
                        return np.mean(values)
                except ValueError:
                    continue
    
    # 方式2：查找黑帧目录，多帧堆叠计算平均黑电平
    blc_dirs = [
        os.path.join(blc_root, f"ISO{iso}"),
        os.path.join(blc_root, f"iso{iso}"),
        os.path.join(blc_root, f"ISO{iso}_dark"),
        os.path.join(blc_root, f"dark_ISO{iso}"),
    ]
    
    for blc_dir in blc_dirs:
        if os.path.isdir(blc_dir):
            # 检查是否有raw文件
            raw_files = [f for f in os.listdir(blc_dir) 
                        if f.endswith(('.raw', '.RAW', '.npy', '.dng', '.DNG', '.ARW'))]
            if len(raw_files) > 0:
                print(f"  从黑帧目录计算BLC: {blc_dir} ({len(raw_files)} 帧)")
                frames = load_raw_frames_with_blc(blc_dir, blc_value=0, 
                                                  bit_depth=bit_depth, 
                                                  normalize=False)
                blc_value = np.mean(frames)
                print(f"  计算得到黑电平值: {blc_value:.2f}")
                return blc_value
    
    print(f"  [警告] 未找到ISO{iso}的BLC数据，将使用默认值0")
    return None


def load_raw_frames_with_blc(folder_path, blc_value=None, bit_depth=14, 
                              max_frames=None, normalize=True, raw_shape=None):
    """
    加载一个亮度档位文件夹下的所有raw帧，并应用BLC校正

    支持格式: .raw (需指定shape), .dng (需要rawpy), .npy (numpy数组)

    参数:
        folder_path: 存放同一亮度档位多帧raw的文件夹路径
        blc_value: 黑电平值（可以是标量或与raw同shape的数组）
        bit_depth: raw数据位深度，用于归一化
        max_frames: 最大加载帧数，None表示全部加载
        normalize: 是否归一化到[0,1]
        raw_shape: raw文件的shape，格式为(H, W)，仅对.raw文件需要

    返回:
        frames: numpy数组, shape=(N, H, W), dtype=float64
    """
    frames = []
    supported_ext = ('.dng', '.DNG', '.npy', '.raw', '.RAW', '.ARW', '.CR2', '.nef', '.NEF')
    max_value = (2 ** bit_depth) - 1

    # 获取文件列表并排序
    files = sorted([f for f in os.listdir(folder_path)
                    if f.endswith(supported_ext)])

    if max_frames is not None:
        files = files[:max_frames]

    if len(files) == 0:
        raise FileNotFoundError(f"在 {folder_path} 中未找到支持的raw文件")

    for fname in files:
        fpath = os.path.join(folder_path, fname)

        if fname.endswith('.npy'):
            # 直接加载numpy数组
            frame = np.load(fpath).astype(np.float64)
            
        elif fname.lower().endswith('.raw'):
            # 加载纯raw二进制文件
            if raw_shape is None:
                # 尝试从文件名解析shape，如 "frame_001_4032x3024.raw"
                import re
                match = re.search(r'(\d+)x(\d+)', fname)
                if match:
                    width, height = int(match.group(1)), int(match.group(2))
                    raw_shape = (height, width)
                else:
                    raise ValueError(f"无法确定raw文件shape，请通过--raw_shape参数指定，或在文件名中包含如'4032x3024'格式")
            
            # 根据位深度选择dtype
            if bit_depth <= 8:
                dtype = np.uint8
            elif bit_depth <= 16:
                dtype = np.uint16
            else:
                dtype = np.uint32
            
            frame = np.fromfile(fpath, dtype=dtype).astype(np.float64)
            frame = frame.reshape(raw_shape)
            
        elif fname.endswith(('.dng', '.DNG', '.ARW', '.CR2', '.nef', '.NEF')):
            # 使用rawpy加载DNG/RAW文件
            try:
                import rawpy
            except ImportError:
                raise ImportError("请安装rawpy: pip install rawpy")
            with rawpy.imread(fpath) as raw:
                # 获取原始Bayer数据，不做任何处理
                frame = raw.raw_image_visible.astype(np.float64)
                # 如果没有提供blc_value，使用rawpy读取的黑电平
                if blc_value is None:
                    blc_value = np.mean(raw.black_level_per_channel)
                    max_value = raw.white_level
        else:
            continue

        # 应用BLC校正
        if blc_value is not None:
            frame = frame - blc_value
            frame = np.maximum(frame, 0)  # 防止负值

        # 归一化到[0, 1]
        if normalize:
            frame = frame / max_value
            frame = np.clip(frame, 0, 1)

        frames.append(frame)

    frames = np.array(frames)
    blc_info = f"BLC={blc_value:.2f}" if blc_value is not None else "BLC=auto"
    print(f"  加载了 {len(frames)} 帧, shape={frames[0].shape}, "
          f"范围=[{frames.min():.4f}, {frames.max():.4f}], {blc_info}")
    return frames


def load_raw_frames(folder_path, max_frames=None):
    """
    加载一个亮度档位文件夹下的所有raw帧（兼容旧接口）

    支持格式: .dng (需要rawpy), .npy (numpy数组)

    参数:
        folder_path: 存放同一亮度档位多帧raw的文件夹路径
        max_frames: 最大加载帧数，None表示全部加载

    返回:
        frames: numpy数组, shape=(N, H, W), dtype=float64, 归一化到[0,1]
    """
    return load_raw_frames_with_blc(folder_path, blc_value=None, max_frames=max_frames)


def simulate_raw_frames(brightness_level, num_frames=50, height=256, width=256,
                        k_true=0.0022, b_true=0.000008):
    """
    模拟不同亮度档位的raw帧数据（用于无真实数据时测试流程）

    模拟策略：
    - 干净信号 = brightness_level（归一化值）
    - 噪声服从 N(0, k*I + b)，即高斯噪声，方差随信号线性变化

    参数:
        brightness_level: 归一化亮度值 [0, 1]
        num_frames: 模拟帧数
        height, width: 帧尺寸
        k_true, b_true: 真实噪声模型参数（模拟数据的ground truth参数）

    返回:
        frames: shape=(num_frames, H, W)
    """
    # 构造干净信号（添加微小空间变化使其更真实）
    np.random.seed(42)
    # 生成一个带有缓慢空间变化的base signal
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xx, yy = np.meshgrid(x, y)
    # 信号在brightness_level附近有±5%的空间变化
    clean_signal = brightness_level * (1.0 + 0.05 * np.sin(2 * np.pi * xx)
                                       * np.cos(2 * np.pi * yy))
    clean_signal = np.clip(clean_signal, 0, 1)

    # 计算每个像素位置的噪声标准差: σ = sqrt(k*I + b)
    variance_map = k_true * clean_signal + b_true
    variance_map = np.maximum(variance_map, 1e-10)  # 防止负方差
    std_map = np.sqrt(variance_map)

    # 生成多帧含噪数据
    frames = np.zeros((num_frames, height, width))
    for i in range(num_frames):
        noise = np.random.randn(height, width) * std_map
        frames[i] = np.clip(clean_signal + noise, 0, 1)

    print(f"  模拟 {num_frames} 帧, 亮度级别={brightness_level:.3f}, "
          f"shape=({height},{width})")
    return frames


def compute_gt_by_stacking(frames):
    """
    多帧堆叠取平均，得到干净信号GT

    原理：噪声是零均值的随机变量，多帧取均值后噪声趋近于0
    N帧平均后噪声标准差降低为原来的 1/sqrt(N)

    参数:
        frames: shape=(N, H, W) 多帧raw数据

    返回:
        gt: shape=(H, W) 干净信号估计
    """
    gt = np.mean(frames, axis=0)
    print(f"  GT计算完成: {frames.shape[0]}帧堆叠平均, "
          f"残余噪声std≈原始的1/{np.sqrt(frames.shape[0]):.1f}")
    return gt


def extract_real_noise(frames, gt):
    """
    提取真实噪声样本: real_noise = raw - GT

    参数:
        frames: shape=(N, H, W) 原始帧
        gt: shape=(H, W) 干净信号GT

    返回:
        noise_samples: shape=(N, H, W) 每帧的噪声
    """
    noise_samples = frames - gt[np.newaxis, :, :]
    return noise_samples


def generate_fake_noise(gt, noise_params, num_samples):
    """
    基于线性噪声模型生成模拟噪声

    线性噪声模型: σ²(I) = k * I + b
    对每个像素位置，根据其信号强度I计算方差，生成高斯噪声

    参数:
        gt: shape=(H, W) 干净信号（作为I）
        noise_params: dict, 包含 'k' 和 'b'
        num_samples: 生成的噪声帧数

    返回:
        fake_noise: shape=(num_samples, H, W)
    """
    k = noise_params['k']
    b = noise_params['b']

    # 计算每个像素位置的噪声方差: σ² = k*I + b
    variance_map = k * gt + b
    # 确保方差非负
    variance_map = np.maximum(variance_map, 1e-10)
    std_map = np.sqrt(variance_map)

    # 生成与真实噪声相同数量的fake noise帧
    H, W = gt.shape
    fake_noise = np.zeros((num_samples, H, W))
    for i in range(num_samples):
        fake_noise[i] = np.random.randn(H, W) * std_map

    return fake_noise


def build_noise_histogram(noise_samples, num_bins=200, range_sigma=5):
    """
    将噪声样本构建为概率分布（归一化直方图）

    参数:
        noise_samples: 噪声样本数组（可以是任意shape，会被展平）
        num_bins: 直方图bin数量
        range_sigma: 直方图范围，以标准差的倍数表示

    返回:
        prob: 归一化概率分布, shape=(num_bins,)
        bin_centers: bin中心值, shape=(num_bins,)
        bin_edges: bin边界, shape=(num_bins+1,)
    """
    flat_samples = noise_samples.flatten()

    # 根据样本统计量确定直方图范围
    mean_val = np.mean(flat_samples)
    std_val = np.std(flat_samples)

    # 直方图范围: [mean - range_sigma*std, mean + range_sigma*std]
    hist_min = mean_val - range_sigma * std_val
    hist_max = mean_val + range_sigma * std_val

    # 计算直方图
    counts, bin_edges = np.histogram(flat_samples, bins=num_bins,
                                     range=(hist_min, hist_max))

    # 归一化为概率分布（加上极小值防止log(0)）
    prob = counts.astype(np.float64) + 1e-10
    prob = prob / prob.sum()

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    return prob, bin_centers, bin_edges


def compute_kl_divergence(p, q):
    """
    计算KL散度: KL(P || Q)

    衡量用分布Q近似分布P时的信息损失
    KL(P||Q) = Σ P(x) * log(P(x) / Q(x))

    参数:
        p: 真实分布（real noise的概率分布）
        q: 近似分布（fake noise的概率分布）

    返回:
        kl_value: KL散度值（非负数，越小表示拟合越好）
    """
    # scipy.stats.entropy(p, q) 计算的就是 KL(P || Q)
    kl_value = entropy(p, q)
    return kl_value


def compute_symmetric_kl(p, q):
    """
    计算对称KL散度 (Jensen-Shannon散度的简化版)

    SKL(P, Q) = 0.5 * KL(P||Q) + 0.5 * KL(Q||P)

    对称形式更稳健，不受P、Q顺序影响

    参数:
        p, q: 两个概率分布

    返回:
        skl_value: 对称KL散度值
    """
    kl_pq = entropy(p, q)
    kl_qp = entropy(q, p)
    return 0.5 * kl_pq + 0.5 * kl_qp


def analyze_single_brightness(frames, noise_params_1, noise_params_2,
                              brightness_label, num_bins=200):
    """
    对单个亮度档位执行完整的分析流程

    流程:
    1. 多帧堆叠 → GT
    2. real noise = raw - GT
    3. 2套参数分别生成fake noise
    4. 构建概率分布
    5. 计算KL散度

    参数:
        frames: shape=(N, H, W) 该亮度档位的多帧数据
        noise_params_1: 第1套噪声模型参数
        noise_params_2: 第2套噪声模型参数
        brightness_label: 亮度档位标签（用于打印）
        num_bins: 直方图bin数

    返回:
        result: dict, 包含KL散度等统计结果
    """
    print(f"\n{'='*60}")
    print(f"  亮度档位: {brightness_label}")
    print(f"{'='*60}")

    # ---- Step 1: 多帧堆叠计算GT ----
    gt = compute_gt_by_stacking(frames)
    mean_intensity = np.mean(gt)
    print(f"  GT平均信号强度: {mean_intensity:.4f}")

    # ---- Step 2: 提取真实噪声 ----
    real_noise = extract_real_noise(frames, gt)
    real_noise_std = np.std(real_noise)
    print(f"  真实噪声std: {real_noise_std:.6f}")

    # ---- Step 3: 生成模拟噪声（2套参数） ----
    num_frames = frames.shape[0]
    fake_noise_1 = generate_fake_noise(gt, noise_params_1, num_frames)
    fake_noise_2 = generate_fake_noise(gt, noise_params_2, num_frames)
    print(f"  模型1 fake噪声std: {np.std(fake_noise_1):.6f}")
    print(f"  模型2 fake噪声std: {np.std(fake_noise_2):.6f}")

    # ---- Step 4: 构建概率分布 ----
    # 使用统一的直方图范围（基于real noise的范围）
    all_noise = np.concatenate([real_noise.flatten(),
                                fake_noise_1.flatten(),
                                fake_noise_2.flatten()])
    global_std = np.std(all_noise)
    global_mean = np.mean(all_noise)
    hist_range = (global_mean - 5 * global_std, global_mean + 5 * global_std)

    # 统一bin边界，确保分布可比
    bin_edges = np.linspace(hist_range[0], hist_range[1], num_bins + 1)

    # 构建三个分布的直方图
    hist_real, _ = np.histogram(real_noise.flatten(), bins=bin_edges)
    hist_fake1, _ = np.histogram(fake_noise_1.flatten(), bins=bin_edges)
    hist_fake2, _ = np.histogram(fake_noise_2.flatten(), bins=bin_edges)

    # 归一化为概率分布（加平滑项防止除零）
    eps = 1e-10
    prob_real = (hist_real + eps) / (hist_real + eps).sum()
    prob_fake1 = (hist_fake1 + eps) / (hist_fake1 + eps).sum()
    prob_fake2 = (hist_fake2 + eps) / (hist_fake2 + eps).sum()

    # ---- Step 5: 计算KL散度 ----
    # KL(real || fake): 用fake近似real时的信息损失
    kl_1 = compute_kl_divergence(prob_real, prob_fake1)
    kl_2 = compute_kl_divergence(prob_real, prob_fake2)

    # 对称KL散度（更稳健的度量）
    skl_1 = compute_symmetric_kl(prob_real, prob_fake1)
    skl_2 = compute_symmetric_kl(prob_real, prob_fake2)

    print(f"\n  --- KL散度结果 ---")
    print(f"  KL(real || 模型1): {kl_1:.6f}")
    print(f"  KL(real || 模型2): {kl_2:.6f}")
    print(f"  对称KL(real, 模型1): {skl_1:.6f}")
    print(f"  对称KL(real, 模型2): {skl_2:.6f}")

    # 判断哪个模型更优
    better_model = "模型1" if kl_1 < kl_2 else "模型2"
    print(f"  → 更优模型: {better_model} (KL更小=拟合更好)")

    result = {
        'brightness_label': brightness_label,
        'mean_intensity': mean_intensity,
        'real_noise_std': real_noise_std,
        'fake1_noise_std': np.std(fake_noise_1),
        'fake2_noise_std': np.std(fake_noise_2),
        'kl_model1': kl_1,
        'kl_model2': kl_2,
        'skl_model1': skl_1,
        'skl_model2': skl_2,
        'prob_real': prob_real,
        'prob_fake1': prob_fake1,
        'prob_fake2': prob_fake2,
        'bin_edges': bin_edges,
    }

    return result


def plot_results(results, output_dir):
    """
    绘制KL散度结果的可视化图

    包含：
    1. 各亮度档位的KL散度对比柱状图
    2. 各亮度档位的噪声分布对比图
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---- 图1: KL散度随亮度变化的趋势图 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    brightness_levels = [r['mean_intensity'] for r in results]
    kl_1_values = [r['kl_model1'] for r in results]
    kl_2_values = [r['kl_model2'] for r in results]
    skl_1_values = [r['skl_model1'] for r in results]
    skl_2_values = [r['skl_model2'] for r in results]

    # 子图1: KL(P||Q)
    ax = axes[0]
    x = np.arange(len(results))
    width = 0.35
    ax.bar(x - width/2, kl_1_values, width, label='Model 1', color='steelblue')
    ax.bar(x + width/2, kl_2_values, width, label='Model 2', color='coral')
    ax.set_xlabel('Brightness Level (mean intensity)')
    ax.set_ylabel('KL Divergence')
    ax.set_title('KL(Real || Fake) at Each Brightness Level')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{bl:.3f}" for bl in brightness_levels], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 子图2: 对称KL
    ax = axes[1]
    ax.bar(x - width/2, skl_1_values, width, label='Model 1', color='steelblue')
    ax.bar(x + width/2, skl_2_values, width, label='Model 2', color='coral')
    ax.set_xlabel('Brightness Level (mean intensity)')
    ax.set_ylabel('Symmetric KL Divergence')
    ax.set_title('Symmetric KL at Each Brightness Level')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{bl:.3f}" for bl in brightness_levels], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'kl_divergence_comparison.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  KL散度对比图已保存: {fig_path}")

    # ---- 图2: 每个亮度档位的噪声分布对比 ----
    n_levels = len(results)
    fig, axes = plt.subplots(1, n_levels, figsize=(5 * n_levels, 4))
    if n_levels == 1:
        axes = [axes]

    for i, r in enumerate(results):
        ax = axes[i]
        bin_centers = (r['bin_edges'][:-1] + r['bin_edges'][1:]) / 2.0
        ax.plot(bin_centers, r['prob_real'], 'k-', linewidth=1.5,
                label='Real Noise', alpha=0.8)
        ax.plot(bin_centers, r['prob_fake1'], 'b--', linewidth=1.2,
                label='Model 1', alpha=0.7)
        ax.plot(bin_centers, r['prob_fake2'], 'r--', linewidth=1.2,
                label='Model 2', alpha=0.7)
        ax.set_title(f"Intensity={r['mean_intensity']:.3f}\n"
                     f"KL1={r['kl_model1']:.4f}, KL2={r['kl_model2']:.4f}",
                     fontsize=9)
        ax.set_xlabel('Noise Value')
        ax.set_ylabel('Probability')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'noise_distribution_comparison.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  噪声分布对比图已保存: {fig_path}")


def export_results_to_excel(results, output_path, iso, noise_params_1, noise_params_2):
    """
    将KL散度结果导出到Excel表格
    
    参数:
        results: 分析结果列表
        output_path: Excel文件输出路径
        iso: ISO值
        noise_params_1, noise_params_2: 噪声模型参数
    """
    if not HAS_PANDAS:
        print("[警告] pandas未安装，无法导出Excel。请运行: pip install pandas openpyxl")
        return None
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # ============================================================
    # Sheet 1: KL散度详细结果表
    # ============================================================
    detail_data = []
    for r in results:
        better = "模型1" if r['kl_model1'] < r['kl_model2'] else "模型2"
        detail_data.append({
            '亮度档位': r['brightness_label'],
            '平均信号强度': r['mean_intensity'],
            '真实噪声std': r['real_noise_std'],
            '模型1噪声std': r['fake1_noise_std'],
            '模型2噪声std': r['fake2_noise_std'],
            'KL(real||模型1)': r['kl_model1'],
            'KL(real||模型2)': r['kl_model2'],
            '对称KL(模型1)': r['skl_model1'],
            '对称KL(模型2)': r['skl_model2'],
            '更优模型': better,
        })
    
    df_detail = pd.DataFrame(detail_data)
    
    # ============================================================
    # Sheet 2: KL矩阵数据 (亮度 x 模型)
    # ============================================================
    brightness_labels = [r['brightness_label'] for r in results]
    kl_matrix_data = {
        '亮度档位': brightness_labels,
        '模型1 KL': [r['kl_model1'] for r in results],
        '模型2 KL': [r['kl_model2'] for r in results],
        '模型1 对称KL': [r['skl_model1'] for r in results],
        '模型2 对称KL': [r['skl_model2'] for r in results],
    }
    df_kl_matrix = pd.DataFrame(kl_matrix_data)
    
    # ============================================================
    # Sheet 3: 统计汇总
    # ============================================================
    kl_1 = np.array([r['kl_model1'] for r in results])
    kl_2 = np.array([r['kl_model2'] for r in results])
    skl_1 = np.array([r['skl_model1'] for r in results])
    skl_2 = np.array([r['skl_model2'] for r in results])
    
    summary_data = {
        '统计指标': ['KL均值', 'KL标准差', 'KL最大值', 'KL最小值', 
                   '对称KL均值', '模型胜出次数'],
        '模型1': [kl_1.mean(), kl_1.std(), kl_1.max(), kl_1.min(),
                 skl_1.mean(), np.sum(kl_1 < kl_2)],
        '模型2': [kl_2.mean(), kl_2.std(), kl_2.max(), kl_2.min(),
                 skl_2.mean(), np.sum(kl_2 < kl_1)],
    }
    df_summary = pd.DataFrame(summary_data)
    
    # ============================================================
    # Sheet 4: 模型参数配置
    # ============================================================
    config_data = {
        '参数项': ['ISO', '模型1 k值', '模型1 b值', '模型2 k值', '模型2 b值',
                  '分析亮度档位数', '综合更优模型'],
        '参数值': [
            iso,
            noise_params_1['k'],
            noise_params_1['b'],
            noise_params_2['k'],
            noise_params_2['b'],
            len(results),
            '模型1' if kl_1.mean() < kl_2.mean() else '模型2'
        ]
    }
    df_config = pd.DataFrame(config_data)
    
    # ============================================================
    # Sheet 5: 概率分布数据（用于后续分析）
    # ============================================================
    # 为每个亮度档位保存概率分布
    prob_data_list = []
    for r in results:
        bin_centers = (r['bin_edges'][:-1] + r['bin_edges'][1:]) / 2.0
        for i, (bc, pr, pf1, pf2) in enumerate(zip(
                bin_centers, r['prob_real'], r['prob_fake1'], r['prob_fake2'])):
            prob_data_list.append({
                '亮度档位': r['brightness_label'],
                'Bin中心': bc,
                'Real概率': pr,
                '模型1概率': pf1,
                '模型2概率': pf2,
            })
    df_prob = pd.DataFrame(prob_data_list)
    
    # ============================================================
    # 写入Excel文件
    # ============================================================
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_detail.to_excel(writer, sheet_name='KL散度详细结果', index=False)
        df_kl_matrix.to_excel(writer, sheet_name='KL矩阵', index=False)
        df_summary.to_excel(writer, sheet_name='统计汇总', index=False)
        df_config.to_excel(writer, sheet_name='模型配置', index=False)
        df_prob.to_excel(writer, sheet_name='概率分布数据', index=False)
    
    print(f"\n  Excel结果已保存: {output_path}")
    print(f"    - Sheet 'KL散度详细结果': 各亮度档位的完整分析结果")
    print(f"    - Sheet 'KL矩阵': KL散度矩阵数据")
    print(f"    - Sheet '统计汇总': 模型对比统计")
    print(f"    - Sheet '模型配置': 参数配置信息")
    print(f"    - Sheet '概率分布数据': 噪声分布概率数据")
    
    return output_path


def print_summary_statistics(results):
    """
    打印所有亮度档位的统计汇总

    包括各模型KL散度的均值、标准差、最大最小值
    """
    print(f"\n{'='*70}")
    print(f"  线性噪声模型KL散度评估 - 统计汇总")
    print(f"{'='*70}")

    kl_1 = np.array([r['kl_model1'] for r in results])
    kl_2 = np.array([r['kl_model2'] for r in results])
    skl_1 = np.array([r['skl_model1'] for r in results])
    skl_2 = np.array([r['skl_model2'] for r in results])

    print(f"\n  {'指标':<20} {'模型1':<20} {'模型2':<20}")
    print(f"  {'-'*60}")
    print(f"  {'KL均值':<20} {kl_1.mean():<20.6f} {kl_2.mean():<20.6f}")
    print(f"  {'KL标准差':<18} {kl_1.std():<20.6f} {kl_2.std():<20.6f}")
    print(f"  {'KL最大值':<18} {kl_1.max():<20.6f} {kl_2.max():<20.6f}")
    print(f"  {'KL最小值':<18} {kl_1.min():<20.6f} {kl_2.min():<20.6f}")
    print(f"  {'对称KL均值':<16} {skl_1.mean():<20.6f} {skl_2.mean():<20.6f}")

    # 综合判定
    print(f"\n  --- 综合评估 ---")
    if kl_1.mean() < kl_2.mean():
        winner = "模型1"
        ratio = kl_2.mean() / kl_1.mean()
    else:
        winner = "模型2"
        ratio = kl_1.mean() / kl_2.mean()

    print(f"  综合更优模型: {winner}")
    print(f"  优势比 (劣/优): {ratio:.2f}x")

    # 逐亮度档位胜负统计
    model1_wins = np.sum(kl_1 < kl_2)
    model2_wins = np.sum(kl_2 < kl_1)
    print(f"  模型1胜出档位数: {model1_wins}/{len(results)}")
    print(f"  模型2胜出档位数: {model2_wins}/{len(results)}")

    # 输出详细表格
    print(f"\n  {'亮度档位':<12} {'平均强度':<12} {'KL(模型1)':<14} {'KL(模型2)':<14} {'更优':<8}")
    print(f"  {'-'*60}")
    for r in results:
        better = "模型1" if r['kl_model1'] < r['kl_model2'] else "模型2"
        print(f"  {r['brightness_label']:<12} {r['mean_intensity']:<12.4f} "
              f"{r['kl_model1']:<14.6f} {r['kl_model2']:<14.6f} {better:<8}")


def run_with_real_data(data_root, iso, noise_params_1, noise_params_2, output_dir,
                       blc_root=None, bit_depth=14, raw_shape=None):
    """
    使用真实raw数据运行KL散度分析

    期望数据目录结构:
        data_root/
        ├── ISO200/
        │   ├── brightness_low/       # 低亮度档位
        │   │   ├── frame_001.raw
        │   │   ├── frame_002.raw
        │   │   └── ...
        │   ├── brightness_mid/       # 中亮度档位
        │   │   ├── frame_001.raw
        │   │   └── ...
        │   └── brightness_high/      # 高亮度档位
        │       ├── frame_001.raw
        │       └── ...
        └── ...

    BLC数据目录结构:
        blc_root/
        ├── ISO200/
        │   └── blc.npy (或多帧黑帧raw)
        ├── ISO400/
        │   └── blc.npy
        └── ...
    
    参数:
        data_root: 数据根目录
        iso: ISO值
        noise_params_1, noise_params_2: 噪声模型参数
        output_dir: 输出目录
        blc_root: BLC黑帧数据目录
        bit_depth: raw数据位深度
        raw_shape: raw文件shape (H, W)
    """
    # 加载BLC数据
    blc_value = load_blc_data(blc_root, iso, bit_depth)
    
    iso_dir = os.path.join(data_root, f"ISO{iso}")
    if not os.path.isdir(iso_dir):
        # 尝试其他命名方式
        iso_dir = os.path.join(data_root, f"iso{iso}")
    if not os.path.isdir(iso_dir):
        iso_dir = data_root  # 直接在data_root下查找亮度子目录

    # 获取所有亮度档位子目录
    brightness_dirs = sorted([
        d for d in os.listdir(iso_dir)
        if os.path.isdir(os.path.join(iso_dir, d))
    ])

    if len(brightness_dirs) == 0:
        raise FileNotFoundError(
            f"在 {iso_dir} 下未找到亮度档位子目录。\n"
            f"期望结构: {iso_dir}/brightness_xxx/ 目录，每个目录下包含多帧raw文件"
        )

    print(f"\n找到 {len(brightness_dirs)} 个亮度档位: {brightness_dirs}")
    if blc_value is not None:
        print(f"使用BLC黑电平值: {blc_value}")

    results = []
    for brightness_dir in brightness_dirs:
        folder_path = os.path.join(iso_dir, brightness_dir)
        print(f"\n  正在处理: {brightness_dir}")

        # 加载该亮度档位的所有帧（应用BLC校正）
        frames = load_raw_frames_with_blc(
            folder_path, 
            blc_value=blc_value,
            bit_depth=bit_depth,
            raw_shape=raw_shape
        )

        # 执行单个亮度档位的完整分析
        result = analyze_single_brightness(
            frames, noise_params_1, noise_params_2,
            brightness_label=brightness_dir
        )
        results.append(result)

    return results


def run_with_simulated_data(noise_params_1, noise_params_2, output_dir,
                            num_frames=50, image_size=(256, 256)):
    """
    使用模拟数据运行KL散度分析（用于验证流程和无真实数据时的测试）

    模拟不同亮度档位：从暗到亮，覆盖传感器动态范围
    模拟噪声使用一个"真实参数"生成，然后用2套参数去拟合

    参数:
        noise_params_1, noise_params_2: 2套待评估的噪声模型参数
        output_dir: 输出目录
        num_frames: 每个亮度档位的帧数
        image_size: 模拟图像尺寸 (H, W)
    """
    # 定义多个亮度档位（归一化值）
    # 模拟从暗场到接近饱和的不同曝光条件
    brightness_levels = [0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90]
    brightness_labels = ['极暗', '暗', '较暗', '中低', '中等', '中高', '较亮', '亮']

    # 模拟时使用的"真实"噪声参数
    # （实际场景中这就是传感器的真实物理特性）
    k_true = 0.0022
    b_true = 0.000008
    print(f"\n  模拟数据的真实噪声参数: k_true={k_true}, b_true={b_true}")
    print(f"  待评估模型1: k={noise_params_1['k']}, b={noise_params_1['b']}")
    print(f"  待评估模型2: k={noise_params_2['k']}, b={noise_params_2['b']}")

    results = []
    for level, label in zip(brightness_levels, brightness_labels):
        print(f"\n  正在处理亮度档位: {label} (intensity={level:.3f})")

        # 模拟该亮度档位的多帧raw数据
        frames = simulate_raw_frames(
            brightness_level=level,
            num_frames=num_frames,
            height=image_size[0],
            width=image_size[1],
            k_true=k_true,
            b_true=b_true
        )

        # 执行单个亮度档位的完整分析
        result = analyze_single_brightness(
            frames, noise_params_1, noise_params_2,
            brightness_label=f"{label}({level:.2f})"
        )
        results.append(result)

    return results


def main():
    """主函数：解析参数并执行KL散度评估流程"""

    parser = argparse.ArgumentParser(
        description='线性噪声模型KL散度评估工具（支持RAW格式和BLC校正）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用模拟数据测试
  python kl_count_function.py --simulate

  # 使用真实raw数据，指定BLC黑帧目录
  python kl_count_function.py --data_root /path/to/raw_data --blc_root /path/to/blc --iso 200

  # 指定raw文件格式参数
  python kl_count_function.py --data_root /path/to/raw_data --blc_root /path/to/blc --iso 200 --bit_depth 14 --raw_shape 3024 4032

  # 自定义噪声模型参数
  python kl_count_function.py --simulate --k1 0.002 --b1 0.00001 --k2 0.003 --b2 0.000005

  # 自定义帧数和图像尺寸（模拟模式）
  python kl_count_function.py --simulate --num_frames 100 --image_size 512

  # 指定输出Excel文件
  python kl_count_function.py --data_root /path/to/raw_data --blc_root /path/to/blc --output_excel my_results.xlsx
        """
    )

    # 数据源选择
    parser.add_argument('--data_root', type=str, default=None,
                        help='真实raw数据根目录路径')
    parser.add_argument('--blc_root', type=str, default=None,
                        help='BLC黑帧数据根目录路径（各ISO的黑帧数据）')
    parser.add_argument('--simulate', action='store_true',
                        help='使用模拟数据运行（无需真实raw文件）')
    parser.add_argument('--iso', type=int, default=200,
                        help='ISO值 (默认: 200)')

    # RAW格式参数
    parser.add_argument('--bit_depth', type=int, default=14,
                        help='RAW数据位深度 (默认: 14)')
    parser.add_argument('--raw_shape', type=int, nargs=2, default=None,
                        metavar=('H', 'W'),
                        help='RAW文件尺寸，格式: 高度 宽度 (如 3024 4032)')

    # 噪声模型参数
    parser.add_argument('--k1', type=float, default=None,
                        help='模型1的k参数 (默认使用脚本内置值)')
    parser.add_argument('--b1', type=float, default=None,
                        help='模型1的b参数 (默认使用脚本内置值)')
    parser.add_argument('--k2', type=float, default=None,
                        help='模型2的k参数 (默认使用脚本内置值)')
    parser.add_argument('--b2', type=float, default=None,
                        help='模型2的b参数 (默认使用脚本内置值)')

    # 模拟参数
    parser.add_argument('--num_frames', type=int, default=50,
                        help='每个亮度档位的帧数 (默认: 50)')
    parser.add_argument('--image_size', type=int, default=256,
                        help='模拟图像尺寸 (默认: 256x256)')
    parser.add_argument('--num_bins', type=int, default=200,
                        help='直方图bin数量 (默认: 200)')

    # 输出
    parser.add_argument('--output_dir', type=str, default='./kl_results',
                        help='结果输出目录 (默认: ./kl_results)')
    parser.add_argument('--output_excel', type=str, default=None,
                        help='Excel输出文件名 (默认: kl_results_ISO{iso}.xlsx)')
    parser.add_argument('--no_plot', action='store_true',
                        help='不生成可视化图')
    parser.add_argument('--no_excel', action='store_true',
                        help='不生成Excel文件')

    args = parser.parse_args()

    # 更新噪声模型参数（如果命令行指定了）
    noise_params_1 = NOISE_PARAMS_1.copy()
    noise_params_2 = NOISE_PARAMS_2.copy()

    if args.k1 is not None:
        noise_params_1['k'] = args.k1
    if args.b1 is not None:
        noise_params_1['b'] = args.b1
    if args.k2 is not None:
        noise_params_2['k'] = args.k2
    if args.b2 is not None:
        noise_params_2['b'] = args.b2

    # 更新label
    noise_params_1['label'] = f"模型1 (k={noise_params_1['k']}, b={noise_params_1['b']})"
    noise_params_2['label'] = f"模型2 (k={noise_params_2['k']}, b={noise_params_2['b']})"

    # 处理raw_shape参数
    raw_shape = tuple(args.raw_shape) if args.raw_shape else None

    # ============================================================
    # 打印运行配置
    # ============================================================
    print(f"\n{'#'*70}")
    print(f"#  ISO{args.iso} 线性噪声模型KL散度评估")
    print(f"#  噪声模型: σ² = k·I + b")
    print(f"#  {noise_params_1['label']}")
    print(f"#  {noise_params_2['label']}")
    if args.blc_root:
        print(f"#  BLC数据目录: {args.blc_root}")
    print(f"#  位深度: {args.bit_depth}-bit")
    if raw_shape:
        print(f"#  RAW尺寸: {raw_shape[0]} x {raw_shape[1]}")
    print(f"{'#'*70}")

    # ============================================================
    # 执行分析
    # ============================================================
    if args.simulate or args.data_root is None:
        if args.data_root is None and not args.simulate:
            print("\n  [提示] 未指定--data_root，自动使用模拟数据模式")
            print("  [提示] 如需使用真实数据，请指定: --data_root /path/to/data")

        results = run_with_simulated_data(
            noise_params_1=noise_params_1,
            noise_params_2=noise_params_2,
            output_dir=args.output_dir,
            num_frames=args.num_frames,
            image_size=(args.image_size, args.image_size)
        )
    else:
        results = run_with_real_data(
            data_root=args.data_root,
            iso=args.iso,
            noise_params_1=noise_params_1,
            noise_params_2=noise_params_2,
            output_dir=args.output_dir,
            blc_root=args.blc_root,
            bit_depth=args.bit_depth,
            raw_shape=raw_shape
        )

    # ============================================================
    # 输出统计汇总
    # ============================================================
    print_summary_statistics(results)

    # ============================================================
    # 可视化
    # ============================================================
    if not args.no_plot:
        plot_results(results, args.output_dir)

    # ============================================================
    # Excel输出
    # ============================================================
    if not args.no_excel:
        excel_filename = args.output_excel or f"kl_results_ISO{args.iso}.xlsx"
        excel_path = os.path.join(args.output_dir, excel_filename)
        export_results_to_excel(
            results=results,
            output_path=excel_path,
            iso=args.iso,
            noise_params_1=noise_params_1,
            noise_params_2=noise_params_2
        )

    print(f"\n  分析完成! 结果保存在: {os.path.abspath(args.output_dir)}")
    print(f"{'#'*70}\n")

    return results


if __name__ == '__main__':
    main()
