"""
测试 Unprocessing Pipeline
快速验证代码功能
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from unprocessing import UnprocessingPipeline, save_raw_bayer, visualize_bayer


def create_test_image():
    """创建测试图像（如果没有现成的图像）"""
    # 创建一个彩色渐变图像
    h, w = 512, 512
    test_img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # R 通道：水平渐变
    test_img[:, :, 0] = np.linspace(0, 255, w).astype(np.uint8)
    
    # G 通道：垂直渐变
    test_img[:, :, 1] = np.linspace(0, 255, h).reshape(-1, 1).astype(np.uint8)
    
    # B 通道：对角渐变
    x, y = np.meshgrid(np.linspace(0, 255, w), np.linspace(0, 255, h))
    test_img[:, :, 2] = ((x + y) / 2).astype(np.uint8)
    
    cv2.imwrite('test_image_synthetic.png', cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR))
    print("创建了合成测试图像: test_image_synthetic.png")
    
    return test_img


def test_basic_unprocessing():
    """测试1：基本功能"""
    print("\n" + "="*60)
    print("测试1：基本 Unprocessing 功能")
    print("="*60)
    
    # 尝试读取图像，如果不存在则创建
    try:
        # srgb = cv2.imread('test_image_synthetic.png')
        # testPath = f'D:\\02-Study\\02-dataset\\03-test-ISP\\eric_person.jpg'
        srgb = cv2.imread('test_image_synthetic.png')
        if srgb is None:
            raise FileNotFoundError
        srgb = cv2.cvtColor(srgb, cv2.COLOR_BGR2RGB)
    except:
        print("未找到测试图像，创建合成图像...")
        srgb = create_test_image()
    
    # 创建 Pipeline
    unprocessor = UnprocessingPipeline(
        random_ccm=False,
        random_gains=False,
        add_noise=True
    )
    
    # Unprocessing
    raw_bayer, metadata = unprocessor.unprocess(srgb, iso=1600)
    
    # 验证
    assert raw_bayer.shape == srgb.shape[:2], f"形状错误：{raw_bayer.shape} vs {srgb.shape[:2]}"
    assert raw_bayer.min() >= 0 and raw_bayer.max() <= 1, f"范围错误：[{raw_bayer.min()}, {raw_bayer.max()}]"
    
    print(f"✅ 基本功能测试通过")
    print(f"   输入: {srgb.shape}")
    print(f"   输出: {raw_bayer.shape}")
    print(f"   ISO: {metadata['iso']}")
    
    # 保存
    save_raw_bayer(raw_bayer, 'test_output_raw.png')
    
    return raw_bayer, metadata


def test_different_iso():
    """测试2：不同 ISO 的噪声"""
    print("\n" + "="*60)
    print("测试2：不同 ISO 的噪声水平")
    print("="*60)
    
    # 读取图像
    try:
        srgb = cv2.imread('test_image_synthetic.png')
        srgb = cv2.cvtColor(srgb, cv2.COLOR_BGR2RGB)
    except:
        srgb = create_test_image()
    
    # 生成干净 Raw
    unprocessor = UnprocessingPipeline(add_noise=False)
    clean_raw, _ = unprocessor.unprocess(srgb, verbose=False)
    
    # 测试不同 ISO
    iso_values = [400, 800, 1600, 3200, 6400]
    noise_levels = []
    
    print(f"\n{'ISO':<8} {'噪声标准差':<15} {'PSNR (dB)':<12}")
    print("-" * 40)
    
    for iso in iso_values:
        # 添加噪声
        noisy_raw = unprocessor.step6_add_noise(clean_raw, iso)
        
        # 计算噪声水平
        noise = noisy_raw - clean_raw
        noise_std = np.std(noise)
        noise_levels.append(noise_std)
        
        # 计算 PSNR
        mse = np.mean((clean_raw - noisy_raw) ** 2)
        psnr = 20 * np.log10(1.0 / np.sqrt(mse + 1e-10))
        
        print(f"{iso:<8} {noise_std:<15.6f} {psnr:<12.2f}")
    
    # 验证：噪声应该随 ISO 单调递增
    for i in range(len(noise_levels) - 1):
        assert noise_levels[i] < noise_levels[i+1], "噪声未随 ISO 递增！"
    
    print(f"\n✅ ISO 噪声测试通过（噪声随 ISO 单调递增）")
    
    return noise_levels


def test_bayer_patterns():
    """测试3：不同 Bayer 模式"""
    print("\n" + "="*60)
    print("测试3：不同 Bayer 模式")
    print("="*60)
    
    try:
        srgb = cv2.imread('test_image_synthetic.png')
        srgb = cv2.cvtColor(srgb, cv2.COLOR_BGR2RGB)
    except:
        srgb = create_test_image()
    
    # 测试所有 Bayer 模式
    patterns = ['RGGB', 'BGGR', 'GRBG', 'GBRG']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    for i, pattern in enumerate(patterns):
        unprocessor = UnprocessingPipeline(
            add_noise=False,
            bayer_pattern=pattern
        )
        
        raw_bayer, _ = unprocessor.unprocess(srgb, verbose=False)
        
        # 简单去马赛克
        bayer_uint8 = (raw_bayer * 255).astype(np.uint8)
        
        # 选择对应的 OpenCV 转换模式
        cv_pattern = {
            'RGGB': cv2.COLOR_BAYER_RGGB2RGB,
            'BGGR': cv2.COLOR_BAYER_BGGR2RGB,
            'GRBG': cv2.COLOR_BAYER_GRBG2RGB,
            'GBRG': cv2.COLOR_BAYER_GBRG2RGB,
        }[pattern]
        
        rgb_demosaic = cv2.cvtColor(bayer_uint8, cv_pattern)
        
        ax = axes[i//2, i%2]
        ax.imshow(rgb_demosaic)
        ax.set_title(f'Bayer Pattern: {pattern}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('bayer_patterns_test.png', dpi=150)
    print(f"✅ Bayer 模式测试通过")
    print(f"   可视化保存到: bayer_patterns_test.png")


def test_visualization():
    """测试4：可视化功能"""
    print("\n" + "="*60)
    print("测试4：可视化功能")
    print("="*60)
    
    try:
        srgb = cv2.imread('test_image_synthetic.png')
        srgb = cv2.cvtColor(srgb, cv2.COLOR_BGR2RGB)
    except:
        srgb = create_test_image()
    
    # Unprocessing
    unprocessor = UnprocessingPipeline()
    raw_bayer, metadata = unprocessor.unprocess(srgb, iso=1600, verbose=False)
    
    # 可视化
    visualize_bayer(raw_bayer, save_path='test_visualization.png')
    
    print(f"✅ 可视化测试通过")
    print(f"   保存到: test_visualization.png")


def test_step_by_step():
    """测试5：逐步执行"""
    print("\n" + "="*60)
    print("测试5：逐步执行每个步骤")
    print("="*60)
    
    try:
        srgb = cv2.imread('test_image_synthetic.png')
        srgb = cv2.cvtColor(srgb, cv2.COLOR_BGR2RGB)
    except:
        srgb = create_test_image()
    
    srgb_norm = srgb.astype(np.float32) / 255.0
    
    unprocessor = UnprocessingPipeline(
        random_ccm=False,
        random_gains=False,
        add_noise=False
    )
    
    # 逐步执行
    print("\n逐步执行：")
    
    # Step 1
    linear = unprocessor.step1_inverse_gamma(srgb_norm)
    print(f"Step 1 完成: 逆 Gamma")
    print(f"  输出范围: [{linear.min():.4f}, {linear.max():.4f}]")
    
    # Step 2
    linear_scaled, scale = unprocessor.step2_inverse_tone_mapping(linear)
    print(f"Step 2 完成: 逆色调映射")
    print(f"  缩放因子: {scale:.4f}")
    print(f"  输出范围: [{linear_scaled.min():.4f}, {linear_scaled.max():.4f}]")
    
    # Step 3
    camera_rgb = unprocessor.step3_inverse_color_correction(
        linear_scaled,
        unprocessor.ccm_inv_mean
    )
    print(f"Step 3 完成: 逆色彩校正")
    print(f"  输出范围: [{camera_rgb.min():.4f}, {camera_rgb.max():.4f}]")
    
    # Step 4
    rgb_inv_wb, gains = unprocessor.step4_inverse_white_balance(camera_rgb)
    print(f"Step 4 完成: 逆白平衡")
    print(f"  Gains: R={gains['red']:.3f}, G={gains['green']:.3f}, B={gains['blue']:.3f}")
    print(f"  输出范围: [{rgb_inv_wb.min():.4f}, {rgb_inv_wb.max():.4f}]")
    
    # Step 5
    bayer = unprocessor.step5_mosaic(rgb_inv_wb)
    print(f"Step 5 完成: Mosaic")
    print(f"  形状变化: {rgb_inv_wb.shape} → {bayer.shape}")
    print(f"  输出范围: [{bayer.min():.4f}, {bayer.max():.4f}]")
    
    # Step 6
    bayer_noisy = unprocessor.step6_add_noise(bayer, iso=1600)
    print(f"Step 6 完成: 添加噪声")
    noise_level = np.std(bayer_noisy - bayer)
    print(f"  噪声水平: {noise_level:.6f}")
    print(f"  输出范围: [{bayer_noisy.min():.4f}, {bayer_noisy.max():.4f}]")
    
    print(f"\n✅ 逐步执行测试通过")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("开始运行所有测试")
    print("="*60)
    
    try:
        # 测试1：基本功能
        raw_bayer, metadata = test_basic_unprocessing()
        
        # 测试2：不同 ISO
        noise_levels = test_different_iso()
        
        # 测试3：Bayer 模式
        test_bayer_patterns()
        
        # 测试4：可视化
        test_visualization()
        
        # 测试5：逐步执行
        test_step_by_step()
        
        print("\n" + "="*60)
        print("✅ 所有测试通过！")
        print("="*60)
        print("\n生成的文件：")
        print("  - test_output_raw.png (Raw Bayer 图像)")
        print("  - test_visualization.png (可视化)")
        print("  - bayer_patterns_test.png (Bayer 模式对比)")
        print("  - test_image_synthetic.png (合成测试图像)")
        print("\n代码可以正常使用！")
        
    except Exception as e:
        print("\n" + "="*60)
        print("❌ 测试失败")
        print("="*60)
        print(f"错误信息: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    run_all_tests()

