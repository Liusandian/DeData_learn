"""
测试 Bayer 到 RGB 的去马赛克功能

演示如何使用不同的插值方法进行去马赛克
"""

import numpy as np
import cv2
from unprocessing import (
    UnprocessingPipeline, 
    bayer_to_rgb,
    compare_demosaic_methods,
    visualize_bayer
)


def test_demosaic_methods():
    """测试不同的去马赛克方法"""
    
    print("=" * 70)
    print("Bayer 去马赛克测试")
    print("=" * 70)
    
    # 1. 读取一张 sRGB 图像
    img_path = "D://04-dataset//test_isp//shevy_and_son.jpg"
    print(f"\n1. 读取测试图像: {img_path}")
    
    srgb = cv2.imread(img_path)
    if srgb is None:
        print("错误：无法读取图像，请检查路径")
        return
    
    srgb = cv2.cvtColor(srgb, cv2.COLOR_BGR2RGB)
    print(f"   图像形状: {srgb.shape}")
    
    # 2. 转换为 Bayer 图像
    print("\n2. 转换为 Bayer Raw 图像")
    unprocessor = UnprocessingPipeline(
        random_ccm=False,  # 固定 CCM
        random_gains=False,  # 固定白平衡
        add_noise=False,  # 不添加噪声
        visualize=False
    )
    
    bayer, metadata = unprocessor.unprocess(srgb, verbose=False)
    print(f"   Bayer 形状: {bayer.shape}")
    print(f"   Bayer 模式: {metadata['bayer_pattern']}")
    
    # 3. 使用不同方法去马赛克
    print("\n3. 测试不同的去马赛克方法")
    print("-" * 70)
    
    bayer_pattern = metadata['bayer_pattern']
    
    # 方法 1: 双线性插值
    print("\n方法 1: 双线性插值（手动实现）")
    rgb_bilinear = bayer_to_rgb(bayer, bayer_pattern=bayer_pattern, method='bilinear')
    print(f"   输出形状: {rgb_bilinear.shape}")
    print(f"   值范围: [{rgb_bilinear.min():.4f}, {rgb_bilinear.max():.4f}]")
    
    # 方法 2: OpenCV 内置
    print("\n方法 2: OpenCV 内置方法")
    rgb_opencv = bayer_to_rgb(bayer, bayer_pattern=bayer_pattern, method='opencv')
    print(f"   输出形状: {rgb_opencv.shape}")
    print(f"   值范围: [{rgb_opencv.min():.4f}, {rgb_opencv.max():.4f}]")
    
    # 方法 3: 边缘感知
    print("\n方法 3: 边缘感知插值")
    rgb_edge_aware = bayer_to_rgb(bayer, bayer_pattern=bayer_pattern, method='edge_aware')
    print(f"   输出形状: {rgb_edge_aware.shape}")
    print(f"   值范围: [{rgb_edge_aware.min():.4f}, {rgb_edge_aware.max():.4f}]")
    
    # 4. 可视化单个方法
    print("\n4. 可视化 OpenCV 方法结果")
    visualize_bayer(
        bayer, 
        bayer_pattern=bayer_pattern,
        save_path='demosaic_opencv.png',
        method='opencv'
    )
    
    # 5. 对比所有方法
    print("\n5. 生成方法对比图")
    compare_demosaic_methods(
        bayer,
        bayer_pattern=bayer_pattern,
        save_path='demosaic_comparison.png'
    )
    
    # 6. 保存各方法结果
    print("\n6. 保存各方法的 RGB 结果")
    cv2.imwrite('demosaic_bilinear.png', 
                cv2.cvtColor((rgb_bilinear * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite('demosaic_opencv.png', 
                cv2.cvtColor((rgb_opencv * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite('demosaic_edge_aware.png', 
                cv2.cvtColor((rgb_edge_aware * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    
    print("   ✓ 已保存: demosaic_bilinear.png")
    print("   ✓ 已保存: demosaic_opencv.png")
    print("   ✓ 已保存: demosaic_edge_aware.png")
    
    print("\n" + "=" * 70)
    print("✓ 测试完成！")
    print("=" * 70)
    
    print("\n📊 输出文件：")
    print("   1. demosaic_opencv.png - OpenCV 方法的 Bayer + RGB 对比")
    print("   2. demosaic_comparison.png - 三种方法的对比图")
    print("   3. demosaic_bilinear.png - 双线性插值结果")
    print("   4. demosaic_opencv.png - OpenCV 方法结果")
    print("   5. demosaic_edge_aware.png - 边缘感知插值结果")


def test_different_bayer_patterns():
    """测试不同的 Bayer 模式"""
    
    print("\n" + "=" * 70)
    print("测试不同 Bayer Pattern")
    print("=" * 70)
    
    # 读取图像
    img_path = "D://04-dataset//test_isp//shevy_and_son.jpg"
    srgb = cv2.imread(img_path)
    srgb = cv2.cvtColor(srgb, cv2.COLOR_BGR2RGB)
    
    patterns = ['RGGB', 'BGGR', 'GRBG', 'GBRG']
    
    for pattern in patterns:
        print(f"\n测试 Bayer Pattern: {pattern}")
        
        # 创建 Pipeline
        unprocessor = UnprocessingPipeline(
            random_ccm=False,
            random_gains=False,
            add_noise=False,
            bayer_pattern=pattern,
            visualize=False
        )
        
        # 转换为 Bayer
        bayer, _ = unprocessor.unprocess(srgb, verbose=False)
        
        # 去马赛克
        rgb = bayer_to_rgb(bayer, bayer_pattern=pattern, method='opencv')
        
        # 保存
        output_path = f'demosaic_{pattern.lower()}.png'
        cv2.imwrite(output_path, 
                   cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        print(f"   ✓ 已保存: {output_path}")
    
    print("\n" + "=" * 70)
    print("✓ 不同 Bayer Pattern 测试完成！")
    print("=" * 70)


if __name__ == '__main__':
    # 测试 1: 不同去马赛克方法
    test_demosaic_methods()
    
    # 测试 2: 不同 Bayer 模式
    # test_different_bayer_patterns()

