"""
测试 Unprocessing Pipeline 的可视化功能

快速体验每个逆ISP步骤的可视化效果

用法:
    python test_visualization.py
"""

import numpy as np
import cv2
from unprocessing import UnprocessingPipeline
import os


def create_test_image():
    """
    创建一个测试图像（包含各种颜色和亮度）
    如果没有测试图像，可以生成一个合成图像
    """
    # 创建一个渐变图像，包含：
    # - 红色块
    # - 绿色块
    # - 蓝色块
    # - 白色高光
    # - 黑色阴影
    # - 灰色中间调
    
    height, width = 512, 512
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 第一行：RGB三原色
    img[0:170, 0:170] = [255, 0, 0]      # 红色
    img[0:170, 171:341] = [0, 255, 0]    # 绿色
    img[0:170, 342:512] = [0, 0, 255]    # 蓝色
    
    # 第二行：CMY次色
    img[171:341, 0:170] = [0, 255, 255]   # 青色
    img[171:341, 171:341] = [255, 0, 255] # 品红
    img[171:341, 342:512] = [255, 255, 0] # 黄色
    
    # 第三行：灰度渐变
    for i in range(512):
        gray_value = int(255 * i / 512)
        img[342:512, i] = [gray_value, gray_value, gray_value]
    
    return img


def test_with_sample_image(image_path):
    """使用指定的图像进行测试"""
    
    print("=" * 70)
    print("Unprocessing Pipeline 可视化测试")
    print("=" * 70)
    
    # 读取图像
    if os.path.exists(image_path):
        print(f"\n✓ 使用测试图像: {image_path}")
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        print(f"\n✗ 未找到图像 {image_path}")
        print("✓ 生成合成测试图像...")
        img = create_test_image()
        # 保存合成图像
        cv2.imwrite('test_synthetic.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"✓ 合成图像已保存: test_synthetic.png")
    
    print(f"  图像形状: {img.shape}")
    print(f"  图像类型: {img.dtype}")
    print(f"  值范围: [{img.min()}, {img.max()}]")
    
    # 创建 Unprocessing Pipeline（启用可视化）
    print("\n" + "=" * 70)
    print("创建 Pipeline（可视化模式）")
    print("=" * 70)
    
    unprocessor = UnprocessingPipeline(
        random_ccm=True,
        random_gains=True,
        add_noise=True,
        bayer_pattern='RGGB',
        visualize=True,              # 启用可视化
        output_dir='visualization'   # 可视化保存目录
    )
    
    # 执行 Unprocessing
    print("\n" + "=" * 70)
    print("开始处理...")
    print("=" * 70)
    
    raw_bayer, metadata = unprocessor.unprocess(
        img,
        iso=1600,
        verbose=True
    )
    
    # 保存结果
    from unprocessing import save_raw_bayer
    save_raw_bayer(raw_bayer, 'output_raw_bayer.png', bit_depth=12)
    
    # 总结
    print("\n" + "=" * 70)
    print("✓ 测试完成！")
    print("=" * 70)
    
    print("\n📁 可视化结果已保存到 visualization/ 目录：")
    print("   ├── step1_inverse_gamma.png          - 步骤1: 逆Gamma校正")
    print("   ├── step2_inverse_tone_mapping.png   - 步骤2: 逆色调映射")
    print("   ├── step3_inverse_color_correction.png - 步骤3: 逆色彩校正")
    print("   ├── step4_inverse_white_balance.png  - 步骤4: 逆白平衡")
    print("   ├── step5_mosaic.png                 - 步骤5: Mosaic马赛克化")
    print("   ├── step6_add_noise.png              - 步骤6: 添加噪声")
    print("   ├── step6_noise_detail.png           - 步骤6: 噪声详细分析")
    print("   └── pipeline_summary.png             - ⭐ 完整流程总结")
    
    print("\n📊 Raw Bayer 输出：")
    print(f"   output_raw_bayer.png")
    
    print("\n💡 下一步：")
    print("   1. 打开 visualization/pipeline_summary.png 查看完整流程")
    print("   2. 依次查看每个步骤的可视化，理解数据退化过程")
    print("   3. 对比 step6_add_noise.png 和 step6_noise_detail.png 理解噪声模型")
    
    print("\n" + "=" * 70)


def test_different_iso():
    """测试不同ISO的效果"""
    
    print("\n" + "=" * 70)
    print("测试不同ISO的噪声效果")
    print("=" * 70)
    
    # 创建测试图像
    img = create_test_image()
    
    iso_values = [400, 800, 1600, 3200]
    
    for iso in iso_values:
        print(f"\n处理 ISO {iso}...")
        
        unprocessor = UnprocessingPipeline(
            random_ccm=False,  # 固定CCM，便于对比
            random_gains=False,  # 固定白平衡
            add_noise=True,
            visualize=True,
            output_dir=f'visualization_iso{iso}'
        )
        
        raw, metadata = unprocessor.unprocess(img, iso=iso, verbose=False)
        
        print(f"  ✓ ISO {iso} 完成")
        print(f"  噪声水平: {np.std(raw):.6f}")
        print(f"  可视化保存: visualization_iso{iso}/")
    
    print("\n" + "=" * 70)
    print("✓ 不同ISO测试完成！")
    print("=" * 70)
    print("\n可以对比不同ISO目录下的 step6_noise_detail.png")


def test_different_bayer_patterns():
    """测试不同Bayer模式"""
    
    print("\n" + "=" * 70)
    print("测试不同Bayer Pattern")
    print("=" * 70)
    
    img = create_test_image()
    
    patterns = ['RGGB', 'BGGR', 'GRBG', 'GBRG']
    
    for pattern in patterns:
        print(f"\n处理 {pattern}...")
        
        unprocessor = UnprocessingPipeline(
            random_ccm=False,
            random_gains=False,
            add_noise=False,  # 不加噪声，便于观察Bayer模式
            bayer_pattern=pattern,
            visualize=True,
            output_dir=f'visualization_{pattern}'
        )
        
        raw, metadata = unprocessor.unprocess(img, verbose=False)
        
        print(f"  ✓ {pattern} 完成")
        print(f"  可视化保存: visualization_{pattern}/")
    
    print("\n" + "=" * 70)
    print("✓ 不同Bayer Pattern测试完成！")
    print("=" * 70)
    print("\n可以对比不同模式下的 step5_mosaic.png 和 pipeline_summary.png")


def main():
    """主测试函数"""
    
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║     Unprocessing Pipeline 可视化测试                              ║
║                                                                   ║
║     目标：可视化理解每个逆ISP步骤                                  ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    import sys
    
    # 检查是否提供了测试图像
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    else:
        # 尝试常见的测试图像
        test_candidates = [
            'test.jpg', 'test.png',
            'sample.jpg', 'sample.png',
            'lena.jpg', 'lena.png'
        ]
        
        test_image = None
        for candidate in test_candidates:
            if os.path.exists(candidate):
                test_image = candidate
                break
        
        if test_image is None:
            test_image = 'test_synthetic.png'  # 将使用合成图像
    
    # 测试1: 基本可视化
    print("\n【测试 1】基本可视化流程")
    test_with_sample_image(test_image)
    
    # 询问是否继续高级测试
    print("\n" + "=" * 70)
    response = input("\n是否继续进行高级测试？(y/N): ").strip().lower()
    
    if response == 'y':
        # 测试2: 不同ISO
        print("\n【测试 2】不同ISO对比")
        test_different_iso()
        
        # 测试3: 不同Bayer模式
        print("\n【测试 3】不同Bayer Pattern对比")
        test_different_bayer_patterns()
    
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║     ✓ 所有测试完成！                                              ║
║                                                                   ║
║     查看可视化结果，深入理解逆ISP流程                              ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """)


if __name__ == '__main__':
    main()

