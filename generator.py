# create_folders.py
import os


def create_project_structure():
    """在当前目录创建项目文件夹结构"""
    project_folders = [
        'raw_data',  # 存放原始数据文件
        'processed_data',  # 存放处理后的股票数据
        'results',  # 存放分析结果
    ]

    for folder in project_folders:
        try:
            os.makedirs(folder, exist_ok=True)
            print(f"✓ 已创建文件夹: {folder}")
        except Exception as e:
            print(f"✗ 创建文件夹失败 {folder}: {e}")

    print("\n✅ 项目文件夹结构创建完成！")

    # 显示文件夹结构
    print("\n📁 项目结构:")
    print("当前目录/")
    print("├── raw_data/          # 存放原始数据文件")
    print("├── processed_data/    # 存放处理后的股票数据")
    print("└── results/           # 存放分析结果")

    return True


if __name__ == "__main__":
    create_project_structure()