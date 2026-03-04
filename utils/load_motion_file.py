import yaml  # 导入PyYAML库
import os
import glob
from typing import Dict, List

def read_yaml_file(file_path: str) -> Dict:
    """
    读取指定的YAML文件并返回其内容。
    
    :param file_path: YAML文件的路径。
    :return: YAML文件的内容字典，若读取失败则返回空字典。
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
        return data if data is not None else {}
    except FileNotFoundError:
        print(f"文件 {file_path} 不存在。")
        return {}
    except yaml.YAMLError as exc:
        print(f"YAML解析错误: {exc}")
        return {}

def collect_npz_paths(yaml_path: str = "motion_file.yaml") -> Dict[str, List[str]]:
    """
    从指定的YAML文件中读取配置，按照motion_group分组收集NPZ文件路径列表。
    - 对于每个motion_group下的组：
      - 先添加file_name中指定的NPZ文件路径。
      - 然后添加folder_name中每个文件夹下的所有NPZ文件路径，但避免与file_name中文件名的重复（基于文件名）。
      - 最后剔除wo_file_name中指定的文件路径和wo_folder_name中文件夹下的所有NPZ文件路径。
    
    :param yaml_path: YAML文件的路径，默认为"motion_file.yaml"。
    :return: 以组名为键的字典，每个值为该组的NPZ文件路径列表。
    """
    data = read_yaml_file(yaml_path)
    motion_groups = data.get('motion_group', {})
    result = {}

    for group_name, group_data in motion_groups.items():
        # 提取组配置
        file_names = group_data.get('file_name', [])
        folder_names = group_data.get('folder_name', [])
        wo_file_names = group_data.get('wo_file_name', [])
        wo_folder_names = group_data.get('wo_folder_name', [])

        # 使用集合存储路径，以避免重复
        npz_paths = set()
        existing_basenames = set()

        # 第一步：添加file_name中的NPZ文件路径，并记录其basename
        for path in file_names:
            if path.endswith('.npz') and os.path.exists(path):
                npz_paths.add(path)
                existing_basenames.add(os.path.basename(path))

        # 第二步：添加folder_name中每个文件夹及其子目录下的NPZ文件路径，避免与现有basename重复
        for folder in folder_names:
            if os.path.isdir(folder):
                pattern = os.path.join(folder, '**', '*.npz')
                for npz_file in glob.glob(pattern, recursive=True):
                    basename = os.path.basename(npz_file)
                    if basename not in existing_basenames:
                        npz_paths.add(npz_file)
                        existing_basenames.add(basename)  # 更新basename集合，避免后续重复

        # 第三步：剔除wo_file_name中的指定文件路径
        for wo_path in wo_file_names:
            npz_paths.discard(wo_path)

        # 第四步：剔除wo_folder_name中每个文件夹及其子目录下的所有NPZ文件路径
        for wo_folder in wo_folder_names:
            if os.path.isdir(wo_folder):
                pattern = os.path.join(wo_folder, '**', '*.npz')
                for npz_file in glob.glob(pattern, recursive=True):
                    npz_paths.discard(npz_file)

        # 保存排序后的列表
        result[group_name] = sorted(list(npz_paths))

    return result

if __name__ == "__main__":
    npz_groups = collect_npz_paths("configs/data/motion_file.yaml")
    print("Collected NPZ file paths by motion group:")
    for group_name, paths in npz_groups.items():
        print(f"\nGroup: {group_name}")
        for path in paths:
            print(path)
    first_key = next(iter(npz_groups))
    print(f"\nfisrt_key: {first_key}")
    first_value = next(iter(npz_groups.values()))
    print(f"\nTotal NPZ files in the first group: {len(first_value)}")
    for path in first_value:
        print(f" - {path}")
