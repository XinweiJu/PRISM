from collections import defaultdict

# 定义文件路径
for splits in ['train', 'val', 'test']:
    train_files_path = f"/Workspace/Baselines/monodepth2_feast_ac_rema_intervals/splits/hk_interval/{splits}_files.txt"
    new_train_files_path = f"/Workspace/Baselines/monodepth2_feast_ac_rema_intervals/splits/hk_interval/{splits}_files_interval20.txt"

    # 读取 train_files.txt 内容
    with open(train_files_path, 'r') as f:
        lines = f.readlines()

    # 使用字典按 subsequence 进行分类
    subseq_dict = defaultdict(list)

    for line in lines:
        line = line.strip()
        parts = line.split('/')
        if len(parts) > 3:
            subseq = parts[-2]  # 例如 cecum_t1_a
            subseq_dict[subseq].append(line)

    # 处理每个 subsequence，删除前四个和最后四个
    filtered_lines = []
    for subseq, paths in subseq_dict.items():
        if len(paths) > 8:  # 确保有足够的文件进行裁剪
            filtered_paths = paths[19:-19]
            filtered_lines.extend(filtered_paths)
        else:
            print(f"Skipping {subseq} as it has less than 9 images.")

    # 重新写入文件
    with open(new_train_files_path, 'w') as f:
        for line in filtered_lines:
            f.write(line + '\n')

    print("Processing complete. Updated train_files.txt.")
