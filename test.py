def count_occurrences(file_path, target_string):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            occurrences = content.count(target_string)
            return occurrences
    except FileNotFoundError:
        print("文件未找到！")
        return -1

# 请将 'your_file.txt' 替换为你的文件路径，将 'target_string' 替换为你想要查找的特定字符串
file_path = r"D:\Files\赛博情绪垃圾桶\0423.txt"
# target_string = '叶叶 2024/4/23'
target_string = "Scabbards. 2024/4/23"

result = count_occurrences(file_path, target_string)
if result != -1:
    print(f"'{target_string}' 出现的次数为: {result}")
