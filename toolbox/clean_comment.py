import re
import os


def is_pass_line(line):
    """检查该行是否应该被忽略，基于正则表达式模式。"""
    regular_expressions = [
        r"/'.*#.*/'",
        r'".*#.*"',
        r"/'/'/'.*#.*/'/'/",
        r'"/"/".*#.*"/"/"'
    ]
    for pattern in regular_expressions:
        if re.search(pattern, line):
            return True
    return False


def remove_comments_and_blank_lines(content):
    """去除内容中的单行和多行注释。"""
    # 去除多行注释（文档字符串）
    content = re.sub(r'""".*?"""', '', content, flags=re.DOTALL)
    content = re.sub(r"'''.*?'''", '', content, flags=re.DOTALL)

    # 去除单行注释
    lines = content.splitlines()
    cleaned_lines = []
    for line in lines:
        index = line.find('#')
        if index != -1:
            line = line[:index]  # 保留注释前的代码
        if line.strip():  # 添加非空行
            cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)


def read_file(file_name):
    """读取并处理文件，去除注释和空行。"""
    with open(file_name, 'r', encoding='utf-8') as f:
        content = f.read()

    new_str = remove_comments_and_blank_lines(content)
    log_str = f'/n%20s/n' % (os.path.basename(file_name))  # 输出日志

    return new_str, log_str


def make_clean_file(src_path, desc_path, file_list):
    """创建去除注释的清理文件。"""
    full_str = ""
    for file in file_list:
        cur_str, log_str = read_file(os.path.join(src_path, file))
        with open(os.path.join(desc_path, f'without_note_{file}'), 'w', encoding='utf-8') as fNew:
            fNew.write(cur_str)
        full_str += cur_str
    return full_str


def read_original_code(alg_name="genetic_algorithm"):
    with open(f"./code_base/{alg_name}/algorithm.py", 'r', encoding='utf-8') as f:
        content = f.read()

    return content


def return_cleaned_code_str(alg_name="genetic_algorithm"):
    if not os.path.exists('./code_temp/cleaned'):
        os.makedirs('./code_temp/cleaned')

    file_list = [
        name for root, dirs, files in os.walk(f"./code_base/{alg_name}")
        for name in files if name.endswith('.py')
    ]

    return make_clean_file(f"./code_base/{alg_name}", './code_temp/cleaned', file_list)
