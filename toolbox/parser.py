import ast

import astor


def get_source_segment(source, node):
    """
    提取与给定 AST 节点对应的源代码段。
    """
    lines = source.splitlines()  # 将源代码按行分割
    start_line = node.lineno - 1  # 节点的起始行号
    # 尝试获取节点的结束行号，如果没有，则默认为起始行号
    end_line = getattr(node, 'end_lineno', node.lineno)
    # 返回从起始行到结束行的源代码段
    return "\n".join(lines[start_line:end_line])


def add_parents(tree):
    """
    为每个 AST 节点添加 `parent` 属性，以便于父子节点关系的遍历。
    """
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node


def parse_python_file(file_path):
    """
    解析 Python 文件，返回包含类定义和独立函数定义的字典。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        source = f.read()  # 读取源代码

    tree = ast.parse(source)  # 解析源代码成 AST
    add_parents(tree)  # 为每个节点添加父节点信息
    result = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # 如果节点是类定义，提取类名和源代码段
            name = node.name
            code_segment = get_source_segment(source, node)
            result[name] = code_segment
        elif isinstance(node, ast.FunctionDef):
            # 如果节点是函数定义，且不在类内，则提取函数名和源代码段
            if not isinstance(node.parent, ast.ClassDef):
                name = node.name
                code_segment = get_source_segment(source, node)
                result[name] = code_segment

    return result


def parse_python_code(python_code):
    """
    解析 Python 文件，返回包含类定义和独立函数定义的字典。
    """

    source = python_code
    tree = ast.parse(source)  # 解析源代码成 AST
    add_parents(tree)  # 为每个节点添加父节点信息
    result = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # 如果节点是类定义，提取类名和源代码段
            name = node.name
            code_segment = get_source_segment(source, node)
            result[name] = code_segment
        elif isinstance(node, ast.FunctionDef):
            # 如果节点是函数定义，且不在类内，则提取函数名和源代码段
            if not isinstance(node.parent, ast.ClassDef):
                name = node.name
                code_segment = get_source_segment(source, node)
                result[name] = code_segment

    return result


def extract_function_and_class_names(source_code):
    """
    从给定的源代码中提取所有的函数名和类名。

    :param source_code: Python 源代码字符串
    :return: 一个包含函数名和类名的字典
    """
    tree = ast.parse(source_code)
    function_names = []
    class_names = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            function_names.append(node.name)
        elif isinstance(node, ast.ClassDef):
            class_names.append(node.name)

    # 确保不返回类内函数名
    return class_names, function_names


class NodeTransformer(ast.NodeTransformer):
    def __init__(self, node_type, name, new_code):
        self.node_type = node_type
        self.name = name
        self.new_code = new_code
        self.node_found = False
        self.main_block_found = False
        super().__init__()

    def visit_FunctionDef(self, node):
        if self.node_type == 'function' and node.name == self.name:
            self.node_found = True
            new_node = ast.parse(self.new_code).body[0]
            return ast.copy_location(new_node, node)
        return self.generic_visit(node)

    def visit_ClassDef(self, node):
        if self.node_type == 'class' and node.name == self.name:
            self.node_found = True
            new_node = ast.parse(self.new_code).body[0]
            return ast.copy_location(new_node, node)
        return self.generic_visit(node)

    def visit_If(self, node):
        # Check if the block is an 'if __name__ == "__main__":'
        if isinstance(node.test, ast.Compare) and \
                isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__" and \
                isinstance(node.test.ops[0], ast.Eq) and \
                isinstance(node.test.comparators[0], ast.Str) and node.test.comparators[0].s == "__main__":
            self.main_block_found = True
        return self.generic_visit(node)

    def add_node_if_not_found(self, tree):
        if not self.node_found:
            new_node = ast.parse(self.new_code).body[0]
            if self.main_block_found:
                # Insert before the 'if __name__ == "__main__":' block
                index = next((i for i, node in enumerate(tree.body) if isinstance(node, ast.If) and
                              isinstance(node.test, ast.Compare) and
                              isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__"), len(tree.body))
                tree.body.insert(index, new_node)
            else:
                # Append to the end
                tree.body.append(new_node)


def replace_node_in_code(source_code, node_type, name, new_code):
    tree = ast.parse(source_code)
    transformer = NodeTransformer(node_type, name, new_code)
    new_tree = transformer.visit(tree)
    transformer.add_node_if_not_found(new_tree)
    new_source_code = astor.to_source(new_tree)

    return new_source_code


def return_parsed_code_dict(algorithm_name):
    file_path = f'./code_base/{algorithm_name}/algorithm.py'
    parsed_content = parse_python_file(file_path)
    return parsed_content
