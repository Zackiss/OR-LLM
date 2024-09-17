import os
import re

import dotenv
from openai import OpenAI

from toolbox.clean_comment import return_cleaned_code_str, read_original_code
from toolbox.parser import return_parsed_code_dict, replace_node_in_code


def get_openai_response(prompt, document, model='gpt-4o', history=[]):
    """
    使用 OpenAI API 获取模型的回复。

    :param history: 对话历史
    :param prompt: 发送给模型的输入提示
    :param document: 使用的代码文本
    :param model: 使用的模型名称
    :return: 模型生成的回复文本
    """
    full_query = prompt + "\n" + document
    client = OpenAI(
        base_url="https://api.gptsapi.net/v1",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": f"{full_query}"
        }] + history
    )
    return response.choices[0].message.content


def extract_python_code(text):
    """
    从给定的文本中提取所有的 Python 代码片段。

    :param text: 包含代码片段的文本
    :return: Python 代码片段列表
    """
    pattern = r"```python\n(.*?)\n```"
    matches = re.findall(pattern, text, re.DOTALL)
    # 返回一个去除多余空白的代码片段列表
    return [match.strip() for match in matches]


if __name__ == "__main__":
    dotenv.load_dotenv(dotenv_path="../.env")

    algorithm_name = ["depth_first_search", "genetic_algorithm", "greedy", "simulated_annealing"][2]
    org_code = read_original_code(algorithm_name)
    clean_code = return_cleaned_code_str(algorithm_name)
    parsed_code = return_parsed_code_dict(algorithm_name)

    problem_des = """如下是三维装箱问题的描述:
    物流公司在流通过程中，需要将打包完毕的箱子装入到一个货车的车厢中，为了提高物流效率，需要将车厢尽量填满，车厢能够填满 85% 以上，认为装箱策略是优化的。
    设车厢为长方形，其长宽高分别为 L，W，H；共有 n 个箱子，箱子也为长方形，第 i 个箱子的长宽高为 li，wi，hi。
    对于三维装箱问题，我们做如下基本假设：
    1. 长方形的车厢共有 8 个角，并设靠近驾驶室并位于下端的一个角的坐标为（0,0,0），车厢共 6 个面，其中 4 个面较长，2 个面接近正方形；
    2. 需要计算出每个箱子在车厢中的坐标，即每个箱子摆放后，其和车厢坐标为（0,0,0）的角相对应的角在车厢中的坐标，并计算车厢的填充率；
    3. 静态装箱，即从 n 个箱子中选取 m 个箱子，并实现 m 个箱子在车厢中的摆放（无需考虑箱子从内向外，从下向上这种在车厢中的装箱顺序）；
    三维装箱问题的优化目标如下：
    车厢体积为 V，所有装入的箱子总体积为 S，填充率指的是 S / V ∗ 100%。问题目标是要确定一个可行的箱子放置方案使得在满足给定装载约束的情况下，填充率尽可能的大。
    三维装箱问题有以下基本约束:
    1. 箱子必须在车厢内（不能超过车厢容纳范围）;
    2. 任何两个箱子不能重叠;
    3. 所有箱子只能沿 x、y、z 轴防止，不能侧倾或倚靠放置;
    4. 箱底必须有支撑，禁止悬空，即箱子底部必须被全部覆盖（支撑物上表面积大于所放箱子的底面积）;\n"""

    # 第一部分：提出解决方案，优化约束表达
    before_code_desc = "以下实现三维装箱的基本算法代码：\n"
    # constraint = "长、宽、高为：108, 76, 30 的箱子，必须放置在长、宽、高为：92, 81, 55 的箱子上方。\n"
    # constraint = "设第一类箱子的重量为 1kg，第二类箱子的重量为 2kg，第三类箱子的重量为 3kg，我希望箱子的总重量不超过100kg，但确保装入尽可能多的箱子，你需要给出所有箱子重量的和。\n"
    constraint = "长、宽、高为：108, 76, 30 的箱子和长、宽、高为：110, 43, 25 的箱子，二者装箱数量必须保持一致。" \
                 "即便两种箱子数量溢出，也依然要确保装入的数量相同，而不可以只装入一种。就像桌子和椅子在运送的时候，需要成套配对在一个集装箱里运送。\n"
    prog = f"这是一个额外的约束，以下是这个约束条件的简述：{constraint}，请只优化这一条额外的约束，将这个额外的约束条件重新表述成一句更详细具体的话，" \
           f"清晰化，明确化，然后给出你对添加这个约束到代码的具体分步思路，以格式 ”额外约束为：，实现思路为：“ 直接返回给我。\n"
    req_a = problem_des + before_code_desc + clean_code + prog
    constraint = get_openai_response(req_a, "")

    # 第二部分：返回完整代码
    new_code = clean_code
    history = []
    codes = []
    response_text = ""
    while not codes:
        if not history:
            prog = f"你的任务是在此基础上，重构代码以考虑这一个额外的约束：\n{constraint}\n你需要添加该额外约束到算法中。如下是完整的三维装箱代码，请你返回修改后的完整代码，以满足约束要求：\n"
            hint = f"另外，请重点关注 find_next_block，只添加简单的条件判断并不能解决问题。\n"
            sec_req = problem_des + prog + hint + org_code
            print(sec_req)
            response_text += get_openai_response(sec_req, new_code, history=history)
            history.append({"role": "user", "content": sec_req})
            history.append({"role": "assistant", "content": response_text})
        else:
            prog = "请直接继续输出："
            sec_req = prog
            text = get_openai_response(sec_req, "", history=history).replace("```python", "")
            response_text += text
            history.append({"role": "user", "content": sec_req})
            history.append({"role": "assistant", "content": text})
        codes = extract_python_code(response_text)
        print(response_text)

    with open("code_temp/refined/refined_algorithm.py", 'w', encoding="utf-8") as file:
        file.write(codes[0])
