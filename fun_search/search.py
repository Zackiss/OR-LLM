from textwrap import dedent

from openai import OpenAI
from rag.rag import extract_python_code

import os


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


def refine_algorithm(data_flow, edited_constraint):
    # Read the input code
    with open("./algorithm.py", 'r', encoding='utf-8') as f:
        org_code = f.read()

    # Define the problem description
    problem_des = dedent("""\
        如下是三维装箱问题的描述:
            物流公司在流通过程中，需要将打包完毕的箱子装入到一个货车的车厢中，为了提高物流效率，需要将车厢尽量填满，车厢能够填满 85% 以上，认为装箱策略是优化的。
            设车厢为长方形，其长宽高分别为 L，W，H；共有 n 个箱子，箱子也为长方形，第 i 个箱子的长宽高为 li，wi，hi。
        如下基本假设：
            1. 长方形的车厢共有 8 个角，并设靠近驾驶室并位于下端的一个角的坐标为（0,0,0），车厢共 6 个面，其中 4 个面较长，2 个面接近正方形；
            2. 需要计算出每个箱子在车厢中的坐标，即每个箱子摆放后，其和车厢坐标为（0,0,0）的角在车厢中的坐标，并计算车厢的填充率；
            3. 静态装箱，即从 n 个箱子中选取 m 个箱子，并实现 m 个箱子在车厢中的摆放（无需考虑箱子从内向外，从下向上这种在车厢中的装箱顺序）；
        优化目标如下：
            车厢体积为 V，所有装入的箱子总体积为 S，填充率指的是 S / V ∗ 100%。问题目标是要确定一个可行的箱子放置方案使得在满足给定装载约束的情况下，填充率尽可能的大。
        以下基本约束:
            1. 箱子必须在车厢内（不能超过车厢容纳范围）;
            2. 任何两个箱子不能重叠;
            3. 所有箱子只能沿 x、y、z 轴防止，不能侧倾或倚靠放置;
            4. 箱底必须有支撑，禁止悬空，即箱子底部必须被全部覆盖（支撑物上表面积大于所放箱子的底面积）;\n
    """)

    # Prepare the prompts
    before_code_desc = "以下是实现三维装箱的代码：\n"
    prog = f"这是一个额外的约束，以下是这个约束条件的简述：{edited_constraint}，请只优化这一条额外的约束，将这个额外的约束条件重新表述成一句更详细具体的话，" \
           f"清晰化，明确化，然后给出你对添加这个约束到代码的具体分步思路，以格式 ”额外约束为：，实现思路为：“ 直接返回给我。\n"
    req_a = problem_des + before_code_desc + org_code + prog

    # Request and processing
    constraint = get_openai_response(req_a, "")
    history, codes = [], []
    response_text = ""
    while not codes:
        if not history:
            prog = f"你的任务是在此基础上，重构代码以考虑一个额外的约束，如下是完整的三维装箱代码：\n"
            before_data_flow = "\n如下是修改后的装箱算法需要使用的数据，以 python 格式给出，请你使用这些数据为参考修改代码：\n"
            cons_desc = f"\n你需要重构代码，以考虑如下这个额外的约束：\n{constraint}\n你需要添加该额外约束到算法中。请你返回修改后的完整代码，以满足约束要求。\n"
            sec_req = problem_des + prog + org_code + before_data_flow + data_flow + cons_desc
            print(sec_req)
            text = get_openai_response(sec_req, org_code, history=history)
        else:
            prog = "请直接继续输出："
            sec_req = prog
            text = get_openai_response(sec_req, "", history=history).replace("```python", "")
        response_text += text
        history.append({"role": "user", "content": sec_req})
        history.append({"role": "assistant", "content": text})
        codes = extract_python_code(response_text)
    refined_code = codes[0]
    # Save refined algorithm to a file
    with open("./refined_algorithm.py", 'w', encoding="utf-8") as file:
        file.write(refined_code)

    return refined_code
