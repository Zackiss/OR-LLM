from collections import defaultdict
from textwrap import dedent

import gradio as gr
import pandas as pd
import dotenv

from fun_search.RLDT_optimization import main_entry_opt
from fun_search.draw import visualize_packing
from fun_search.search import refine_algorithm


# Define default data for the tables
default_container_data = pd.DataFrame({
    "Length (L)": [12.0],
    "Width (W)": [2.33],
    "Height (H)": [2.39],
    "Capacity (kg)": [23000],
    "Amount": [1.0],
})

default_box_data = pd.DataFrame({
    "Length (l)": [1.09, 1.095, 0.915],
    "Width (w)": [0.5, 0.495, 0.37],
    "Height (h)": [0.885, 1.48, 0.615],
    "Weight (kg)": [68, 115, 36],
    "Amount": [66, 8, 80]
})


def process_data(container_df, box_df):
    # Extract container dimensions and other relevant data
    container_dimensions = container_df.iloc[0]
    container_length = container_dimensions["Length (L)"]
    container_width = container_dimensions["Width (W)"]
    container_height = container_dimensions["Height (H)"]
    container_capacity = container_dimensions["Capacity (kg)"]
    container_amount = container_dimensions["Amount"]

    # Extract and aggregate box data
    boxes = box_df[["Length (l)", "Width (w)", "Height (h)", "Amount", "Weight (kg)"]].to_dict(orient='records')
    dimension_counts = defaultdict(int)
    weight_counts = defaultdict(int)
    for box in boxes:
        dimensions = (box["Length (l)"], box["Width (w)"], box["Height (h)"])
        amount = int(box["Amount"])
        weight = box["Weight (kg)"]
        dimension_counts[dimensions] += amount
        weight_counts[dimensions] = weight

    box_list_str = ' + '.join(
        f"({dimensions}) * {count}"
        for dimensions, count in dimension_counts.items()
    )
    boxes_weight_str = ', '.join(
        f"{weight}" for _, weight in weight_counts.items()
    )

    code = dedent(f"""\
        L, W, H = {container_length}, {container_width}, {container_height}
        container_capacity, container_amount = {container_capacity}, {container_amount}
        boxes = [{box_list_str}]
        boxes_weight = [{boxes_weight_str}]
    """)

    return f"Data successfully parsed with {len(box_list_str)} boxes!", code.strip()


def load_css():
    with open('style.css', 'r') as file:
        css_content = file.read()
    return css_content


def process_visualize(processed_data, constraint):
    global all_figs
    refined_code = refine_algorithm(processed_data, constraint)

    exec(refined_code, globals())
    figs = globals().get('figs')
    placed_box_list = globals().get('placed_box_list')
    failed_box_list = globals().get('failed_box_list')
    logs = ""
    for i, (l, w, h, _, _, _) in enumerate(failed_box_list):
        logs += f"Box {i + len(placed_box_list)} ({l:.2f}x{w:.2f}x{h:.2f}) cannot be placed\n"
    for i, (l, w, h, x, y, z) in enumerate(placed_box_list):
        logs += f"Box {i} ({l:.2f}x{w:.2f}x{h:.2f}) placed at ({x:.2f},{y:.2f},{z:.2f})\n"

    all_figs = figs
    return refined_code, figs[0], logs


def process_refine_visualize():
    global all_figs

    figs_refined, fill_rate, reordered_boxes_code = main_entry_opt()
    all_figs = figs_refined
    return figs_refined[1], fill_rate, reordered_boxes_code


def gradio_interface():
    with gr.Blocks(theme="freddyaboulton/dracula_revamped", css=load_css()) as demo:
        gr.Markdown("## 3D Bin Packing Algorithm Refinement")

        with gr.Column():
            gr.Markdown("### Container and Box Data")
            container_table = gr.DataFrame(
                value=default_container_data,
                headers=["Length (L)", "Width (W)", "Height (H)", "Capacity (kg)", "Amount"],
                col_count=(5, "fixed"),
                row_count=(1, "fixed"),
                label="Container Specifications",
            )

            box_table = gr.DataFrame(
                value=default_box_data,
                headers=["Length (l)", "Width (w)", "Height (h)",  "Weight (kg)", "Amount"],
                col_count=(5, "fixed"),
                row_count=3,
                label="Box Specifications",
            )
            process_btn = gr.Button("Process Data")
        gr.Markdown("<br>")

        with gr.Column():
            gr.Markdown("### Select and Edit Constraint")
            constraint_selection = gr.Dropdown(
                label="Select Constraint",
                choices=list(constraints_dict.keys()),
                value="放置位置约束"
            )
            editable_constraint = gr.Textbox(
                label="Edit constraint",
                lines=5,
                value=constraints_dict["放置位置约束"],
                placeholder="Type your constraint here..."
            )
            constraint_selection.change(
                fn=lambda x: constraints_dict.get(x, ""),
                inputs=constraint_selection,
                outputs=editable_constraint
            )
            submit_btn = gr.Button("Generate Algorithm")
            refine_btn = gr.Button("Refine Packing with Reinforcement Learning")

        gr.Markdown("<br>")
        gr.Markdown("### Output & Result")
        with gr.Column():
            with gr.Row():
                output_text = gr.Textbox(
                    label="Processed Logs",
                    lines=12,
                    max_lines=22,
                    placeholder="Processed log will appear here...",
                    interactive=False
                )
                with gr.Column():
                    output_plot = gr.Plot(
                        label="Output Packing Result"
                    )

                    # Dropdown to select which plot to display
                    @gr.render(triggers=[output_text.change, demo.load])
                    def dynamic_dropdown():
                        plot_selection = gr.Dropdown(
                            label="Select Container to Plot",
                            choices=[(f"Container {i + 1}", i) for i in range(len(all_figs))] if len(all_figs) != 0 else [("Container 1", 0)],
                            value=0
                        )

                        plot_selection.change(
                            fn=update_plot,
                            inputs=plot_selection,
                            outputs=output_plot
                        )
            output_code_data = gr.Code(
                label="Data flow",
                language="python"
            )
            output_code = gr.Code(
                label="Generated Packing Algorithm",
                language="python"
            )

        def update_plot(selected_index):
            print(len(all_figs))
            print(selected_index)
            if len(all_figs) == 0:
                return None
            return all_figs[selected_index]

        process_btn.click(
            fn=process_data,
            inputs=[container_table, box_table],
            outputs=[output_text, output_code_data]
        )
        submit_btn.click(
            fn=process_visualize,
            inputs=[output_code_data, editable_constraint],
            outputs=[output_code, output_plot, output_text]
        )
        refine_btn.click(
            fn=process_refine_visualize,
            outputs=[output_plot, output_text, output_code_data]
        )

    demo.launch()


if __name__ == "__main__":
    all_figs = []
    dotenv.load_dotenv(dotenv_path="../.env")
    constraints_dict = {
        "放置位置约束": "如果我们放置了长、宽、高为（0.915, 0.37, 0.615）的箱子，那么它必须被放置在地面上，否则我们不放置它。\n",
        "重量约束": "箱子的重量等于它体积的 150 倍，单位是kg，例如 (3, 3, 3) 的箱子，其重量为 3 * 3 * 3 * 150 = 4050kg，我希望箱子的总"
                "重量不超过2300kg，但确保装入尽可能多的箱子，你需要打印出所有箱子重量的和。\n",
        "成套约束": "长、宽、高为：(1.09, 0.5, 0.885) 的箱子和长、宽、高为：(1.095, 0.495, 1.48) 的箱子，二者的装箱数量必须保持一致。"
                "你可以不从大到小装载箱子，而是尝试在开始时交替装载这两种箱子，一旦一种箱子装不下时，另一种箱子必须同时停止装载，以防止两种箱子不成套出现。"
                "你需要打印出三种箱子各自已装载的数量。\n",
        "多箱约束": "假设我们有两个集装箱，大小与第一个集装箱相等，我希望你能够将第一个箱子中无法放置的多余的箱子装入第二个集装箱。你需要为多个集装箱绘制多次结果，并保存在列表中。\n",
        "自定义约束": ""
    }
    gradio_interface()
