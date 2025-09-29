import gradio as gr

def dummy_function(*args):
    return "尚未实现", None  # 返回一个提示



with gr.Blocks() as app:
    with gr.Row():
        with gr.Column():
            audio_player = gr.Audio(label="Audio", type="filepath", interactive=True, elem_classes=["audio-container"])

    with gr.Row():
        with gr.Column():
            text = gr.TextArea(
                label="Text",
                placeholder="recognition text",
                lines=2  # 控制高度为 2 行
            )
    with gr.Row():
        with gr.Column():
            pron_acc = gr.Slider(1, 5, value=3, step=0.5, label="Pronunciation Accuracy")
            real = gr.Slider(1, 5, value=3, step=0.5, label="Real People")
        with gr.Column():    
            trans = gr.Button("下一条", variant="primary")
            slicer = gr.Button("上一条", variant="primary")
            slicer2 = gr.Button("导出csv", variant="primary")
    
    with gr.Row():            
        with gr.Column():
            text_output = gr.Textbox(label="导出csv状态")
    
    with gr.Row():
        progress_text = gr.Text(label="Progress", interactive=False)
            

        #     speaker = gr.Dropdown(choices=speakers, value=speakers[0], label="Speaker")

        #     prompt_mode = gr.Radio(
        #         ["Text prompt", "Audio prompt"],
        #         label="Prompt Mode",
        #         value="Text prompt",
        #     )

        #     text_prompt = gr.Textbox(
        #         label="Text prompt", placeholder="例如：Happy", value="", visible=True
        #     )
        #     audio_prompt = gr.Audio(label="Audio prompt", type="filepath", visible=False)

        #     pron_acc = gr.Slider(1, 5, value=3, step=0.5, label="Pronunciation Accuracy")
        #     real = gr.Slider(1, 5, value=3, step=0.5, label="Real People")
        #     # noise_scale_w = gr.Slider(0.1, 2, value=0.9, step=0.1, label="Noise_W")
        #     # length_scale = gr.Slider(0.1, 2, value=1.0, step=0.1, label="Length")
        #     # language = gr.Dropdown(choices=languages, value=languages[0], label="Language")

        #     btn = gr.Button("生成音频！", variant="primary")

        # with gr.Column():
        #     with gr.Accordion("融合文本语义", open=False):
        #         style_text = gr.Textbox(label="辅助文本")
        #         style_weight = gr.Slider(0, 1, value=0.7, step=0.1, label="Weight")
        #     with gr.Row():
        #         interval_between_sent = gr.Slider(
        #             0, 5, value=0.2, step=0.1, label="句间停顿(秒)"
        #         )
        #         interval_between_para = gr.Slider(
        #             0, 10, value=1, step=0.1, label="段间停顿(秒)"
        #         )
        #         opt_cut_by_sent = gr.Checkbox(label="按句切分")
        #     slicer2 = gr.Button("切分生成", variant="primary")
        #     text_output = gr.Textbox(label="状态信息")
        #     audio_output = gr.Audio(label="输出音频")

    # 连接所有按钮到 dummy 函数
    # btn.click(dummy_function, inputs=[], outputs=[text_output, audio_output])
    trans.click(dummy_function, inputs=[], outputs=[text])
    slicer.click(dummy_function, inputs=[], outputs=[text])
    slicer2.click(dummy_function, inputs=[], outputs=[text])
    # formatter.click(lambda x, y: ("mix", x), inputs=[text, speaker], outputs=[language, text])

    # prompt_mode.change(
    #     lambda x: (
    #         {"visible": x == "Text prompt", "__type__": "update"},
    #         {"visible": x == "Audio prompt", "__type__": "update"},
    #     ),
    #     inputs=[prompt_mode],
    #     outputs=[text_prompt, audio_prompt],
    # )


app.launch(
    server_port=7680,
    server_name="127.0.0.1",
    share=False
)
