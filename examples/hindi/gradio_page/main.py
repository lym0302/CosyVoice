# main.py
# python main.py --pages page1 page2
import uvicorn
from fastapi import FastAPI
import gradio as gr
import importlib
import argparse

# ------------------ 命令行参数 ------------------
parser = argparse.ArgumentParser()
parser.add_argument("--pages", nargs="+", help="要加载的页面，例如：page1 page2")
args = parser.parse_args()

app = FastAPI()

# ------------------ 动态挂载页面 ------------------
for page in args.pages:
    module = importlib.import_module(f"pages.{page}")
    if hasattr(module, "create_demo"):
        demo = module.create_demo()
        app = gr.mount_gradio_app(app, demo, path=f"/{page}")
        print(f"✅ 挂载 {page} 到 /{page}")

# ------------------ 启动服务器 ------------------
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7680)

