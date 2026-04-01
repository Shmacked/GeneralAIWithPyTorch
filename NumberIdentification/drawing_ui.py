"""
Tkinter UI: draw a digit; debounced inference runs after you pause or release the mouse.
Predictions are shown in dedicated MLP and ResNet sections.
"""

import logging
import tkinter as tk
from tkinter import ttk

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw

import run_model as rm
from helpers.clean_up import clean_up

CANVAS_SIZE = 280
IMAGE_SIZE = 28
DEBOUNCE_MS = 300
STROKE_WIDTH_CANVAS = 14
STROKE_WIDTH_MODEL = 2

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

_mnist_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomInvert(p=1.0),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)


def _format_top3(logits: torch.Tensor) -> str:
    probs = torch.softmax(logits[0], dim=-1)
    values, indices = torch.topk(probs, k=3)
    return ", ".join(f"{int(i)}: {float(v):.3f}" for v, i in zip(values, indices))


class DrawingUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("MNIST drawing — predict")
        self.root.resizable(False, False)

        try:
            self.root.iconbitmap("icon.ico")
        except tk.TclError:
            pass

        self._predict_job = None
        self.last_x = None
        self.last_y = None
        self.pil_image = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 255)
        self.pil_draw = ImageDraw.Draw(self.pil_image)

        main = ttk.Frame(root, padding=8)
        main.pack(fill=tk.BOTH, expand=True)

        toolbar = ttk.Frame(main)
        toolbar.pack(fill=tk.X, pady=(0, 8))

        ttk.Button(toolbar, text="Clear", command=self.clear).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(toolbar, text="Predict now", command=self.predict_now).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Label(
            toolbar,
            text="Draw with the mouse; prediction runs after you pause (~300 ms) or release.",
        ).pack(side=tk.LEFT)

        canvas_frame = ttk.Frame(main, relief=tk.SUNKEN, borderwidth=1)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(
            canvas_frame,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="white",
            highlightthickness=0,
        )
        self.canvas.pack()

        self.canvas.bind("<Button-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_paint)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        predictions_frame = ttk.Frame(main)
        predictions_frame.pack(fill=tk.X, pady=(8, 0))

        mlp_frame = ttk.LabelFrame(predictions_frame, text="MLP", padding=8)
        mlp_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        self.mlp_guess_var = tk.StringVar(value="No prediction yet")
        ttk.Label(mlp_frame, textvariable=self.mlp_guess_var).pack(anchor=tk.W)

        resnet_frame = ttk.LabelFrame(predictions_frame, text="ResNet", padding=8)
        resnet_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))
        self.resnet_guess_var = tk.StringVar(value="No prediction yet")
        ttk.Label(resnet_frame, textvariable=self.resnet_guess_var).pack(anchor=tk.W)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _canvas_to_model(self, x: float, y: float) -> tuple[int, int]:
        mx = int(x * IMAGE_SIZE / CANVAS_SIZE)
        my = int(y * IMAGE_SIZE / CANVAS_SIZE)
        mx = max(0, min(IMAGE_SIZE - 1, mx))
        my = max(0, min(IMAGE_SIZE - 1, my))
        return mx, my

    def on_press(self, event: tk.Event) -> None:
        self.last_x, self.last_y = event.x, event.y

    def on_paint(self, event: tk.Event) -> None:
        if self.last_x is None:
            self.last_x, self.last_y = event.x, event.y
            return

        self.canvas.create_line(
            self.last_x,
            self.last_y,
            event.x,
            event.y,
            fill="black",
            width=STROKE_WIDTH_CANVAS,
            capstyle=tk.ROUND,
            smooth=True,
        )

        x0, y0 = self._canvas_to_model(self.last_x, self.last_y)
        x1, y1 = self._canvas_to_model(event.x, event.y)
        self.pil_draw.line((x0, y0, x1, y1), fill=0, width=STROKE_WIDTH_MODEL)

        self.last_x, self.last_y = event.x, event.y
        self._schedule_predict()

    def on_release(self, event: tk.Event) -> None:
        self.last_x, self.last_y = None, None
        self._cancel_predict_job()
        self.root.after_idle(self._run_predict)

    def clear(self) -> None:
        self._cancel_predict_job()
        self.canvas.delete("all")
        self.pil_image = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 255)
        self.pil_draw = ImageDraw.Draw(self.pil_image)
        self.last_x, self.last_y = None, None
        self.mlp_guess_var.set("No prediction yet")
        self.resnet_guess_var.set("No prediction yet")

    def _cancel_predict_job(self) -> None:
        if self._predict_job is not None:
            self.root.after_cancel(self._predict_job)
            self._predict_job = None

    def _schedule_predict(self) -> None:
        self._cancel_predict_job()
        self._predict_job = self.root.after(DEBOUNCE_MS, self._run_predict)

    def predict_now(self) -> None:
        self._cancel_predict_job()
        self._run_predict()

    def _pil_to_tensor(self) -> torch.Tensor:
        return _mnist_transform(self.pil_image)

    def _run_predict(self) -> None:
        self._predict_job = None
        tensor = self._pil_to_tensor()
        try:
            out_mlp, out_resnet = rm.run_model(tensor)
        except Exception as e:
            logger.exception("Prediction failed: %s", e)
            self.mlp_guess_var.set("Prediction failed")
            self.resnet_guess_var.set("Prediction failed")
            return

        self.mlp_guess_var.set(_format_top3(out_mlp))
        self.resnet_guess_var.set(_format_top3(out_resnet))

    def on_close(self) -> None:
        self._cancel_predict_job()
        try:
            clean_up(
                [
                    rm.simple_mlp,
                    rm.simple_resnet,
                    rm.state_dict1,
                    rm.state_dict2,
                    rm.test_loader,
                    rm.test_dataset,
                    rm.test_transform,
                ],
                rm.__dict__,
            )
        except Exception as e:
            logger.warning("Cleanup encountered an issue: %s", e)
        finally:
            self.root.destroy()


def main() -> None:
    root = tk.Tk()
    DrawingUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
