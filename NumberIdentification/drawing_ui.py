"""
Tkinter UI: draw a digit; debounced inference runs after you pause or release the mouse.
Predictions are logged via the standard logging module and shown in the window.
"""

import logging
import tkinter as tk
from tkinter import ttk

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw

import run_model as rm

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

        log_frame = ttk.LabelFrame(main, text="Log", padding=4)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        self.log_text = tk.Text(log_frame, height=6, state=tk.DISABLED, wrap=tk.WORD)
        scroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self._log_handler = _TextHandler(self.log_text)
        self._log_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(self._log_handler)
        logger.setLevel(logging.INFO)

        logger.info("Ready. Draw a digit (0–9).")

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
        logger.info("Canvas cleared.")

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
            return

        logger.info(
            "MLP top-3: %s | ResNet top-3: %s",
            _format_top3(out_mlp),
            _format_top3(out_resnet),
        )


class _TextHandler(logging.Handler):
    def __init__(self, widget: tk.Text):
        super().__init__()
        self.widget = widget

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record) + "\n"

        def append() -> None:
            self.widget.configure(state=tk.NORMAL)
            self.widget.insert(tk.END, msg)
            self.widget.see(tk.END)
            self.widget.configure(state=tk.DISABLED)

        self.widget.after(0, append)


def main() -> None:
    root = tk.Tk()
    DrawingUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
