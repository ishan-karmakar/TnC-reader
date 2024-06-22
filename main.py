import sys
from PyQt6.QtCore import QThread
from PyQt6.QtWidgets import QApplication, QMainWindow, QPlainTextEdit, QPushButton
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
WIN_SIZE = (800, 1200)
INP_TEXT_HEIGHT = int(WIN_SIZE[1] / 2 - 20)
BUTTON_HEIGHT = 40
BUTTON_WIDTH = int(WIN_SIZE[0] / 2 - 20)
OUT_TEXT_HEIGHT = WIN_SIZE[1] - INP_TEXT_HEIGHT - BUTTON_HEIGHT - 40
MODEL = "facebook/bart-large-cnn"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class QSummarizer(QThread):
    def __init__(self, tokenizer, summarizer, text):
        super().__init__()
        self.tokenizer = tokenizer
        self.summarizer = summarizer
        self.in_text = text
    
    def run(self):
        inputs = self.tokenizer(self.in_text, return_tensors="pt").to(DEVICE)
        summary_ids = self.summarizer.generate(inputs["input_ids"], min_length=200, max_length=1000).to(DEVICE)
        self.out_text = self.tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0]

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.summarizer = AutoModelForSeq2SeqLM.from_pretrained(MODEL).to(DEVICE)

        self.setFixedSize(*WIN_SIZE)
        self.setWindowTitle("T&C Reader")

        self.cur_size = (self.size().width(), self.size().height())

        self.inp_text = QPlainTextEdit(self)
        self.inp_text.move(10, 10)
        self.inp_text.setFixedSize(WIN_SIZE[0] - 20, INP_TEXT_HEIGHT)
        self.inp_text.setPlaceholderText("Terms and Conditions here")

        self.sButton = QPushButton("Summarize", self)
        self.sButton.move(10, 10 + INP_TEXT_HEIGHT + 10)
        self.sButton.setFixedSize(BUTTON_WIDTH, BUTTON_HEIGHT)
        self.sButton.clicked.connect(self.initSummarizeText)

        self.cButton = QPushButton("Clear", self)
        self.cButton.move(int(WIN_SIZE[0] / 2 + 10), 10 + INP_TEXT_HEIGHT + 10)
        self.cButton.setFixedSize(BUTTON_WIDTH, BUTTON_HEIGHT)
        self.cButton.clicked.connect(self.clearText)

        self.out_text = QPlainTextEdit(self)
        self.out_text.setReadOnly(True)
        self.out_text.move(10, 10 + INP_TEXT_HEIGHT + 10 + BUTTON_HEIGHT + 10)
        self.out_text.setFixedSize(WIN_SIZE[0] - 20, OUT_TEXT_HEIGHT)

    def initSummarizeText(self):
        self.sButton.setEnabled(False)
        inp_text = self.inp_text.toPlainText()
        self.summarizer_thread = QSummarizer(self.tokenizer, self.summarizer, inp_text)
        self.summarizer_thread.finished.connect(self.finishSummarizeText)
        self.summarizer_thread.start()
    
    def finishSummarizeText(self):
        self.out_text.setPlainText(self.summarizer_thread.out_text)
        del self.summarizer_thread
        self.sButton.setEnabled(True)

    def clearText(self):
        self.inp_text.clear()
        self.out_text.clear()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec())
