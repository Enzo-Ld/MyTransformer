import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QWidget, QTextEdit, QPushButton, QLabel, QComboBox, QMessageBox
)
from PyQt5.QtCore import Qt
import torch
from transformers import AutoTokenizer
from model import Transformer


class TranslationApp(QMainWindow):
    def __init__(self, model_path="./model.pt", tokenizer_path="./tokenizer"):
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = 128   # 生成最大长度
        self.init_ui()
        self.load_resources(model_path, tokenizer_path)

    def init_ui(self):
        self.setWindowTitle("NLP Transformer 翻译器")
        self.setGeometry(200, 200, 800, 600)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        main_widget.setLayout(layout)

        # 语言选择
        lang_layout = QHBoxLayout()
        self.src_lang = QComboBox()
        self.src_lang.addItems(["英语"])
        self.tgt_lang = QComboBox()
        self.tgt_lang.addItems(["中文"])
        lang_layout.addWidget(QLabel("源语言:"))
        lang_layout.addWidget(self.src_lang)
        lang_layout.addWidget(QLabel("目标语言:"))
        lang_layout.addWidget(self.tgt_lang)
        layout.addLayout(lang_layout)

        # 输入区
        layout.addWidget(QLabel("输入文本:"))
        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText("请输入要翻译的文本...")
        layout.addWidget(self.input_text)

        # 输出区
        layout.addWidget(QLabel("翻译结果:"))
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        layout.addWidget(self.output_text)

        # 按钮区
        btn_layout = QHBoxLayout()
        self.translate_btn = QPushButton("翻译")
        self.translate_btn.clicked.connect(self.translate_text)
        self.clear_btn = QPushButton("清空")
        self.clear_btn.clicked.connect(self.clear_text)
        btn_layout.addWidget(self.translate_btn)
        btn_layout.addWidget(self.clear_btn)
        layout.addLayout(btn_layout)

    def load_resources(self, model_path, tokenizer_path):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.tokenizer.add_special_tokens({"bos_token": "<s>"})
            vocab_size = self.tokenizer.vocab_size + len(self.tokenizer.special_tokens_map)

            self.model = Transformer(
                enc_vocab_size=vocab_size,
                dec_vocab_size=vocab_size,
                pad_idx=self.tokenizer.pad_token_id,
                n_layers=3,
                heads=4,
                d_model=128,
                d_ff=512,
                dropout=0.1,
                max_seq_len=512
            ).to(self.device)

            if torch.cuda.is_available():
                self.model.load_state_dict(torch.load(model_path))
            else:
                self.model.load_state_dict(torch.load(model_path, map_location="cpu"))

            self.model.eval()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载模型或分词器失败: {e}")

    def greedy_decode(self, input_ids):
        """逐步生成翻译结果"""
        batch_size = input_ids.size(0)
        de_in = torch.ones(batch_size, self.max_len, dtype=torch.long).to(self.device) * self.tokenizer.pad_token_id
        de_in[:, 0] = self.tokenizer.bos_token_id

        with torch.no_grad():
            for i in range(1, self.max_len):
                pred = self.model(input_ids, de_in[:, :i])   # 只输入已有部分
                next_token = pred[:, -1, :].argmax(-1)
                de_in[:, i] = next_token
                if (next_token == self.tokenizer.eos_token_id).all():
                    break
        return de_in

    def translate_text(self):
        text = self.input_text.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "提示", "请输入要翻译的文本")
            return

        try:
            # 编码输入
            input_ids = self.tokenizer(
                text,
                padding="max_length",
                max_length=128,
                truncation=True,
                return_tensors="pt"
            )["input_ids"].to(self.device)

            # 解码
            output_ids = self.greedy_decode(input_ids)
            result = self.tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)

            self.output_text.setPlainText(result)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"翻译失败: {e}")

    def clear_text(self):
        self.input_text.clear()
        self.output_text.clear()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TranslationApp()
    window.show()
    sys.exit(app.exec_())
