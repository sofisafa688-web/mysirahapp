import json
import os
from flask import Flask, jsonify, render_template, request
from huggingface_hub import hf_hub_download
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# ضع هنا اسم مستودعك على هاقينج فيس
REPO_ID = "your-username/serah-assistant-model"

print("جاري تحميل الملفات من Hugging Face...")

# تحميل ملف المودل والتوكنايزر من Hugging Face
model_path = hf_hub_download(
    repo_id=REPO_ID, filename="assistant_code_best.keras"
)
tokenizer_path = hf_hub_download(
    repo_id=REPO_ID, filename="assistant_code_bpe_tokenizer"
)

# تحميل المودل
model = tf.keras.models.load_model(model_path)


# استبدل هذا الفئة حسب مكتبة Tokenizer التي تستخدمها (مثل tokenizers من Hugging Face أو الكلاس الخاص بك)
# إذا كان ملف assistant_code_bpe_tokenizer هو ملف Tokenizer قياسي، يمكنك تحميله هكذا:
from tokenizers import Tokenizer

tok = Tokenizer.from_file(tokenizer_path)

# --- استرجاع الثوابت كما في الصورة ---
MAX_LEN = model.input_shape[0][1] + 1
PAD, UNK, USER_ID, ASST_ID, EOS = (
    tok.token_to_id(s)
    for s in ["<PAD>", "<UNK>", "<|user|>", "<|assistant|>", "<EOS>"]
)


def generate_reply(prompt_ids, max_new_tokens=120, temperature=0.8, top_p=0.9):
  ids = list(prompt_ids)[-(MAX_LEN - 1) :]
  out = []
  for _ in range(max_new_tokens):
    if len(ids) >= MAX_LEN - 1:
      ids = ids[-(MAX_LEN - 1) :]
    x = np.zeros((1, MAX_LEN - 1), dtype=np.int32)
    x[0, : len(ids)] = ids
    doc_ids = np.zeros((1, MAX_LEN - 1), dtype=np.int32)
    doc_ids[0, : len(ids)] = 1
    lg = (
        model([x, doc_ids], training=False)
        .numpy()[0, len(ids) - 1]
        .astype(np.float64)
    )

    if temperature and temperature > 0:
      lg = lg / temperature
      p = np.exp(lg - lg.max())
      p /= p.sum()
      if 0 < top_p < 1:  # nucleus sampling
        order = np.argsort(p)[::-1]
        keep = order[
            max(1, int(np.searchsorted(np.cumsum(p[order]), top_p) + 1))
        ]
        mask = np.zeros_like(p)
        mask[keep] = p[keep]
        p = mask / mask.sum()
      nid = int(np.random.choice(len(p), p=p))
    else:
      nid = int(lg.argmax())  # greedy

    if nid in (PAD, EOS, USER_ID, ASST_ID):
      break
    ids.append(nid)
    out.append(nid)
  return tok.decode(out).strip()


def chat(message, history=None, **kw):
  prompt = ""
  for u, a in history or []:
    prompt += f"<|user|> {u} <|assistant|> {a} <|EOS|> "
  prompt += f"<|user|> {message} <|assistant|>"
  return generate_reply(tok.encode(prompt).ids, **kw)


@app.route("/")
def home():
  return "مرحباً، مساعد السيرة النبوية يعمل بنجاح!"


@app.route("/predict", methods=["POST"])
def predict():
  data = request.json
  message = data.get("message", "")
  history = data.get("history", None)

  # توليد الرد
  reply = chat(message, history=history, temperature=0.8, top_p=0.9)

  return jsonify({"response": reply})


if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5000)
