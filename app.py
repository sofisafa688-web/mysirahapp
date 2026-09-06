import json
import os
from huggingface_hub import hf_hub_download
import numpy as np
import streamlit as st
import tensorflow as tf
from tokenizers import Tokenizer

# إعدادات الصفحة
st.set_page_config(
    page_title="مساعد السيرة النبوية", page_icon="📖", layout="centered"
)

# تخصيص التصميم ليشبه محادثات واتساب (فقاعات وترتيب أسطر متعددة)
st.markdown(
    """
    <style>
    .stChatMessage {
        padding: 10px 15px;
        border-radius: 15px;
        margin-bottom: 10px;
        max-width: 85%;
        word-wrap: break-word;
        white-space: pre-wrap;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.title("📖 مساعد السيرة النبوية الذكي")
st.write("اطرح سؤالك وسيجيبك النموذج في محادثة مستمرة ومتعددة الأسطر.")

st.title("اختبار التطبيق والرموز تلقائياً")

# ضع هنا اسم مستودعك على هاقينج فيس
REPO_ID = "sofisofi88/sirah-assistant-model"


@st.cache_resource
def load_model_and_tokenizer():
  with st.spinner("جاري تحميل المودل والتوكنايزر من Hugging Face..."):
    model_path = hf_hub_download(
        repo_id=REPO_ID, filename="assistant_code_model.keras"
    )
    tokenizer_path = hf_hub_download(
        repo_id=REPO_ID, filename="assistant_code_bpe_tokenizer.json"
    )

    loaded_model = tf.keras.models.load_model(model_path)
    loaded_tok = Tokenizer.from_file(tokenizer_path)
  return loaded_model, loaded_tok


model, tok = load_model_and_tokenizer()




    
    st.subheader("🧪 نتائج اختبار الترميز")
    
    # 3. تطبيق كود الاختبار مباشرة وعرضه بعنوان
    for text in ["عمر بن الخطاب", "رسول الله"]:
        enc = tok.encode(text)
        
        # عرض النتيجة على شاشة موقع ستريم لايت مباشرة
        st.markdown(f"### النص: {text}")
        st.write(f"**Tokens:** `{enc.tokens}`")
        st.write(f"**Decode:** `{tok.decode(enc.ids)}`")
        st.markdown("---")
        
except Exception as e:
    st.error(f"❌ حدث خطأ أثناء التحميل أو الاختبار: {e}")

# استرجاع الثوابت ديناميكياً
MAX_LEN = model.input_shape[0][1] + 1
try:
  PAD, UNK, USER_ID, ASST_ID, EOS = (
      tok.token_to_id(s)
      for s in ["<PAD>", "<UNK>", "<|user|>", "<|assistant|>", "<EOS>"]
  )
except:
  PAD, UNK, USER_ID, ASST_ID, EOS = 0, 1, 3, 4, 2


def generate_reply(prompt_ids, max_new_tokens=150, temperature=0.8, top_p=0.9):
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
      if 0 < top_p < 1:
        order = np.argsort(p)[::-1]
        keep = order[
            max(1, int(np.searchsorted(np.cumsum(p[order]), top_p) + 1))
        ]
        mask = np.zeros_like(p)
        mask[keep] = p[keep]
        p = mask / mask.sum()
      nid = int(np.random.choice(len(p), p=p))
    else:
      nid = int(lg.argmax())

    if nid in (PAD, EOS, USER_ID, ASST_ID):
      break
    ids.append(nid)
    out.append(nid)
  return tok.decode(out).strip()


def chat(message, history=None):
  prompt = ""
  for u, a in history or []:
    prompt += f"<|user|> {u} <|assistant|> {a} <|EOS|> "
  prompt += f"<|user|> {message} <|assistant|>"
  return generate_reply(tok.encode(prompt).ids)


# تخزين سجل المحادثات
if "messages" not in st.session_state:
  st.session_state.messages = []

# عرض الرسائل بتنسيق أسطر متعددة
for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])

# صندوق إدخال الرسائل
if user_input := st.chat_input("اكتب رسالتك هنا..."):
  st.session_state.messages.append({"role": "user", "content": user_input})
  with st.chat_message("user"):
    st.markdown(user_input)

  with st.chat_message("assistant"):
    with st.spinner("جاري الكتابة..."):
      history = [
          (
              st.session_state.messages[i]["content"],
              st.session_state.messages[i + 1]["content"],
          )
          for i in range(0, len(st.session_state.messages) - 1, 2)
      ]
      reply = chat(user_input, history=history)
      st.markdown(reply)
  st.session_state.messages.append({"role": "assistant", "content": reply})
