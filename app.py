import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import requests
import io
from torchvision import transforms
import pickle

# ============== إصلاح دالة توليد الوصف ==============
def generate_caption(model, image_tensor, vocab, device, max_length=25):
    """دالة توليد الوصف المصححة"""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        features = model.encoder(image_tensor)
        
        # ✅ تصحيح: استخدم '<SOS>' 
        caption = [vocab.stoi["<SOS>"]]
        
        # ✅ تصحيح: hidden state من decoder
        h = model.decoder.fc_h(features).unsqueeze(0)
        c = model.decoder.fc_c(features).unsqueeze(0)
        
        for _ in range(max_length):
            last_word = torch.tensor([caption[-1]]).to(device)
            # ✅ تصحيح: استخدم decoder.embed
            embedding = model.decoder.embed(last_word).unsqueeze(1)
            
            lstm_out, (h, c) = model.decoder.lstm(embedding, (h, c))
            output = model.decoder.linear(lstm_out.squeeze(1))
            
            predicted = output.argmax(1).item()
            caption.append(predicted)
            
            # ✅ تصحيح: توقف عند <EOS>
            if predicted == vocab.stoi["<EOS>"]:
                break
        
        # تحويل الأرقام إلى كلمات
        words = []
        for idx in caption[1:]:  # تخطي <SOS>
            if idx == vocab.stoi["<EOS>"]:
                break
            if idx in vocab.itos and idx not in [vocab.stoi["<PAD>"]]:
                words.append(vocab.itos[idx])
        
        return " ".join(words)

# ============== بقية الـGUI ==============
@st.cache_resource
def load_vocab():
    with open("vocab.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource  
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # استيراد النموذج من try1
    from try1 import CaptionModel, Vocabulary
    
    # تحميل vocab الفعلي
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    
    # إنشاء النموذج
    model = CaptionModel(
        embed_size=256,
        hidden_size=512,
        vocab_size=len(vocab.itos)
    )
    
    # تحميل الأوزان
    model.load_state_dict(torch.load("caption_model.pth", map_location=device))
    model.to(device)
    model.eval()
    
    return model, vocab, device

# ============== واجهة Streamlit ==============
st.set_page_config(
    page_title="🤖 نظام وصف الصور",
    page_icon="🖼️",
    layout="wide"
)

# CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .result-box {
        padding: 1.5rem;
        background: #f8f9fa;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🖼️ نظام وصف الصور الذكي</h1></div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ الإعدادات")
    max_length = st.slider("طول الوصف", 10, 40, 20)
    st.markdown("---")
    
    # تحميل النموذج
    if st.button("🔄 تحميل النموذج"):
        with st.spinner("جاري التحميل..."):
            if 'model' not in st.session_state:
                model, vocab, device = load_model()
                st.session_state.model = model
                st.session_state.vocab = vocab
                st.session_state.device = device
                st.success("✅ تم تحميل النموذج")
            else:
                st.info("✅ النموذج محمل بالفعل")

# تبويبات
tab1, tab2 = st.tabs(["📤 رفع صورة", "🔗 رابط إنترنت"])

with tab1:
    st.markdown("### رفع صورة")
    uploaded_file = st.file_uploader("اختر صورة", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file and 'model' in st.session_state:
        col1, col2 = st.columns(2)
        
        with col1:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, use_column_width=True)
            
        with col2:
            if st.button("🚀 توليد الوصف"):
                with st.spinner("جاري إنشاء الوصف..."):
                    try:
                        transform = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
                        
                        img_tensor = transform(image)
                        caption = generate_caption(
                            st.session_state.model,
                            img_tensor,
                            st.session_state.vocab,
                            st.session_state.device,
                            max_length
                        )
                        
                        st.markdown(f"""
                        <div class="result-box">
                            <h4>📝 الوصف:</h4>
                            <p style="font-size: 18px; color: #1E3A8A;">
                            {caption}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # تحليل الوصف
                        words = caption.split()
                        col_stats = st.columns(3)
                        col_stats[0].metric("عدد الكلمات", len(words))
                        col_stats[1].metric("كلمات فريدة", len(set(words)))
                        col_stats[2].metric("متوسط الطول", f"{sum(len(w) for w in words)/len(words):.1f}")
                        
                    except Exception as e:
                        st.error(f"خطأ: {str(e)}")

with tab2:
    st.markdown("### رابط صورة")
    url = st.text_input("أدخل رابط الصورة")
    
    if url and 'model' in st.session_state:
        try:
            response = requests.get(url, timeout=10)
            image = Image.open(io.BytesIO(response.content)).convert('RGB')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, use_column_width=True)
                
            with col2:
                if st.button("🚀 توليد الوصف", key="url"):
                    with st.spinner("جاري إنشاء الوصف..."):
                        transform = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
                        
                        img_tensor = transform(image)
                        caption = generate_caption(
                            st.session_state.model,
                            img_tensor,
                            st.session_state.vocab,
                            st.session_state.device,
                            max_length
                        )
                        
                        st.success(f"**الوصف:** {caption}")
                        
        except Exception as e:
            st.error(f"خطأ في تحميل الصورة: {e}")

# ملاحظة
if 'model' not in st.session_state:
    st.info("⚠️ اضغط على 'تحميل النموذج' في الشريط الجانبي أولاً")