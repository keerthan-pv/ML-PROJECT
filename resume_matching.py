import os
import re
import io
import math
import torch
import joblib
import PyPDF2
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import hstack
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

st.set_page_config(page_title="Resume–JD Matcher + Analysis", layout="wide")
st.title("Resume–Job Description Matcher")
st.markdown(
    "Upload a resume PDF and paste a job description. App will show match score, "
    "category prediction, keyword overlap, and insights."
)

def clean_text_basic(text):
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return ' '.join(tokens)

def extract_text_from_pdf(file) -> str:
    text = ""
    try:
        reader = PyPDF2.PdfReader(file)
        for p in reader.pages:
            page_text = p.extract_text()
            if page_text:
                text += page_text + " "
    except Exception:
        try:
            if isinstance(file, (bytes, io.BytesIO)):
                reader = PyPDF2.PdfReader(io.BytesIO(file))
                for p in reader.pages:
                    page_text = p.extract_text()
                    if page_text:
                        text += page_text + " "
        except Exception:
            raise
    return text.strip()

@st.cache_resource
def train_or_load_models(dataset_path="resumes_dataset.jsonl",
                         fine_tuned_model_path="fine_tuned_resume_model"):
    if not os.path.exists(dataset_path):
        st.error(f" Dataset file not found at {dataset_path}.")
        return None, None, None, None, None

    df = pd.read_json(dataset_path, lines=True)
    df['combined'] = (
        df.get('Summary', '').fillna('') + " " +
        df.get('Skills', '').fillna('') + " " +
        df.get('Experience', '').fillna('') + " " +
        df.get('Education', '').fillna('') + " " +
        df.get('Text', '').fillna('')
    ).apply(clean_text_basic)

    X = df['combined'].tolist()
    y = df['Category'].values
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    base_model_name = 'all-MiniLM-L6-v2'
    model = SentenceTransformer(base_model_name)
    fine_tuned = False

    if not os.path.exists(fine_tuned_model_path):
        st.info("🔹 Attempting to fine-tune SentenceTransformer...")
        examples = []
        max_samples = min(400, len(X_train_texts))
        for text, category in zip(X_train_texts[:max_samples], y_train[:max_samples]):
            jd_pos = f"This job requires skills in {category.lower()} and relevant experience."
            examples.append(InputExample(texts=[text, jd_pos], label=1.0))
            other_idx = torch.randint(0, len(X_train_texts), (1,)).item()
            jd_neg = f"This job requires skills in {y_train[other_idx].lower()}."
            examples.append(InputExample(texts=[text, jd_neg], label=0.0))

        train_dataloader = DataLoader(examples, shuffle=True, batch_size=8)
        train_loss = losses.CosineSimilarityLoss(model)
        try:
            model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=2, warmup_steps=50, show_progress_bar=True)
            model.save(fine_tuned_model_path)
            fine_tuned = True
            st.success("Fine-tuning succeeded.")
        except Exception as e:
            st.warning(f"Fine-tuning skipped: {e}")
            model = SentenceTransformer(base_model_name)
    else:
        try:
            model = SentenceTransformer(fine_tuned_model_path)
            fine_tuned = True
        except Exception:
            model = SentenceTransformer(base_model_name)

    X_train_emb = model.encode(X_train_texts, convert_to_tensor=True, show_progress_bar=True)
    X_test_emb = model.encode(X_test_texts, convert_to_tensor=True, show_progress_bar=True)
    X_train_emb_np = X_train_emb.cpu().numpy()
    X_test_emb_np = X_test_emb.cpu().numpy()

    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 3))
    X_train_tfidf = tfidf.fit_transform(X_train_texts)
    X_test_tfidf = tfidf.transform(X_test_texts)

    count_vect = CountVectorizer(max_features=500)
    X_train_count = count_vect.fit_transform(X_train_texts)
    X_test_count = count_vect.transform(X_test_texts)

    X_train_combined = hstack([X_train_emb_np, X_train_tfidf, X_train_count])
    X_test_combined = hstack([X_test_emb_np, X_test_tfidf, X_test_count])

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=20, min_samples_split=5,
        min_samples_leaf=2, class_weight='balanced',
        random_state=42, n_jobs=-1
    )
    clf.fit(X_train_combined, y_train)

    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, ConfusionMatrixDisplay
    from sklearn.preprocessing import label_binarize

    train_preds = clf.predict(X_train_combined)
    test_preds = clf.predict(X_test_combined)

    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)

    train_prec = precision_score(y_train, train_preds, average='weighted', zero_division=0)
    test_prec = precision_score(y_test, test_preds, average='weighted', zero_division=0)

    train_rec = recall_score(y_train, train_preds, average='weighted')
    test_rec = recall_score(y_test, test_preds, average='weighted')

    train_f1 = f1_score(y_train, train_preds, average='weighted')
    test_f1 = f1_score(y_test, test_preds, average='weighted')

    cm = confusion_matrix(y_test, test_preds)
    specificity_per_class = []
    for i in range(len(cm)):
        tn = cm.sum() - (cm[i,:].sum() + cm[:,i].sum() - cm[i,i])
        fp = cm[:,i].sum() - cm[i,i]
        specificity_per_class.append(tn / (tn + fp + 1e-8))
    specificity_avg = sum(specificity_per_class)/len(specificity_per_class)

    try:
        y_test_bin = label_binarize(y_test, classes=list(set(y_test)))
        test_preds_bin = label_binarize(test_preds, classes=list(set(y_test)))
        auc_score = roc_auc_score(y_test_bin, test_preds_bin, average='weighted', multi_class='ovr')
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(y_test_bin.shape[1]):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], test_preds_bin[:, i])
            roc_auc[i] = roc_auc_score(y_test_bin[:, i], test_preds_bin[:, i])
    except Exception as e:
        auc_score = None
        print("ROC-AUC not computed:", e)

    print("\n" + "="*70)
    print("MODEL TRAINING & EVALUATION SUMMARY (Terminal Output)")
    print("="*70)
    print(f"Train Accuracy  : {train_acc:.4f}, Precision: {train_prec:.4f}, Recall: {train_rec:.4f}, F1-score: {train_f1:.4f}")
    print(f"Test  Accuracy  : {test_acc:.4f}, Precision: {test_prec:.4f}, Recall: {test_rec:.4f}, F1-score: {test_f1:.4f}")
    print(f"Test Specificity (avg across classes): {specificity_avg:.4f}")
    if auc_score is not None:
        print(f"Weighted ROC-AUC : {auc_score:.4f}")

    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, test_preds))
    print("Confusion Matrix (Test Set):")
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png", bbox_inches='tight')
    plt.close()
    print("Confusion matrix saved as confusion_matrix.png")

    if auc_score is not None:
        plt.figure(figsize=(8,6))
    for i in range(y_test_bin.shape[1]):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0,1], [0,1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves per Class')
    plt.legend()
    plt.savefig("roc_curves.png", bbox_inches='tight')
    plt.close()
    print("ROC curves saved as roc_curves.png")

    if auc_score is not None:
        plt.figure(figsize=(8,6))
        for i in range(y_test_bin.shape[1]):
            plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
        plt.plot([0,1], [0,1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves per Class')
        plt.legend()
        plt.show()
    print("="*70 + "\n")

    joblib.dump(clf, "resume_classifier.pkl")
    joblib.dump(tfidf, "tfidf_vectorizer.pkl")
    joblib.dump(count_vect, "count_vectorizer.pkl")

    metrics = {
        "train_acc": train_acc, "test_acc": test_acc,
        "train_f1": train_f1, "test_f1": test_f1,
        "y_test": y_test, "test_preds": test_preds,
        "X_test_combined": X_test_combined
    }
    return clf, model, tfidf, count_vect, metrics

def plot_confusion_matrix(y_true, y_pred, classes, figsize=(8,6)):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    return fig

def analyze_resume_vs_jd(resume_text, jd_text, top_k_keywords=30):
    resume_clean = clean_text_basic(resume_text)
    jd_clean = clean_text_basic(jd_text)

    model.eval()
    resume_emb = model.encode([resume_clean], convert_to_tensor=True, normalize_embeddings=True)
    jd_emb = model.encode([jd_clean], convert_to_tensor=True, normalize_embeddings=True)

    dot_product = torch.sum(resume_emb * jd_emb)
    norm_product = torch.norm(resume_emb) * torch.norm(jd_emb)
    semantic_similarity = float((dot_product / norm_product).clamp(-1, 1).item())

    dot_product = torch.sum(resume_emb * jd_emb)
    norm_product = torch.norm(resume_emb) * torch.norm(jd_emb)
    similarity = float((dot_product / norm_product).clamp(-1, 1).item())

    if resume_clean.strip() == jd_clean.strip():
        similarity = 1.0

    resume_emb_np = resume_emb.cpu().numpy()
    resume_tfidf = tfidf.transform([resume_clean])
    resume_count = count_vect.transform([resume_clean])
    X_combined = hstack([resume_emb_np, resume_tfidf, resume_count])
    predicted_category = clf.predict(X_combined)[0]

    resume_tokens = [w for w in resume_clean.split() if len(w) > 2]
    jd_tokens = [w for w in jd_clean.split() if len(w) > 2]

    resume_set = set(resume_tokens)
    jd_set = set(jd_tokens)

    overlap = resume_set & jd_set 
    missing = jd_set - resume_set  
    overlap_ratio = len(overlap) / (len(jd_set) + 1e-8)

    overlap = sorted(list(resume_set & jd_set))
    missing = sorted(list(jd_set - resume_set))
    overlap_ratio = len(overlap) / (len(jd_set) + 1e-8)

    jd_vec = tfidf.transform([jd_clean])
    feature_names = tfidf.get_feature_names_out()
    tfidf_scores = jd_vec.toarray()[0]
    top_idx = tfidf_scores.argsort()[::-1][:top_k_keywords]
    jd_top = [feature_names[i] for i in top_idx if tfidf_scores[i] > 0]

    relevance_score = round(similarity * 100, 2)
    coverage_score = round(overlap_ratio * 100, 2)

    composite_score = (similarity * 0.9 + overlap_ratio * 0.6) * 100

    if 0.4 <= similarity < 0.7:
        composite_score += 5
    composite_score = min(composite_score, 100.0)

    if resume_clean.strip() == jd_clean.strip():
        composite_score = 100.0

    insights = []

    if coverage_score > 15 and composite_score >= 80:
        insights.append("Resume is strongly aligned with the job description.")
    elif 7 < coverage_score <= 15:
        insights.append("Resume is moderately aligned — highlight more JD-specific terms.")
    else:  
        insights.append("Resume needs improvement — tailor skills and experience toward JD.")

    wc_resume = WordCloud(width=600, height=400, background_color='white').generate(" ".join(resume_tokens))
    wc_jd = WordCloud(width=600, height=400, background_color='white').generate(" ".join(jd_tokens))

    return {
        "relevance_score": relevance_score,
        "coverage_score": coverage_score,
        "composite_score": composite_score,
        "predicted_category": predicted_category,
        "overlap": overlap,
        "missing": missing,
        "jd_top": jd_top,
        "wc_resume": wc_resume,
        "wc_jd": wc_jd,
        "insights": insights
    }


with st.spinner("Loading / training models (cached) — please wait..."):
    clf, model, tfidf, count_vect, metrics = train_or_load_models()
if clf is None or model is None:
    st.stop()

resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
jd_text = st.text_area("Paste Job Description", height=220)

if st.button("Analyze Resume vs JD"):
    if not resume_file:
        st.error("Please upload a resume PDF.")
    elif not jd_text.strip():
        st.error("Please paste a job description.")
    else:
        resume_text = extract_text_from_pdf(resume_file)
        analysis = analyze_resume_vs_jd(resume_text, jd_text)  

        if analysis is None or not isinstance(analysis, dict):
            st.error("Resume analysis failed. Please check the input or try a different resume.")
            st.stop()

        st.markdown("### 🔹 Insights")
        insights = analysis.get('insights', ["No insights available."])
        for insight in insights:
            st.markdown(f"""
            <p style='font-size:24px; padding:5px; border-radius:5px;'>{insight}</p>
        """, unsafe_allow_html=True)
        composite_score = analysis.get('composite_score', 0.0)
        st.markdown(f"""
            <p style='font-size:24px; padding:3px; border-radius:2px;'>
                Overall Resume–JD Match Score: <b>{composite_score:.2f}%</b>
            </p>
        """, unsafe_allow_html=True)
        with st.expander("See detailed breakdown"):
            st.markdown(f"<p style='font-size:18px;'>JD Keyword Coverage: <b>{analysis['coverage_score']}%</b></p>", unsafe_allow_html=True)

            st.markdown("#### Keyword Overlap (common terms)")
            st.write(", ".join(analysis['overlap'][:60]) if analysis['overlap'] else "No overlap found.")

            st.markdown("#### Missing / Important JD Keywords")
            st.write(", ".join(analysis['missing'][:60]) if analysis['missing'] else "Resume covers most JD keywords.")

        st.success("Analysis completed")