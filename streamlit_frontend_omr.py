


"""
Streamlit OMR Evaluation & Scoring System
Features:
- Two sections in the UI:
  1) Upload Answer Key (CSV or JSON) or manually input answers
  2) Upload one or more answered OMR sheet images (JPG/PNG)
- Detects the OMR sheet region, warps perspective, thresholds and finds filled bubbles
- Grades the sheet against the answer key and returns per-question and per-subject scores
- Attempts to extract Student Name and Roll No using OCR (pytesseract) from the top area

Limitations & notes:
- This is a generic, template-based solution. You should adapt bubble-grid coordinates (rows/cols), and the location of 'Name'/'Roll' fields to match your printed OMR template for highest accuracy.
- OCR accuracy depends on image quality and printed template. For best results, use high-resolution straight scans or well-aligned photos.

Run:
  pip install -r requirements.txt
  streamlit run streamlit_omr_app.py

Requirements (put in requirements.txt):
streamlit
numpy
opencv-python
pytesseract
pandas
Pillow

Make sure tesseract is installed on your system and accessible (e.g., on Windows: install Tesseract OCR and add to PATH).
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pytesseract
import pandas as pd
import io
import json

# ---------- Helper functions ----------

def read_answer_key(file) -> dict:
    """Accepts CSV or JSON uploaded file and returns a dict {q: 'A'}."""
    try:
        content = file.read()
        file.seek(0)
        if file.name.lower().endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
            # Expect columns: question,answer
            if 'question' in df.columns and 'answer' in df.columns:
                return {int(r['question']): str(r['answer']).strip().upper() for _, r in df.iterrows()}
            else:
                # If CSV is just answers in order
                answers = [str(x).strip().upper() for x in df.iloc[:,0].tolist()]
                return {i+1: answers[i] for i in range(len(answers))}
        else:
            # assume JSON
            j = json.loads(content.decode('utf-8'))
            # either {q:ans} or list
            if isinstance(j, dict):
                return {int(k): str(v).strip().upper() for k,v in j.items()}
            elif isinstance(j, list):
                return {i+1: str(j[i]).strip().upper() for i in range(len(j))}
    except Exception:
        st.error('Unable to parse answer key file. Please upload CSV (question,answer) or JSON.')
        return {}


def resize(image, width=None, height=None):
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        ratio = height / float(h)
        dim = (int(w * ratio), height)
    else:
        ratio = width / float(w)
        dim = (width, int(h * ratio))
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


def find_paper_contour(gray):
    """Find the largest 4-point contour that likely represents the sheet."""
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4,2)
    return None


def order_points(pts):
    rect = np.zeros((4,2), dtype='float32')
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0,maxHeight-1]], dtype='float32')
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def extract_ocr_text(img):
    # Convert to PIL and use pytesseract
    pil = Image.fromarray(img)
    txt = pytesseract.image_to_string(pil)
    return txt


def extract_name_roll_from_top(warped):
    """Attempt to extract name and roll by running OCR on top region and searching for keywords.
    This is a heuristic and will need template adjustment for production.
    """
    h, w = warped.shape[:2]
    top = warped[0:int(h*0.18), :]  # top 18% of sheet
    gray_top = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
    # optional threshold
    _, th = cv2.threshold(gray_top, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    data = pytesseract.image_to_data(th, output_type=pytesseract.Output.DICT)
    full_text = []
    for i, word in enumerate(data['text']):
        if word.strip():
            full_text.append(word.strip())
    full_join = ' '.join(full_text)

    # naive parsing: look for patterns like 'Name: John' or 'Roll: 12345'
    name = ''
    roll = ''
    # try to find 'Name' and next words
    for i, wword in enumerate(data['text']):
        if wword.lower().startswith('name'):
            # take next 5 words as name
            name = ' '.join([x for x in data['text'][i+1:i+6] if x.strip()])
        if wword.lower().startswith('roll') or wword.lower().startswith('rno'):
            roll = ''.join([x for x in data['text'][i+1:i+3] if x.strip()])
    # fallback: try regex on full_join
    if not name or not roll:
        import re
        m = re.search(r'Name[:\-\s]+([A-Za-z ]{2,50})', full_join)
        if m:
            name = m.group(1).strip()
        m2 = re.search(r'Roll[:\-\s]+(\w{2,20})', full_join)
        if m2:
            roll = m2.group(1).strip()
    return name, roll


def grade_sheet(warped_gray, answer_key, questions=100, choices=4, layout=None):
    """Detect bubbles and grade. layout is optional dict with rows,cols and mapping if custom.
    Default assumption: a simple grid of questions arranged top-to-bottom left-to-right.

    Returns: per_question_result dict {q: {'marked': 'A','correct': True/False, 'conf':value}}
    """
    # simple threshold
    blur = cv2.GaussianBlur(warped_gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 10)

    # find contours of bubbles
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubble_contours = []
    for c in cnts:
        (x,y,w,h) = cv2.boundingRect(c)
        ar = w/float(h)
        if 15 < w < 60 and 15 < h < 60 and 0.7 <= ar <= 1.3:
            bubble_contours.append((x,y,w,h,c))

    # sort contours left-to-right then top-to-bottom using y then x
    bubble_contours = sorted(bubble_contours, key=lambda b: (b[1], b[0]))

    # The mapping of bubbles to question & choice needs template knowledge (e.g., 100 Q, 4 choices arranged as 4 columns per row)
    # We'll attempt a simple grouping: group by rows where items in a row are close in y coordinate.

    rows = []
    current_row = []
    last_y = -999
    for b in bubble_contours:
        x,y,w,h,c = b
        if last_y == -999:
            current_row.append(b)
            last_y = y
        else:
            if abs(y - last_y) <= 12:  # same row
                current_row.append(b)
            else:
                rows.append(current_row)
                current_row = [b]
                last_y = y
    if current_row:
        rows.append(current_row)

    # sort each row left->right
    for r in rows:
        r.sort(key=lambda b: b[0])

    # flatten and map to questions
    per_question = {}
    qnum = 1
    for row in rows:
        # assume row contains 'choices' bubbles per question repeated
        # if each question is represented by 'choices' bubbles per row per question, grouping necessary
        for i in range(0, len(row), choices):
            group = row[i:i+choices]
            if len(group) < choices:
                continue
            marks = []
            for j, b in enumerate(group):
                x,y,w,h,c = b
                mask = np.zeros(thresh.shape, dtype='uint8')
                cv2.drawContours(mask, [c], -1, 255, -1)
                total = cv2.countNonZero(mask & thresh)
                marks.append(total)
            # choose max
            max_idx = int(np.argmax(marks))
            # heuristic: if max value strong enough
            marked_choice = chr(ord('A') + max_idx)
            per_question[qnum] = {'marked': marked_choice, 'conf': int(marks[max_idx])}
            qnum += 1
            if qnum > questions:
                break
        if qnum > questions:
            break

    # compute correctness
    results = {}
    correct = 0
    for q in range(1, questions+1):
        ans = answer_key.get(q, None)
        marked = per_question.get(q, {'marked': None, 'conf': 0})['marked']
        is_correct = (ans is not None and marked == ans)
        if is_correct:
            correct += 1
        results[q] = {'answer': ans, 'marked': marked, 'correct': is_correct, 'conf': per_question.get(q, {'conf':0})['conf']}

    return results, correct

# ---------- Streamlit UI ----------

st.set_page_config(page_title='Automated OMR Evaluation', layout='wide')
st.title('Automated OMR Evaluation & Scoring System')
st.write('Two sections: 1) Upload Answer Key  2) Upload Answered OMR Sheet(s). The app will try to grade and extract Name/Roll using OCR.')

col1, col2 = st.columns(2)

with col1:
    st.header('1) Upload Answer Key')
    key_file = st.file_uploader('Upload answer key (CSV with columns: question,answer OR single-column list OR JSON)', type=['csv','json'])
    manual_key = st.text_area('Or paste answers as comma-separated values (like A,B,C,...)', value='')
    answer_key = {}
    if key_file is not None:
        answer_key = read_answer_key(key_file)
        st.success(f'Loaded {len(answer_key)} answers from file')
    elif manual_key.strip():
        answers = [x.strip().upper() for x in manual_key.split(',') if x.strip()]
        answer_key = {i+1: answers[i] for i in range(len(answers))}
        st.success(f'Loaded {len(answer_key)} answers from manual input')
    else:
        st.info('Please upload or paste an answer key. For testing, you can paste a small set of answers.')

with col2:
    st.header('Options')
    questions = st.number_input('Total number of questions', min_value=1, max_value=500, value=100)
    choices = st.selectbox('Choices per question', options=[2,3,4,5], index=2)
    st.write('Note: Template-specific tuning may be required for robust detection.')

st.markdown('---')

st.header('2) Upload Answered OMR Sheet Image(s)')
uploaded = st.file_uploader('Upload image(s) of answered OMR sheets', type=['png','jpg','jpeg'], accept_multiple_files=True)

if uploaded and answer_key:
    for file in uploaded:
        st.subheader(f'Processing: {file.name}')
        image = Image.open(file).convert('RGB')
        img_np = np.array(image)
        orig = img_np.copy()
        # resize for speed but keep aspect ratio
        img_resized = resize(img_np, width=1200)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        paper = find_paper_contour(gray)
        if paper is not None:
            warped = four_point_transform(img_resized, paper)
            warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        else:
            st.warning('Could not detect sheet border — trying to process the full image as sheet.')
            warped = img_resized
            warped_gray = gray

        # show warped preview
        st.image(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), caption='Detected / Warped Sheet Preview', use_column_width=True)

        # extract name & roll
        name, roll = extract_name_roll_from_top(warped)
        st.write(f'**Extracted Name:** {name if name else "(not found)"}   **Roll:** {roll if roll else "(not found)"}')

        # grade
        results, correct = grade_sheet(warped_gray, answer_key, questions=int(questions), choices=int(choices))
        score = correct
        st.write(f'**Total Correct:** {correct} / {questions}   |   **Score (raw):** {score}')

        # show per-question summary (paginated small table)
        df = pd.DataFrame.from_dict(results, orient='index')
        df.index.name = 'question'
        st.dataframe(df)

        # You can add per-subject aggregation if you provide mapping of questions->subject
        st.info('If you want per-subject scores, upload a mapping file (CSV) with columns: question,subject')
        subj_map_file = st.file_uploader(f'Optional: Upload question->subject mapping for {file.name}', type=['csv','json'], key=file.name + '_map')
        if subj_map_file is not None:
            try:
                sm_df = pd.read_csv(subj_map_file) if subj_map_file.name.endswith('.csv') else pd.read_json(subj_map_file)
                mapping = {int(r['question']): r['subject'] for _, r in sm_df.iterrows()}
                df['subject'] = df.index.map(lambda q: mapping.get(int(q), 'Unknown'))
                agg = df.groupby('subject')['correct'].sum().reset_index()
                st.write('Per-subject scores:')
                st.table(agg)
            except Exception as e:
                st.error('Could not parse subject map file. Ensure columns question and subject exist.')

        # Optionally allow download of detailed result as CSV
        to_download = df.reset_index().to_csv(index=False).encode('utf-8')
        st.download_button('Download detailed results CSV', data=to_download, file_name=f'results_{file.name}.csv', mime='text/csv')

elif uploaded and not answer_key:
    st.warning('Please upload an answer key first (left panel).')

else:
    st.info('Upload answer key and OMR image(s) to begin grading.')

st.markdown('---')
st.write('Template tuning tips:')
st.write('- If the bubble detection misses marks, capture a sample scanned sheet and note bubble positions. You can create a custom layout (rows, columns, exact coords) and modify the grading routine to use those coordinates directly.')
st.write('- For robust Name/Roll extraction, consider designing a separate machine-readable QR/Barcode or dedicated boxes for each character (OMR-like) so that OCR becomes unnecessary.')


# EOF
