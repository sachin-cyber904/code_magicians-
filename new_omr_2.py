

"""
omr_grader.py
Enhanced OMR evaluation script using OpenCV for scalable, automated grading.
Supports multiple sheet versions (A-D), per-subject scoring, and mobile camera images.
Designed for integration with a web app (e.g., Streamlit).

Key Enhancements:
- Detects sheet version (A, B, C, D) from top row of bubbles.
- Supports per-subject scoring (5 subjects, 20 questions each, total 100 questions).
- Improved preprocessing for mobile camera images (CLAHE + morphological operations).
- Outputs include per-subject scores, total, and debug visualization.
- Error tolerance: Flags low-confidence detections for manual review (<0.5% error goal).

Assumptions:
- OMR sheet layout: Top row (row 0) has 4 bubbles for versions A, B, C, D.
- Questions: Rows 1-100, 4 choices each (A-D).
- Subjects: Math (1-20), Science (21-40), English (41-60), History (61-80), Geography (81-100).
- Answer key: JSON file, e.g., {'A': {1:'B', ...}, 'B': {...}}.

Usage:
  python omr_grader.py --image path/to/scan.jpg --answers answers.json --questions 100 --choices 4 --output out
"""

import cv2
import numpy as np
import argparse
import json
import os
import csv  # Added to fix "csv is not defined" error

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
import cv2
import numpy as np
import argparse
import json
import os
import csv
import pytesseract
from PIL import Image

# Tesseract path (update for your system if needed)
# Example for Windows:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

CHOICE_LETTERS = "ABCD"

SUBJECT_MAPPING = {
    'PYTHON': range(1, 21),
    'DATA ANANLYSIS': range(21, 41),
    'MySQL': range(41, 61),
    'POWER BI': range(61, 81),
    'Adv STATA': range(81, 101)
}

# ------------------ OCR SECTION -------------------
def extract_text_fields(warped_img, debug=False):
    """
    Extracts student info (name, roll number, etc.) using OCR.
    Assumes text fields are at the top of the OMR sheet.
    Adjust crop regions as per your sheet layout.
    """
    h, w, _ = warped_img.shape

    # Example crop: top strip (10% of height)
    text_region = warped_img[0:int(h*0.1), 0:w]

    # Convert to grayscale & threshold for OCR
    gray = cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    if debug:
        cv2.imwrite("debug_text_region.png", thresh)

    # OCR
    text = pytesseract.image_to_string(thresh, config="--psm 6")
    return text.strip()
# --------------------------------------------------

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
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
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def find_document_contour(gray):
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    edged = cv2.Canny(enhanced, 50, 150)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
    return None

def parse_answers_arg(ans_path):
    if os.path.exists(ans_path):
        with open(ans_path, 'r') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"Answer key file not found: {ans_path}")

def grade_image(image_path, answer_keys, num_questions=100, choices_per_q=4, debug=False):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    orig = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    doc_cnt = find_document_contour(enhanced)
    if doc_cnt is not None:
        warped = four_point_transform(orig, doc_cnt)
    else:
        warped = orig.copy()

    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    enhanced_warped = clahe.apply(warped_gray)
    thresh = cv2.threshold(enhanced_warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    if debug:
        cv2.imwrite("debug_thresh.png", thresh)

    h, w = thresh.shape
    rows = num_questions + 1
    cols = choices_per_q
    cell_h = h / rows
    cell_w = w / cols

    responses = {}
    annotated = warped.copy()
    low_confidence_qs = []

    # -------- OCR STEP --------
    ocr_text = extract_text_fields(warped, debug=debug)
    # --------------------------

    # (OMR detection logic unchanged...)
    # --- version detection
    version_responses = []
    for c in range(cols):
        c_x = int((c + 0.5) * cell_w)
        c_y = int(0.5 * cell_h)
        box_w = int(cell_w * 0.6)
        box_h = int(cell_h * 0.6)
        x1 = max(c_x - box_w // 2, 0)
        y1 = max(c_y - box_h // 2, 0)
        x2 = min(c_x + box_w // 2, w - 1)
        y2 = min(c_y + box_h // 2, h - 1)
        roi = thresh[y1:y2, x1:x2]
        total = cv2.countNonZero(roi)
        version_responses.append((c, total, (x1, y1, x2, y2)))

    ink_values = [v[1] for v in version_responses]
    max_idx = np.argmax(ink_values)
    max_val = ink_values[max_idx]
    median = np.median(ink_values)
    if debug:
        print(f"Version row ink values: {ink_values}")
    if median == 0:
        marked = max_val > 50
    else:
        marked = (max_val >= median * 1.4) or (max_val > 200)
    if not marked:
        raise ValueError("Sheet version not detected.")
    version_col = version_responses[max_idx][0]
    version = CHOICE_LETTERS[version_col]
    cv2.rectangle(annotated, version_responses[max_idx][2], (0, 255, 0), 2)

    if version not in answer_keys:
        raise ValueError(f"No answer key for version {version}")
    answer_key = answer_keys[version]

    # (Loop over all questions - unchanged)
    result_per_q = {}
    correct_count = 0
    for q in range(1, num_questions + 1):
        # ...
        pass  # (your existing question loop goes here)

    # Compute scores (same as your version)
    per_subject_scores = {subj: 0 for subj in SUBJECT_MAPPING}
    total_score = sum(per_subject_scores.values())

    return {
        "annotated_image": annotated,
        "result_per_question": result_per_q,
        "per_subject_scores": per_subject_scores,
        "total_score": total_score,
        "max_score": num_questions,
        "version": version,
        "low_confidence_questions": low_confidence_qs,
        "ocr_text": ocr_text   # <--- NEW FIELD
    }


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
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def find_document_contour(gray):
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    edged = cv2.Canny(enhanced, 50, 150)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
    return None

def parse_answers_arg(ans_path):
    if os.path.exists(ans_path):
        with open(ans_path, 'r') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"Answer key file not found: {ans_path}")

CHOICE_LETTERS = "ABCD"

SUBJECT_MAPPING = {
    'Math': range(1, 21),
    'Science': range(21, 41),
    'English': range(41, 61),
    'History': range(61, 81),
    'Geography': range(81, 101)
}

def grade_image(image_path, answer_keys, num_questions=100, choices_per_q=4, debug=False):
    image = cv2.imread(input())
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    orig = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    doc_cnt = find_document_contour(enhanced)
    if doc_cnt is not None:
        warped = four_point_transform(orig, doc_cnt)
    else:
        warped = orig.copy()

    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    enhanced_warped = clahe.apply(warped_gray)
    thresh = cv2.threshold(enhanced_warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    if debug:
        cv2.imwrite("debug_thresh.png", thresh)

    h, w = thresh.shape
    rows = num_questions + 1
    cols = choices_per_q

    cell_h = h / rows
    cell_w = w / cols

    responses = {}
    annotated = warped.copy()
    low_confidence_qs = []

    version_responses = []
    for c in range(cols):
        c_x = int((c + 0.5) * cell_w)
        c_y = int(0.5 * cell_h)
        box_w = int(cell_w * 0.6)
        box_h = int(cell_h * 0.6)
        x1 = max(c_x - box_w // 2, 0)
        y1 = max(c_y - box_h // 2, 0)
        x2 = min(c_x + box_w // 2, w - 1)
        y2 = min(c_y + box_h // 2, h - 1)
        roi = thresh[y1:y2, x1:x2]
        total = cv2.countNonZero(roi)
        version_responses.append((c, total, (x1, y1, x2, y2)))

    ink_values = [v[1] for v in version_responses]
    max_idx = np.argmax(ink_values)
    max_val = ink_values[max_idx]
    median = np.median(ink_values)
    if debug:
        print(f"Version row ink values: {ink_values}")
        print(f"Max ink: {max_val}, Median ink: {median}")
    if median == 0:
        marked = max_val > 50
    else:
        marked = (max_val >= median * 1.4) or (max_val > 200)
    if not marked:
        raise ValueError("Sheet version not detected.")
    version_col = version_responses[max_idx][0]
    version = CHOICE_LETTERS[version_col]
    cv2.rectangle(annotated, version_responses[max_idx][2], (0, 255, 0), 2)

    if version not in answer_keys:
        raise ValueError(f"No answer key for version {version}")
    answer_key = answer_keys[version]

    for q in range(1, rows):
        q_idx = q
        for c in range(cols):
            c_x = int((c + 0.5) * cell_w)
            c_y = int((q + 0.5) * cell_h)
            box_w = int(cell_w * 0.6)
            box_h = int(cell_h * 0.6)
            x1 = max(c_x - box_w // 2, 0)
            y1 = max(c_y - box_h // 2, 0)
            x2 = min(c_x + box_w // 2, w - 1)
            y2 = min(c_y + box_h // 2, h - 1)
            roi = thresh[y1:y2, x1:x2]
            total = cv2.countNonZero(roi)
            responses.setdefault(q_idx, []).append((c, total, (x1, y1, x2, y2)))

    result_per_q = {}
    correct_count = 0
    for q in range(1, num_questions + 1):
        cols_info = responses[q]
        ink_values = [v[1] for v in cols_info]
        max_idx = np.argmax(ink_values)
        max_val = ink_values[max_idx]
        median = np.median(ink_values)
        if median == 0:
            marked = max_val > 50
        else:
            marked = (max_val >= median * 1.4) or (max_val > 200)
        
        sorted_inks = sorted(ink_values, reverse=True)
        if len(sorted_inks) > 1 and sorted_inks[0] > 0 and (sorted_inks[0] - sorted_inks[1]) / sorted_inks[0] < 0.15:
            low_confidence_qs.append(q)
        
        if marked:
            selected_col = cols_info[max_idx][0]
            result_per_q[q] = CHOICE_LETTERS[selected_col]
            x1, y1, x2, y2 = cols_info[max_idx][2]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
        else:
            result_per_q[q] = None

    per_subject_scores = {subj: 0 for subj in SUBJECT_MAPPING}
    for q in range(1, num_questions + 1):
        selected = result_per_q.get(q)
        correct = answer_key.get(str(q)) or answer_key.get(q)
        if selected == correct:
            correct_count += 1
            for subj, q_range in SUBJECT_MAPPING.items():
                if q in q_range:
                    per_subject_scores[subj] += 1
                    break

    total_score = sum(per_subject_scores.values())

    return {
        "annotated_image": annotated,
        "result_per_question": result_per_q,
        "per_subject_scores": per_subject_scores,
        "total_score": total_score,
        "max_score": num_questions,
        "version": version,
        "low_confidence_questions": low_confidence_qs
    }

def save_report(output_img, result, per_subject_scores, total_score, out_prefix):
    img_path = out_prefix + "_annotated.png"
    cv2.imwrite(img_path, output_img)
    csv_path = out_prefix + "_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "selected"])
        for q, sel in sorted(result.items()):
            writer.writerow([q, sel if sel is not None else ""])
    return img_path, csv_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to OMR image")
    parser.add_argument("--answers", required=True, help="Path to JSON answer keys")
    parser.add_argument("--questions", type=int, default=100, help="Number of questions")
    parser.add_argument("--choices", type=int, default=4, help="Choices per question")
    parser.add_argument("--output", default="out", help="Output prefix")
    args = parser.parse_args()

    answer_keys = parse_answers_arg(args.answers)
    res = grade_image(args.image, answer_keys, num_questions=args.questions, choices_per_q=args.choices, debug=True)
    img_path, csv_path = save_report(res["annotated_image"], res["result_per_question"], res["per_subject_scores"], res["total_score"], args.output)
    print(f"Version: {res['version']}")
    print(f"Per-subject scores: {res['per_subject_scores']}")
    print(f"Total Score: {res['total_score']} / {res['max_score']}")
    if res["low_confidence_questions"]:
        print(f"Low confidence questions: {res['low_confidence_questions']}")
    print(f"Annotated image saved to: {img_path}")
    print(f"Per-question CSV saved to: {csv_path}")
