"""
omr_grader.py
Simple OMR evaluation script using OpenCV.

Usage:
  python omr_grader.py --image path/to/scan.jpg --answers 1:A,2:C,3:B,... --questions 25 --choices 4 --output out.png

Or use it as a module from the Streamlit app provided separately.
"""

import cv2
import numpy as np
import argparse
import csv
import os

def order_points(pts):
    # Order points: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4,2), dtype="float32")
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
        [0,0],
        [maxWidth-1, 0],
        [maxWidth-1, maxHeight-1],
        [0, maxHeight-1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def find_document_contour(gray):
    # find largest quadrilateral contour
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4,2)
    return None

def parse_answers_arg(ans_str):
    # expects "1:A,2:C,3:B" etc. returns dict {1:'A', ...}
    ans = {}
    if not ans_str:
        return ans
    pairs = ans_str.split(',')
    for p in pairs:
        if ':' in p:
            q, a = p.split(':')
            ans[int(q.strip())] = a.strip().upper()
    return ans

CHOICE_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def grade_image(image_path, answer_key, num_questions=25, choices_per_q=4, debug=False):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    orig = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    doc_cnt = find_document_contour(gray)
    if doc_cnt is not None:
        warped = four_point_transform(orig, doc_cnt)
    else:
        # fallback use the whole image
        warped = orig.copy()

    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # dimensions & grid detection
    h, w = thresh.shape
    # We'll assume bubbles are arranged row-wise: num_questions rows, choices_per_q columns
    # If your sheet has two columns of questions, pre-process accordingly (see notes below).
    rows = num_questions
    cols = choices_per_q

    # compute size of each cell
    cell_h = h / rows
    cell_w = w / cols

    # For robustness, find blobs and then map to grid cells (we'll sample ROI around expected centers)
    responses = {}
    annotated = warped.copy()
    correct_count = 0

    # use adaptive bounding box for each expected bubble location
    for q in range(rows):
        # y-range for this question (use center)
        for c in range(cols):
            # compute center of expected bubble
            c_x = int((c + 0.5) * cell_w)
            c_y = int((q + 0.5) * cell_h)
            # sample box around center
            box_w = int(cell_w * 0.6)
            box_h = int(cell_h * 0.6)
            x1 = max(c_x - box_w//2, 0)
            y1 = max(c_y - box_h//2, 0)
            x2 = min(c_x + box_w//2, w-1)
            y2 = min(c_y + box_h//2, h-1)
            roi = thresh[y1:y2, x1:x2]
            total = cv2.countNonZero(roi)
            # store raw ink amount
            responses.setdefault(q+1, []).append((c, total, (x1,y1,x2,y2)))

    # decide selected choice: pick the column with maximum ink and above a threshold relative to median
    result_per_q = {}
    for q in range(1, rows+1):
        cols_info = responses[q]
        ink_values = [v[1] for v in cols_info]
        max_idx = int(np.argmax(ink_values))
        max_val = ink_values[max_idx]
        # threshold: must be significant compared to median or mean
        median = np.median(ink_values)
        # simple heuristic: ``marked`` if max_val >= median * 1.5 and absolute > small constant
        if median == 0:
            marked = max_val > 100  # some absolute min (depends on scan resolution)
        else:
            marked = (max_val >= median * 1.6) or (max_val > 300)
        if marked:
            selected_col = cols_info[max_idx][0]
            result_per_q[q] = CHOICE_LETTERS[selected_col]
            # annotate selected bubble
            x1,y1,x2,y2 = cols_info[max_idx][2]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255,0,0), 2)
        else:
            result_per_q[q] = None

    # grading
    marked_image = annotated.copy()
    for q in range(1, rows+1):
        selected = result_per_q.get(q)
        correct = answer_key.get(q)
        # draw text
        text_pos = (10, int((q-0.5) * cell_h + 15))
        if selected is None:
            cv2.putText(marked_image, f"Q{q}: -", (10, int((q-0.5) * cell_h + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        else:
            if correct is not None and selected == correct:
                correct_count += 1
                cv2.putText(marked_image, f"Q{q}: {selected} ✓", (10, int((q-0.5) * cell_h + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,150,0), 1)
            else:
                cv2.putText(marked_image, f"Q{q}: {selected} ✗ (ans:{correct})", (10, int((q-0.5) * cell_h + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    score = correct_count
    return {
        "annotated_image": marked_image,
        "result_per_question": result_per_q,
        "score": score,
        "max_score": rows
    }

def save_report(output_img, result, out_prefix):
    # save annotated image and CSV
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
    parser.add_argument("--image", required=True, help="Path to scanned/photographed OMR image")
    parser.add_argument("--answers", required=True,
                        help="Answer key as comma separated e.g. '1:A,2:C,3:B' or path to a CSV file (q,choice)")
    parser.add_argument("--questions", type=int, default=25, help="Number of questions")
    parser.add_argument("--choices", type=int, default=4, help="Choices per question (e.g., 4 for A-D)")
    parser.add_argument("--output", default="out", help="Output prefix")
    args = parser.parse_args()

    # parse answers
    answer_key = {}
    if os.path.exists(args.answers):
        # read csv q,choice
        with open(args.answers, newline='') as f:
            r = csv.reader(f)
            for row in r:
                if not row: continue
                q = int(row[0])
                ch = row[1].strip().upper()
                answer_key[q] = ch
    else:
        answer_key = parse_answers_arg(args.answers)

    res = grade_image(args.image, answer_key, num_questions=args.questions, choices_per_q=args.choices)
    img_path, csv_path = save_report(res["annotated_image"], res["result_per_question"], args.output)
    print(f"Score: {res['score']} / {res['max_score']}")
    print(f"Annotated image saved to: {img_path}")
    print(f"Per-question CSV saved to: {csv_path}")
