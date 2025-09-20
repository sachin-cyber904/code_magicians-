"""
streamlit_app.py
Streamlit web app for online OMR evaluation.
Updated to handle file cleanup errors robustly.
"""

import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import json
from io import BytesIO
import base64
import os
from omr_grader import grade_image, SUBJECT_MAPPING

st.title("Automated OMR Evaluation System")

# Inputs
uploaded_image = st.file_uploader("Upload OMR Sheet Image (from mobile camera)", type=["jpg", "png"])
uploaded_answers = st.file_uploader("Upload Answer Keys JSON", type=["json"])
student_name = st.text_input("Student Name")
roll_no = st.text_input("Roll Number")
exam_date = st.date_input("Exam Date")

if st.button("Evaluate"):
    if uploaded_image and uploaded_answers and student_name and roll_no:
        # Save uploaded files temporarily
        image_path = "temp_image.jpg"
        try:
            with open(image_path, "wb") as f:
                f.write(uploaded_image.getvalue())
            
            answers_data = json.load(uploaded_answers)
            
            # Grade
            try:
                res = grade_image(image_path, answers_data, num_questions=100, choices_per_q=4)
                
                st.success(f"Version Detected: {res['version']}")
                st.write("Per-Subject Scores (out of 20 each):")
                for subj, score in res['per_subject_scores'].items():
                    st.write(f"{subj}: {score}/20")
                st.write(f"Total Score: {res['total_score']}/100")
                
                if res["low_confidence_questions"]:
                    st.warning(f"Low confidence questions (review recommended): {res['low_confidence_questions']}")
                
                # Display annotated image
                annotated_img = res["annotated_image"]
                cv2.imwrite("temp_annotated.png", annotated_img)
                st.image("temp_annotated.png", caption="Annotated OMR Sheet")
                
                # Generate Certificate PDF
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.axis('off')
                ax.text(0.5, 0.95, "Examination Certificate", ha='center', va='center', fontsize=20, fontweight='bold')
                ax.text(0.5, 0.85, f"Student Name: {student_name}", ha='center', va='center', fontsize=14)
                ax.text(0.5, 0.80, f"Roll Number: {roll_no}", ha='center', va='center', fontsize=14)
                ax.text(0.5, 0.75, f"Exam Date: {exam_date}", ha='center', va='center', fontsize=14)
                ax.text(0.5, 0.70, f"Sheet Version: {res['version']}", ha='center', va='center', fontsize=14)
                ax.text(0.5, 0.65, "Scores:", ha='center', va='center', fontsize=16)
                
                y_pos = 0.60
                for subj, score in res['per_subject_scores'].items():
                    ax.text(0.5, y_pos, f"{subj}: {score}/20", ha='center', va='center', fontsize=12)
                    y_pos -= 0.05
                ax.text(0.5, y_pos - 0.05, f"Total: {res['total_score']}/100", ha='center', va='center', fontsize=14, fontweight='bold')
                ax.text(0.5, 0.10, "Issued by Innomatics Evaluation System", ha='center', va='center', fontsize=10)
                
                pdf_buffer = BytesIO()
                with PdfPages(pdf_buffer) as pdf:
                    pdf.savefig(fig)
                pdf_buffer.seek(0)
                
                b64 = base64.b64encode(pdf_buffer.read()).decode()
                href = f'<a href="data:application/pdf;base64,{b64}" download="certificate.pdf">Download Certificate PDF</a>'
                st.markdown(href, unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"Error during evaluation: {str(e)}")
            
            finally:
                # Clean up temp files with error handling
                for temp_file in [image_path, "temp_annotated.png"]:
                    try:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                    except (PermissionError, FileNotFoundError, OSError) as e:
                        st.warning(f"Could not delete temporary file {temp_file}: {str(e)}")
        
        except Exception as e:
            st.error(f"Error saving uploaded image: {str(e)}")
    else:
        st.error("Please provide all required inputs.")
