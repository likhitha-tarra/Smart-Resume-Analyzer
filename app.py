import streamlit as st
import PyPDF2
import re

try:
    from docx import Document
except ImportError:
    Document = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Smart Resume Analyzer",
    page_icon="📄",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 18px;
    color: #777;
    margin-bottom: 25px;
}

.section-title {
    font-size: 25px;
    font-weight: 700;
    margin-top: 25px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">📄 Smart Resume Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered resume analysis and job matching</div>',
    unsafe_allow_html=True
)


# =========================================================
# SKILLS DATABASE
# =========================================================

ALL_SKILLS = [
    "Python",
    "Java",
    "C++",
    "C",
    "SQL",
    "HTML",
    "CSS",
    "JavaScript",
    "React",
    "Node.js",
    "Machine Learning",
    "Deep Learning",
    "Artificial Intelligence",
    "Data Science",
    "Data Structures",
    "Algorithms",
    "Object Oriented Programming",
    "Git",
    "GitHub",
    "AWS",
    "Docker",
    "FastAPI",
    "REST API",
    "Database Management",
    "Exploratory Data Analysis",
    "Data Preprocessing",
    "Local Storage"
]


# =========================================================
# JOB SKILLS
# =========================================================

JOB_SKILLS = [
    "Python",
    "SQL",
    "Machine Learning",
    "Git",
    "AWS",
    "Docker",
    "FastAPI",
    "REST API"
]


# =========================================================
# RESUME SECTIONS
# =========================================================

SECTION_KEYWORDS = {
    "Education": [
        "education",
        "academic",
        "degree",
        "b.tech",
        "btech"
    ],

    "Experience": [
        "experience",
        "internship",
        "intern"
    ],

    "Skills": [
        "skills",
        "technical skills",
        "programming languages"
    ],

    "Projects": [
        "projects",
        "project"
    ],

    "Certifications": [
        "certifications",
        "certification",
        "certificate"
    ],

    "Achievements": [
        "achievements",
        "achievement",
        "awards"
    ]
}


# =========================================================
# EXTRACT PDF TEXT
# =========================================================

def extract_pdf_text(uploaded_file):

    text = ""

    reader = PyPDF2.PdfReader(uploaded_file)

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# =========================================================
# EXTRACT DOCX TEXT
# =========================================================

def extract_docx_text(uploaded_file):

    if Document is None:
        return ""

    document = Document(uploaded_file)

    text = []

    for paragraph in document.paragraphs:
        text.append(paragraph.text)

    return "\n".join(text)


# =========================================================
# EXTRACT RESUME TEXT
# =========================================================

def extract_resume_text(uploaded_file):

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".pdf"):
        return extract_pdf_text(uploaded_file)

    elif file_name.endswith(".docx"):
        return extract_docx_text(uploaded_file)

    return ""


# =========================================================
# CLEAN TEXT
# =========================================================

def clean_text(text):

    text = text.replace("\x00", " ")

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# FIND RESUME SKILLS
# =========================================================

def find_skills(text):

    found = []

    text_lower = text.lower()

    for skill in ALL_SKILLS:

        if skill.lower() in text_lower:

            found.append(skill)

    return found


# =========================================================
# FIND JOB SKILLS
# =========================================================

def find_job_skills(job_description):

    found = []

    job_lower = job_description.lower()

    for skill in JOB_SKILLS:

        if skill.lower() in job_lower:

            found.append(skill)

    return found


# =========================================================
# FIND RESUME SECTIONS
# =========================================================

def find_sections(text):

    found = []

    text_lower = text.lower()

    for section, keywords in SECTION_KEYWORDS.items():

        for keyword in keywords:

            if keyword.lower() in text_lower:

                found.append(section)

                break

    return found


# =========================================================
# ATS SCORE
# =========================================================

def calculate_ats_score(
    resume_skills,
    found_sections,
    resume_text
):

    skill_score = min(
        len(resume_skills) * 2,
        35
    )

    section_score = min(
        len(found_sections) * 7,
        42
    )

    if len(resume_text) >= 1200:

        length_score = 23

    elif len(resume_text) >= 700:

        length_score = 15

    else:

        length_score = 8

    total = (
        skill_score
        + section_score
        + length_score
    )

    return int(
        min(total, 100)
    )


# =========================================================
# JOB MATCH
# =========================================================

def calculate_job_match(
    resume_skills,
    job_skills
):

    if not job_skills:

        return 0.0

    resume_lower = [
        skill.lower()
        for skill in resume_skills
    ]

    matching = [
        skill
        for skill in job_skills
        if skill.lower() in resume_lower
    ]

    score = (
        len(matching)
        / len(job_skills)
    ) * 100

    return round(
        float(score),
        1
    )


# =========================================================
# SEMANTIC SIMILARITY
# =========================================================

def calculate_semantic_similarity(
    resume_text,
    job_description
):

    if not resume_text or not job_description:

        return 0.0

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True
        )

        vectors = vectorizer.fit_transform(
            [
                resume_text,
                job_description
            ]
        )

        similarity = cosine_similarity(
            vectors[0:1],
            vectors[1:2]
        )[0][0]

        job_keywords = [
            "python",
            "sql",
            "fastapi",
            "docker",
            "aws",
            "git",
            "machine learning",
            "rest api",
            "data science",
            "deep learning",
            "javascript",
            "react"
        ]

        resume_lower = resume_text.lower()

        job_lower = job_description.lower()

        relevant_keywords = [
            word
            for word in job_keywords
            if word in job_lower
        ]

        matched_keywords = [
            word
            for word in relevant_keywords
            if word in resume_lower
        ]

        if relevant_keywords:

            keyword_score = (
                len(matched_keywords)
                / len(relevant_keywords)
            )

        else:

            keyword_score = 0

        final_score = (
            (similarity * 60)
            + (keyword_score * 40)
        )

        return round(
            min(float(final_score), 100.0),
            1
        )

    except Exception:

        return 0.0


# =========================================================
# RESUME UPLOAD
# =========================================================

st.markdown(
    '<div class="section-title">📥 Resume & Job Details</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload your Resume",
    type=["pdf", "docx"]
)


# =========================================================
# JOB DESCRIPTION
# =========================================================

job_description = st.text_area(
    "📝 Paste Job Description",
    height=220,
    placeholder="Paste the job description here..."
)


# =========================================================
# ANALYZE BUTTON
# =========================================================

analyze = st.button(
    "🚀 Analyze Resume",
    use_container_width=True
)


# =========================================================
# ANALYSIS
# =========================================================

if analyze:

    # -----------------------------------------------------
    # CHECK RESUME
    # -----------------------------------------------------

    if uploaded_file is None:

        st.error(
            "Please upload your resume first."
        )

        st.stop()


    # -----------------------------------------------------
    # CHECK JOB DESCRIPTION
    # -----------------------------------------------------

    if not job_description.strip():

        st.error(
            "Please paste a job description."
        )

        st.stop()


    # -----------------------------------------------------
    # EXTRACT RESUME
    # -----------------------------------------------------

    resume_text = extract_resume_text(
        uploaded_file
    )

    resume_text = clean_text(
        resume_text
    )


    if not resume_text:

        st.error(
            "Could not extract text from the resume."
        )

        st.stop()


    # -----------------------------------------------------
    # FIND SKILLS
    # -----------------------------------------------------

    resume_skills = find_skills(
        resume_text
    )


    # -----------------------------------------------------
    # FIND SECTIONS
    # -----------------------------------------------------

    found_sections = find_sections(
        resume_text
    )


    # -----------------------------------------------------
    # FIND JOB SKILLS
    # -----------------------------------------------------

    job_skills = find_job_skills(
        job_description
    )


    # -----------------------------------------------------
    # MATCHING SKILLS
    # -----------------------------------------------------

    matching_skills = [

        skill

        for skill in job_skills

        if skill.lower()
        in [
            x.lower()
            for x in resume_skills
        ]

    ]


    # -----------------------------------------------------
    # MISSING SKILLS
    # -----------------------------------------------------

    missing_skills = [

        skill

        for skill in job_skills

        if skill.lower()
        not in [
            x.lower()
            for x in resume_skills
        ]

    ]


    # -----------------------------------------------------
    # CALCULATE ATS
    # -----------------------------------------------------

    ats_score = calculate_ats_score(
        resume_skills,
        found_sections,
        resume_text
    )


    # -----------------------------------------------------
    # CALCULATE JOB MATCH
    # -----------------------------------------------------

    job_match = calculate_job_match(
        resume_skills,
        job_skills
    )


    # -----------------------------------------------------
    # CALCULATE SEMANTIC SCORE
    # -----------------------------------------------------

    semantic_score = calculate_semantic_similarity(
        resume_text,
        job_description
    )


    # Make sure values are normal Python numbers

    ats_score = int(ats_score)

    job_match = float(job_match)

    semantic_score = float(
        semantic_score
    )


    # =====================================================
    # DASHBOARD
    # =====================================================

    st.divider()

    st.header(
        "📊 Resume Analysis Dashboard"
    )


    # -----------------------------------------------------
    # SCORE CARDS
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "📊 ATS SCORE",
            f"{ats_score}/100"
        )


    with col2:

        st.metric(
            "🎯 JOB MATCH",
            f"{job_match:.0f}%"
        )


    with col3:

        st.metric(
            "🧠 SEMANTIC SCORE",
            f"{semantic_score:.0f}%"
        )


    # =====================================================
    # PROGRESS BARS
    # =====================================================

    st.write("### ATS Score")

    st.progress(
        float(ats_score) / 100.0
    )


    st.write("### Job Match")

    st.progress(
        float(job_match) / 100.0
    )


    st.write("### Semantic Similarity")

    st.progress(
        float(semantic_score) / 100.0
    )


    # =====================================================
    # MATCHING SKILLS
    # =====================================================

    st.subheader(
        "✅ Matching Job Skills"
    )

    if matching_skills:

        for skill in matching_skills:

            st.write(
                f"✅ {skill}"
            )

    else:

        st.warning(
            "No matching job skills found."
        )


    # =====================================================
    # MISSING SKILLS
    # =====================================================

    st.subheader(
        "❌ Missing Job Skills"
    )

    if missing_skills:

        for skill in missing_skills:

            st.write(
                f"❌ {skill}"
            )

    else:

        st.success(
            "Excellent! All important job skills were found."
        )


    # =====================================================
    # ALL RESUME SKILLS
    # =====================================================

    st.subheader(
        "🛠️ Skills Found in Resume"
    )

    if resume_skills:

        st.write(
            " • ".join(resume_skills)
        )

    else:

        st.warning(
            "No skills detected."
        )


    # =====================================================
    # JOB REQUIRED SKILLS
    # =====================================================

    st.subheader(
        "🎯 Skills Required by Job"
    )

    if job_skills:

        st.write(
            " • ".join(job_skills)
        )

    else:

        st.info(
            "No predefined job skills detected."
        )


    # =====================================================
    # RESUME SECTIONS
    # =====================================================

    st.subheader(
        "📑 Resume Sections"
    )

    all_sections = list(
        SECTION_KEYWORDS.keys()
    )

    section_cols = st.columns(3)


    for index, section in enumerate(
        all_sections
    ):

        with section_cols[
            index % 3
        ]:

            if section in found_sections:

                st.success(
                    f"✅ {section}"
                )

            else:

                st.error(
                    f"❌ {section}"
                )


    # =====================================================
    # STRENGTHS
    # =====================================================

    st.subheader(
        "💪 Resume Strengths"
    )


    if len(resume_skills) >= 8:

        st.write(
            "✅ Strong technical skills coverage."
        )

    else:

        st.write(
            "✅ Basic technical skills coverage."
        )


    if len(matching_skills) >= 3:

        st.write(
            f"✅ Resume matches "
            f"{len(matching_skills)} important job skills."
        )

    elif len(matching_skills) > 0:

        st.write(
            f"⚠️ Resume matches only "
            f"{len(matching_skills)} job skills."
        )

    else:

        st.write(
            "⚠️ Resume does not currently match "
            "the target job skills."
        )


    if "Projects" in found_sections:

        st.write(
            "✅ Projects section is present."
        )


    if "Experience" in found_sections:

        st.write(
            "✅ Experience section is present."
        )


    if "Achievements" in found_sections:

        st.write(
            "✅ Achievements section is present."
        )


    # =====================================================
    # AREAS TO IMPROVE
    # =====================================================

    st.subheader(
        "⚠️ Areas to Improve"
    )


    if missing_skills:

        st.write(
            "Consider adding or highlighting "
            "these job-relevant skills if you "
            "genuinely have them:"
        )

        for skill in missing_skills:

            st.write(
                f"• {skill}"
            )


    if "Certifications" not in found_sections:

        st.write(
            "• Add relevant certifications."
        )


    # =====================================================
    # AI IMPROVEMENT PLAN
    # =====================================================

    st.subheader(
        "🤖 AI Resume Improvement Plan"
    )

    st.write(
        "Based on the resume and target job, "
        "here is a prioritized improvement plan."
    )


    # -----------------------------------------------------
    # PRIORITY 1
    # -----------------------------------------------------

    st.write(
        "### 🥇 Priority 1 — Job-Relevant Skills"
    )


    if missing_skills:

        st.write(
            "The following skills appear in the "
            "job description but were not detected "
            "in your resume:"
        )

        for skill in missing_skills:

            st.write(
                f"🔹 {skill}"
            )

        st.caption(
            "Only add skills that you genuinely "
            "know or have experience with."
        )

    else:

        st.success(
            "All important job skills were detected."
        )


    # -----------------------------------------------------
    # PRIORITY 2
    # -----------------------------------------------------

    st.write(
        "### 🥈 Priority 2 — Projects"
    )


    if "Projects" in found_sections:

        st.write(
            "✅ Your resume already contains a "
            "Projects section."
        )

        st.write(
            "💡 Consider highlighting projects "
            "that use the job-relevant technologies."
        )

    else:

        st.write(
            "⚠️ Add at least 1–2 relevant projects."
        )


    # -----------------------------------------------------
    # PRIORITY 3
    # -----------------------------------------------------

    st.write(
        "### 🥉 Priority 3 — Resume Keywords"
    )


    if job_skills:

        st.write(
            "Important job keywords:"
        )

        st.write(
            " • ".join(job_skills)
        )


    # -----------------------------------------------------
    # PRIORITY 4
    # -----------------------------------------------------

    st.write(
        "### 📈 Priority 4 — Achievements"
    )

    st.write(
        "Show measurable results instead of "
        "only listing responsibilities."
    )

    st.write(
        "**Weak:** Developed a Python project."
    )

    st.write(
        "**Better:** Developed a Python application "
        "that reduced processing time by 30%."
    )


    # -----------------------------------------------------
    # PRIORITY 5
    # -----------------------------------------------------

    st.write(
        "### 🎯 Priority 5 — Overall Match"
    )


    if job_match >= 75:

        st.success(
            f"Excellent job match: "
            f"{job_match:.0f}%."
        )

    elif job_match >= 50:

        st.warning(
            f"Moderate job match: "
            f"{job_match:.0f}%."
        )

    else:

        st.warning(
            f"Current job match: "
            f"{job_match:.0f}%. "
            f"Your resume needs customization "
            f"for this role."
        )


    # =====================================================
    # GENERAL SUGGESTIONS
    # =====================================================

    st.subheader(
        "💡 General Suggestions"
    )


    suggestions = [

        "Keep the resume clean and ATS-friendly.",

        "Use strong action words such as "
        "Developed, Built, Implemented and Designed.",

        "Add measurable achievements wherever possible.",

        "Keep skills relevant to the target role.",

        "Add GitHub and LinkedIn links if available.",

        "Do not add skills that you do not genuinely know."

    ]


    for suggestion in suggestions:

        st.write(
            f"• {suggestion}"
        )


    # =====================================================
    # EXTRACTED RESUME TEXT
    # =====================================================

    with st.expander(
        "📄 View Cleaned Resume Text"
    ):

        st.write(
            "### Resume Content"
        )

        st.text_area(
            "Extracted Resume",
            resume_text,
            height=400
        )


    # =====================================================
    # DOWNLOAD REPORT
    # =====================================================

    report = f"""
SMART RESUME ANALYZER
=====================

RESUME ANALYSIS REPORT

ATS SCORE:
{ats_score}/100

JOB MATCH:
{job_match:.0f}%

SEMANTIC SCORE:
{semantic_score:.0f}%


MATCHING JOB SKILLS
-------------------
"""


    if matching_skills:

        for skill in matching_skills:

            report += f"✓ {skill}\n"

    else:

        report += "No matching skills found.\n"


    report += """

MISSING JOB SKILLS
------------------
"""


    if missing_skills:

        for skill in missing_skills:

            report += f"✗ {skill}\n"

    else:

        report += "No missing job skills.\n"


    report += """

SKILLS FOUND IN RESUME
----------------------
"""


    if resume_skills:

        report += ", ".join(
            resume_skills
        )

    else:

        report += "No skills detected."


    report += """

RESUME SECTIONS
---------------
"""


    for section in all_sections:

        if section in found_sections:

            report += f"✓ {section}\n"

        else:

            report += f"✗ {section}\n"


    report += """

RECOMMENDATIONS
---------------
"""


    if missing_skills:

        report += (
            "Consider learning or highlighting "
            "these skills if you genuinely know them:\n"
        )

        for skill in missing_skills:

            report += f"- {skill}\n"


    report += """
- Keep the resume ATS-friendly.
- Use strong action words.
- Add measurable achievements.
- Add GitHub and LinkedIn links if available.
- Only include skills you genuinely know.
"""


    # -----------------------------------------------------
    # DOWNLOAD BUTTON
    # -----------------------------------------------------

    st.subheader(
        "📥 Download Analysis Report"
    )


    st.download_button(
        label="📥 Download Resume Analysis Report",
        data=report,
        file_name="resume_analysis_report.txt",
        mime="text/plain"
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Smart Resume Analyzer • "
    "Python + Streamlit + NLP + Machine Learning"
)