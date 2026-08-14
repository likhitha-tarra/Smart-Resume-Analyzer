import streamlit as st
from pypdf import PdfReader
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
# CSS
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

SKILL_ALIASES = {

    # Programming
    "Python": ["python", "python3"],
    "Java": ["java"],
    "C++": ["c++", "cpp"],
    "C": ["c programming", "c language"],
    "C#": ["c#", "c sharp"],
    "JavaScript": ["javascript", "js"],
    "TypeScript": ["typescript", "ts"],
    "Go": ["golang"],
    "Rust": ["rust"],
    "Kotlin": ["kotlin"],
    "Swift": ["swift"],
    "PHP": ["php"],
    "Ruby": ["ruby"],
    "R": ["r programming", "r language"],

    # Web
    "HTML": ["html", "html5"],
    "CSS": ["css", "css3"],
    "React": ["react", "react.js", "reactjs"],
    "Angular": ["angular"],
    "Vue.js": ["vue", "vue.js", "vuejs"],
    "Next.js": ["next.js", "nextjs"],
    "Node.js": ["node.js", "nodejs", "node"],
    "Express.js": ["express.js", "expressjs"],
    "Django": ["django"],
    "Flask": ["flask"],
    "Spring Boot": ["spring boot"],
    "ASP.NET": ["asp.net", ".net"],

    # AI / ML
    "Machine Learning": [
        "machine learning",
        "ml"
    ],
    "Deep Learning": [
        "deep learning",
        "dl"
    ],
    "Artificial Intelligence": [
        "artificial intelligence",
        "ai"
    ],
    "Natural Language Processing": [
        "natural language processing",
        "nlp"
    ],
    "Computer Vision": [
        "computer vision",
        "cv"
    ],
    "Generative AI": [
        "generative ai",
        "genai",
        "gen ai"
    ],
    "Large Language Models": [
        "large language models",
        "llm",
        "llms"
    ],
    "TensorFlow": ["tensorflow"],
    "PyTorch": ["pytorch"],
    "Scikit-learn": ["scikit-learn", "sklearn"],
    "Keras": ["keras"],
    "OpenCV": ["opencv"],
    "Hugging Face": ["hugging face", "huggingface"],

    # Data
    "Data Science": ["data science"],
    "Data Analysis": ["data analysis"],
    "Data Analytics": ["data analytics"],
    "Data Visualization": ["data visualization"],
    "Data Structures": ["data structures", "dsa"],
    "Algorithms": ["algorithms"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "Matplotlib": ["matplotlib"],
    "Seaborn": ["seaborn"],
    "Power BI": ["power bi"],
    "Tableau": ["tableau"],
    "Excel": ["excel", "microsoft excel"],

    # Databases
    "SQL": ["sql"],
    "MySQL": ["mysql"],
    "PostgreSQL": ["postgresql", "postgres"],
    "MongoDB": ["mongodb", "mongo db"],
    "Oracle": ["oracle"],
    "SQLite": ["sqlite"],
    "Redis": ["redis"],
    "Firebase": ["firebase"],
    "Database Management": [
        "database management",
        "database management systems",
        "dbms"
    ],

    # Cloud
    "AWS": ["aws", "amazon web services"],
    "Microsoft Azure": ["azure", "microsoft azure"],
    "Google Cloud": [
        "google cloud",
        "gcp",
        "google cloud platform"
    ],
    "Cloud Computing": ["cloud computing"],

    # DevOps
    "Git": ["git"],
    "GitHub": ["github"],
    "GitLab": ["gitlab"],
    "Docker": ["docker"],
    "Kubernetes": ["kubernetes", "k8s"],
    "Jenkins": ["jenkins"],
    "CI/CD": ["ci/cd", "cicd", "continuous integration"],
    "Terraform": ["terraform"],
    "Ansible": ["ansible"],

    # APIs / Backend
    "REST API": [
        "rest api",
        "restful api",
        "rest services"
    ],
    "FastAPI": ["fastapi"],
    "GraphQL": ["graphql"],
    "Microservices": ["microservices", "microservices architecture"],

    # Big Data
    "Apache Spark": ["apache spark", "spark"],
    "Hadoop": ["hadoop"],
    "Kafka": ["kafka", "apache kafka"],
    "PySpark": ["pyspark"],
    "ETL": ["etl", "extract transform load"],

    # Mobile
    "Android": ["android"],
    "Android Studio": ["android studio"],
    "Flutter": ["flutter"],
    "React Native": ["react native"],
    "iOS": ["ios"],

    # Tools
    "Streamlit": ["streamlit"],
    "Jupyter": ["jupyter", "jupyter notebook"],
    "VS Code": ["vs code", "visual studio code"],
    "Postman": ["postman"],

    # Concepts
    "Object Oriented Programming": [
        "object oriented programming",
        "oops",
        "oop"
    ],
    "Operating Systems": [
        "operating systems",
        "operating system"
    ],
    "Computer Networks": [
        "computer networks",
        "networking"
    ],
    "System Design": ["system design"],
    "Software Development": [
        "software development",
        "software engineering"
    ],

    # Testing
    "Unit Testing": ["unit testing"],
    "Selenium": ["selenium"],
    "Jest": ["jest"],
    "PyTest": ["pytest", "py test"],

    # Other
    "NLP": ["nlp"],
    "Cybersecurity": [
        "cybersecurity",
        "cyber security"
    ],
    "Blockchain": ["blockchain"],
    "Solidity": ["solidity"],
    "Linux": ["linux"],
}


# =========================================================
# RESUME SECTIONS
# =========================================================

SECTION_KEYWORDS = {

    "Education": [
        "education",
        "academic",
        "degree",
        "b.tech",
        "btech",
        "bachelor"
    ],

    "Experience": [
        "experience",
        "internship",
        "intern",
        "work experience"
    ],

    "Skills": [
        "skills",
        "technical skills",
        "programming languages",
        "technologies"
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
# NORMALIZE TEXT
# =========================================================

def normalize_text(text):

    text = text.lower()

    text = text.replace(
        "\u00a0",
        " "
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# SKILL DETECTION
# =========================================================

def skill_exists(text, skill):

    text = normalize_text(text)

    skill = skill.lower()

    # C++
    if skill == "c++":
        return bool(
            re.search(
                r"(?<!\w)c\+\+(?!\w)",
                text
            )
        )

    # C#
    if skill == "c#":
        return bool(
            re.search(
                r"(?<!\w)c#(?!\w)",
                text
            )
        )

    # C
    if skill == "c programming":
        return bool(
            re.search(
                r"\bc\s+programming\b",
                text
            )
        )

    pattern = (
        r"(?<!\w)"
        + re.escape(skill)
        + r"(?!\w)"
    )

    return bool(
        re.search(
            pattern,
            text
        )
    )


# =========================================================
# FIND SKILLS IN TEXT
# =========================================================

def find_skills(text):

    found = []

    for skill, aliases in SKILL_ALIASES.items():

        for alias in aliases:

            if skill_exists(
                text,
                alias
            ):

                found.append(skill)

                break

    return found


# =========================================================
# FIND JOB SKILLS
# =========================================================

def find_job_skills(job_description):

    return find_skills(
        job_description
    )


# =========================================================
# FIND RESUME SECTIONS
# =========================================================

def find_sections(text):

    found = []

    text_lower = normalize_text(
        text
    )

    for section, keywords in SECTION_KEYWORDS.items():

        for keyword in keywords:

            if keyword.lower() in text_lower:

                found.append(section)

                break

    return found


# =========================================================
# PDF EXTRACTION
# =========================================================

def extract_pdf_text(uploaded_file):

    text = ""

    try:

        reader = PdfReader(
            uploaded_file
        )

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:

                text += (
                    page_text
                    + "\n"
                )

    except Exception:

        return ""

    return text


# =========================================================
# DOCX EXTRACTION
# =========================================================

def extract_docx_text(uploaded_file):

    if Document is None:

        return ""

    try:

        document = Document(
            uploaded_file
        )

        paragraphs = []

        for paragraph in document.paragraphs:

            paragraphs.append(
                paragraph.text
            )

        return "\n".join(
            paragraphs
        )

    except Exception:

        return ""


# =========================================================
# RESUME EXTRACTION
# =========================================================

def extract_resume_text(
    uploaded_file
):

    file_name = (
        uploaded_file.name.lower()
    )

    if file_name.endswith(
        ".pdf"
    ):

        return extract_pdf_text(
            uploaded_file
        )

    if file_name.endswith(
        ".docx"
    ):

        return extract_docx_text(
            uploaded_file
        )

    return ""


# =========================================================
# CLEAN TEXT
# =========================================================

def clean_text(text):

    text = text.replace(
        "\x00",
        " "
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# JOB MATCH
# =========================================================

def calculate_job_match(
    resume_skills,
    job_skills
):

    if not job_skills:

        return 0.0

    resume_set = {
        skill.lower()
        for skill in resume_skills
    }

    matched = [

        skill

        for skill in job_skills

        if skill.lower()
        in resume_set
    ]

    score = (
        len(matched)
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

        return round(
            min(
                float(similarity * 100),
                100.0
            ),
            1
        )

    except Exception:

        return 0.0


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
        min(
            round(total),
            100
        )
    )


# =========================================================
# UI
# =========================================================

st.markdown(
    '<div class="section-title">📥 Resume & Job Details</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload your Resume",
    type=[
        "pdf",
        "docx"
    ]
)

job_description = st.text_area(
    "📝 Paste Job Description",
    height=220,
    placeholder=(
        "Paste the complete job description here..."
    )
)

analyze = st.button(
    "🚀 Analyze Resume",
    use_container_width=True
)


# =========================================================
# ANALYSIS
# =========================================================

if analyze:

    if uploaded_file is None:

        st.error(
            "Please upload your resume first."
        )

        st.stop()

    if not job_description.strip():

        st.error(
            "Please paste a job description."
        )

        st.stop()


    # =====================================================
    # EXTRACT
    # =====================================================

    resume_text = extract_resume_text(
        uploaded_file
    )

    resume_text = clean_text(
        resume_text
    )

    job_description = clean_text(
        job_description
    )

    if not resume_text:

        st.error(
            "Could not extract text from your resume."
        )

        st.stop()


    # =====================================================
    # DETECT SKILLS
    # =====================================================

    resume_skills = find_skills(
        resume_text
    )

    job_skills = find_job_skills(
        job_description
    )


    # =====================================================
    # SECTIONS
    # =====================================================

    found_sections = find_sections(
        resume_text
    )


    # =====================================================
    # MATCHING
    # =====================================================

    resume_skill_set = {
        skill.lower()
        for skill in resume_skills
    }

    matching_skills = [

        skill

        for skill in job_skills

        if skill.lower()
        in resume_skill_set
    ]


    missing_skills = [

        skill

        for skill in job_skills

        if skill.lower()
        not in resume_skill_set
    ]


    # =====================================================
    # SCORES
    # =====================================================

    ats_score = calculate_ats_score(
        resume_skills,
        found_sections,
        resume_text
    )

    job_match = calculate_job_match(
        resume_skills,
        job_skills
    )

    semantic_score = calculate_semantic_similarity(
        resume_text,
        job_description
    )


    # =====================================================
    # DASHBOARD
    # =====================================================

    st.divider()

    st.header(
        "📊 Resume Analysis Dashboard"
    )

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
    # PROGRESS
    # =====================================================

    st.write("### ATS Score")

    st.progress(
        ats_score / 100
    )

    st.write("### Job Match")

    st.progress(
        job_match / 100
    )

    st.write(
        "### Semantic Similarity"
    )

    st.progress(
        semantic_score / 100
    )


    # =====================================================
    # DETECTED JOB SKILLS
    # =====================================================

    st.subheader(
        "🎯 Skills Detected from Job Description"
    )

    if job_skills:

        st.write(
            " • ".join(
                job_skills
            )
        )

    else:

        st.warning(
            "No known technical skills were detected "
            "from this job description."
        )

        st.info(
            "Try pasting the complete job description "
            "including the Requirements or Skills section."
        )


    # =====================================================
    # MATCHING
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
            "No matching skills were detected."
        )


    # =====================================================
    # MISSING
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

        if job_skills:

            st.success(
                "Excellent! All detected job skills "
                "were found in your resume."
            )


    # =====================================================
    # RESUME SKILLS
    # =====================================================

    st.subheader(
        "🛠️ Skills Found in Resume"
    )

    if resume_skills:

        st.write(
            " • ".join(
                resume_skills
            )
        )

    else:

        st.warning(
            "No known skills detected in the resume."
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

    elif len(resume_skills) >= 4:

        st.write(
            "✅ Good technical skills coverage."
        )

    else:

        st.write(
            "⚠️ Basic technical skills coverage."
        )


    if matching_skills:

        st.write(
            f"✅ Resume matches "
            f"{len(matching_skills)} "
            f"detected job skill(s)."
        )


    # =====================================================
    # IMPROVEMENT
    # =====================================================

    st.subheader(
        "⚠️ Areas to Improve"
    )

    if missing_skills:

        st.write(
            "Job-relevant skills not detected "
            "in your resume:"
        )

        for skill in missing_skills:

            st.write(
                f"• {skill}"
            )

        st.caption(
            "Only add skills that you genuinely "
            "know or have experience with."
        )


    if "Projects" not in found_sections:

        st.write(
            "• Add 1–2 relevant projects."
        )


    if "Certifications" not in found_sections:

        st.write(
            "• Add relevant certifications if available."
        )


    # =====================================================
    # IMPROVEMENT PLAN
    # =====================================================

    st.subheader(
        "🤖 Resume Improvement Plan"
    )

    st.write(
        "### 🥇 Priority 1 — Job-Relevant Skills"
    )

    if missing_skills:

        for skill in missing_skills:

            st.write(
                f"🔹 {skill}"
            )

    elif job_skills:

        st.success(
            "All detected job skills are present."
        )

    else:

        st.info(
            "No technical skills were detected "
            "from the job description."
        )


    st.write(
        "### 🥈 Priority 2 — Projects"
    )

    if "Projects" in found_sections:

        st.write(
            "✅ Projects section is present."
        )

        st.write(
            "Highlight projects that use "
            "job-relevant technologies."
        )

    else:

        st.write(
            "⚠️ Add relevant projects."
        )


    st.write(
        "### 🥉 Priority 3 — Keywords"
    )

    if job_skills:

        st.write(
            "Use relevant keywords naturally "
            "in your resume where applicable:"
        )

        st.write(
            " • ".join(
                job_skills
            )
        )


    st.write(
        "### 📈 Priority 4 — Achievements"
    )

    st.write(
        "Use measurable results whenever possible."
    )

    st.write(
        "**Weak:** Developed a Python project."
    )

    st.write(
        "**Better:** Developed a Python application "
        "that reduced processing time by 30%."
    )


    # =====================================================
    # OVERALL MATCH
    # =====================================================

    st.write(
        "### 🎯 Overall Job Match"
    )

    if not job_skills:

        st.info(
            "Job Match cannot be meaningfully calculated "
            "because no known technical skills were detected."
        )

    elif job_match >= 75:

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
            f"Consider customizing your resume "
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

        "Add measurable achievements.",

        "Keep skills relevant to the target role.",

        "Add GitHub and LinkedIn links if available.",

        "Only include skills you genuinely know."

    ]

    for suggestion in suggestions:

        st.write(
            f"• {suggestion}"
        )


    # =====================================================
    # RESUME TEXT
    # =====================================================

    with st.expander(
        "📄 View Extracted Resume Text"
    ):

        st.text_area(
            "Resume Content",
            resume_text,
            height=400
        )


    # =====================================================
    # DOWNLOAD REPORT
    # =====================================================

    report = f"""
SMART RESUME ANALYZER
=====================

ATS SCORE:
{ats_score}/100

JOB MATCH:
{job_match:.0f}%

SEMANTIC SCORE:
{semantic_score:.0f}%


SKILLS DETECTED FROM JOB
------------------------
"""

    if job_skills:

        for skill in job_skills:

            report += (
                f"✓ {skill}\n"
            )

    else:

        report += (
            "No known technical skills detected.\n"
        )


    report += """

MATCHING SKILLS
---------------
"""

    if matching_skills:

        for skill in matching_skills:

            report += (
                f"✓ {skill}\n"
            )

    else:

        report += (
            "No matching skills found.\n"
        )


    report += """

MISSING SKILLS
--------------
"""

    if missing_skills:

        for skill in missing_skills:

            report += (
                f"✗ {skill}\n"
            )

    else:

        report += (
            "No missing detected skills.\n"
        )


    report += """

SKILLS FOUND IN RESUME
----------------------
"""

    if resume_skills:

        report += ", ".join(
            resume_skills
        )

    else:

        report += (
            "No known skills detected."
        )


    report += """

RESUME SECTIONS
---------------
"""

    for section in all_sections:

        if section in found_sections:

            report += (
                f"✓ {section}\n"
            )

        else:

            report += (
                f"✗ {section}\n"
            )


    report += """

RECOMMENDATIONS
---------------
"""

    for skill in missing_skills:

        report += (
            f"- Consider learning or "
            f"highlighting {skill} if you "
            f"genuinely know it.\n"
        )

    report += """
- Keep the resume ATS-friendly.
- Use strong action words.
- Add measurable achievements.
- Add GitHub and LinkedIn links if available.
- Only include skills you genuinely know.
"""


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