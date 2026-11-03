"""Generate a comprehensive, professionally styled PDF documentation for Project NLP - LASER3-CO."""
import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable, Preformatted
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and print total page count."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))
        
        # Header on later pages
        if self._pageNumber > 1:
            self.drawString(54, 755, "Project NLP — Bengali-Hindi Pilot & LASER3-CO Implementation Guide")
            self.setStrokeColor(colors.HexColor("#D0D7DE"))
            self.setLineWidth(0.5)
            self.line(54, 748, 558, 748)
            
        # Footer on all pages
        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(558, 36, page_str)
        self.drawString(54, 36, "Confidential — Natural Language Processing Research Group")
        self.setStrokeColor(colors.HexColor("#D0D7DE"))
        self.setLineWidth(0.5)
        self.line(54, 48, 558, 48)
        self.restoreState()


def build_pdf(filename="LASER3_CO_Project_Guide.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    # Custom Palette
    c_primary = colors.HexColor("#1A365D")   # Deep navy
    c_secondary = colors.HexColor("#2B6CB0") # Medium blue
    c_accent = colors.HexColor("#9B1B30")    # Crimson red accent
    c_dark = colors.HexColor("#2D3748")      # Text body
    c_light_bg = colors.HexColor("#F7FAFC")  # Table / box background
    c_border = colors.HexColor("#E2E8F0")

    # Typography styles
    styles.add(ParagraphStyle('DocTitle', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=22, leading=26, textColor=c_primary, spaceAfter=6))
    styles.add(ParagraphStyle('DocSub', parent=styles['Normal'], fontName='Helvetica', fontSize=12, leading=16, textColor=c_secondary, spaceAfter=14))
    styles.add(ParagraphStyle('MetaBanner', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=8.5, leading=12, textColor=colors.HexColor("#4A5568")))
    
    styles.add(ParagraphStyle('H1', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=14, leading=18, textColor=c_primary, spaceBefore=14, spaceAfter=8, keepWithNext=True))
    styles.add(ParagraphStyle('H2', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=11, leading=15, textColor=c_secondary, spaceBefore=10, spaceAfter=4, keepWithNext=True))
    styles.add(ParagraphStyle('H3', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=9.5, leading=13, textColor=c_accent, spaceBefore=6, spaceAfter=3, keepWithNext=True))
    
    styles.add(ParagraphStyle('Body', parent=styles['Normal'], fontName='Helvetica', fontSize=9, leading=13, textColor=c_dark, spaceAfter=6))
    styles.add(ParagraphStyle('BodyBold', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=9, leading=13, textColor=c_dark, spaceAfter=6))
    styles.add(ParagraphStyle('BulletText', parent=styles['Normal'], fontName='Helvetica', fontSize=8.5, leading=12, textColor=c_dark, spaceAfter=3))
    
    styles.add(ParagraphStyle('CodeBlock', parent=styles['Normal'], fontName='Courier', fontSize=7.5, leading=10, textColor=colors.HexColor("#1A202C")))
    styles.add(ParagraphStyle('TableCell', parent=styles['Normal'], fontName='Helvetica', fontSize=8, leading=11, textColor=c_dark))
    styles.add(ParagraphStyle('TableCellBold', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=8, leading=11, textColor=c_primary))
    styles.add(ParagraphStyle('TableHead', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=8, leading=11, textColor=colors.white))

    story = []

    # Title block
    story.append(Paragraph("Topic-Aware Multilingual Representation & LASER3-CO", styles['DocTitle']))
    story.append(Paragraph("Technical Specification, Mathematical Formulation & Complete Codebase Architecture", styles['DocSub']))
    
    meta_text = "<b>Project:</b> Project NLP 2 (Bengali–Hindi Pilot & Kangri Transfer) &nbsp;|&nbsp; <b>Server:</b> him2 (SLURM) &nbsp;|&nbsp; <b>Date:</b> November 2026"
    story.append(Paragraph(meta_text, styles['MetaBanner']))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_primary, spaceBefore=2, spaceAfter=12))

    # Section 1
    story.append(Paragraph("1. Executive Summary & Objective", styles['H1']))
    p1 = (
        "The objective of this project is to build sentence representations for low-resource languages "
        "(specifically Kangri, with Bengali–Hindi as our validation pilot) where sentences on the same topic "
        "cluster tightly together in vector space, while parallel translations remain reciprocal nearest neighbors. "
        "The project initially evaluated two dual-encoder models (<b>LaBSE</b> and <b>Multilingual-E5</b>) trained "
        "with contrastive translation loss and supervised topic loss (SupCon). "
        "This extension integrates <b>LASER3-CO</b> (Tan et al., EACL 2023) into the same rigorous training, "
        "evaluation, bootstrap statistical testing, and Streamlit dashboard ecosystem."
    )
    story.append(Paragraph(p1, styles['Body']))

    ws_box = [
        [Paragraph("<b>Dual Project Workspace Setup on Server him2:</b>", styles['TableCellBold'])],
        [Paragraph(
            "• <b>Everything Else (<code>/raid/home/loitongbam/Project_NLP</code>):</b> Contains baseline LaBSE & E5 checkpoints, training logs, evaluation results, Week 1 data cleaning, and Week 2 Kangri development clusters.<br/>"
            "• <b>LASER3-CO Related (<code>/raid/home/loitongbam/Project_NLP_2</code>):</b> Dedicated environment for fairseq / laser_encoders, LASER3-CO contrastive training, student checkpoints, evaluation results, and the running dashboard job.<br/>"
            "• <b>Automated Dashboard Cross-Resolution:</b> The dashboard bridges both directories in real time. Week 1 & Week 2 read from <code>Project_NLP</code>, LASER3-CO reads from <code>Project_NLP_2</code> with benchmark rows from <code>Project_NLP</code>, and the Bengali–Hindi pilot unifies all models into a single view.",
            styles['TableCell']
        )]
    ]
    t_ws = Table(ws_box, colWidths=[500])
    t_ws.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), c_light_bg),
        ('BOX', (0, 0), (-1, -1), 1, c_secondary),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(t_ws)
    story.append(Spacer(1, 8))

    # Section 2
    story.append(Paragraph("2. Methodological & Mathematical Framework", styles['H1']))
    story.append(Paragraph("2.1 Dual-Encoder Fine-Tuning (LaBSE & E5)", styles['H2']))
    p2 = (
        "The baseline pilot uses bi-encoders fine-tuned with two loss terms:<br/>"
        "• <b>Translation Loss:</b> In-batch contrastive alignment pulling Bengali sentences to their Hindi counterparts.<br/>"
        "• <b>Topic Loss (SupCon - Khosla et al., 2020):</b> Pulls together all sentences in the batch sharing the same "
        "topic label (regardless of language) while repelling sentences of distinct topics."
    )
    story.append(Paragraph(p2, styles['Body']))

    story.append(Paragraph("2.2 LASER3-CO Contrastive Distillation (Tan et al., 2023)", styles['H2']))
    p3 = (
        "LASER3 features modular single-language encoders mapping into a shared multilingual space defined by LASER2. "
        "<b>LASER3-CO</b> continues training a student encoder via contrastive learning against a permanently frozen teacher:<br/>"
        "• <b>Teacher Encoder:</b> Hindi encoder (<code>hin_Deva</code>), weights permanently frozen.<br/>"
        "• <b>Student Encoder:</b> Bengali encoder (<code>ben_Beng</code>), updated with autograd gradients.<br/>"
        "• <b>InfoNCE Loss with Negative Queue:</b>"
    )
    story.append(Paragraph(p3, styles['Body']))

    loss_box = [
        [Paragraph("<b>InfoNCE Queue Formulation:</b>", styles['TableCellBold'])],
        [Paragraph(
            "&nbsp;&nbsp;&nbsp;&nbsp;<b>L<sub>CO</sub> = - log [ exp(q · k<sub>+</sub> / τ) / ( exp(q · k<sub>+</sub> / τ) + Σ<sub>j</sub> exp(q · k<sub>j</sub> / τ) ) ]</b><br/>"
            "where <b>q</b> = student embedding of Bengali sentence, <b>k<sub>+</sub></b> = teacher embedding of true Hindi translation,<br/>"
            "and negatives <b>k<sub>j</sub></b> come from a FIFO memory queue of size <b>4096</b> plus batch target translations.<br/>"
            "Temperature <b>τ = 0.05</b>. No momentum encoder is required (unlike MoCo) because teacher is frozen.",
            styles['TableCell']
        )]
    ]
    t_loss = Table(loss_box, colWidths=[500])
    t_loss.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), c_light_bg),
        ('BOX', (0, 0), (-1, -1), 1, c_secondary),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(t_loss)
    story.append(PageBreak())

    story.append(Paragraph("2.3 Evaluated Model Variants", styles['H2']))
    variants_table_data = [
        [Paragraph("Variant Name", styles['TableHead']), Paragraph("Type", styles['TableHead']), Paragraph("Description & Loss Formulation", styles['TableHead'])],
        [Paragraph("<code>laser</code>", styles['TableCellBold']), Paragraph("Untrained Baseline", styles['TableCell']), Paragraph("Untouched official LASER3 encoders for Bengali and Hindi; establishes the zero-shot baseline.", styles['TableCell'])],
        [Paragraph("<code>laser_co</code>", styles['TableCellBold']), Paragraph("LASER3-CO", styles['TableCell']), Paragraph("Standard queue contrastive distillation (Q=4096, batch=32, τ=0.05, lr=1e-5, shuffled batches).", styles['TableCell'])],
        [Paragraph("<code>laser_co_filter</code>", styles['TableCellBold']), Paragraph("LASER3-CO-Filter", styles['TableCell']), Paragraph("Length-sorted batches (harder negatives) + cosine filter dropping negatives with cos(k<sub>i</sub>, k<sub>+</sub>) ≥ 0.9.", styles['TableCell'])],
        [Paragraph("<code>laser_co_topic</code>", styles['TableCellBold']), Paragraph("LASER3-CO + Topic", styles['TableCell']), Paragraph("LASER3-CO joint objective adding Supervised Contrastive topic loss (weight 1.0) to cluster topics.", styles['TableCell'])],
    ]
    t_var = Table(variants_table_data, colWidths=[95, 105, 300])
    t_var.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_primary),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_light_bg]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_var)

    story.append(Spacer(1, 10))

    # Section 3
    story.append(Paragraph("3. Complete Codebase & File Structure", styles['H1']))
    p_files = (
        "The repository follows a clean, modular layout where data preprocessing, training, evaluation, "
        "dashboard visualization, and SLURM workload batch jobs are decoupled. Below is the file tree and detailed "
        "breakdown of responsibilities."
    )
    story.append(Paragraph(p_files, styles['Body']))

    tree_str = (
        "Project_NLP_2/\n"
        "+-- configs/\n"
        "|   \\-- pilot_bn.yaml              # Config: paths, datasets, hyperparams, compare pairs\n"
        "+-- src/\n"
        "|   +-- laser/\n"
        "|   |   +-- check_laser.py         # Inspects encoders & checks language support\n"
        "|   |   \\-- train_laser_co.py      # LASER3-CO contrastive training loop (autograd + queue)\n"
        "|   +-- eval/\n"
        "|   |   +-- evaluate.py            # Primary scoring: KNN topic acc, translation error\n"
        "|   |   +-- evaluate_laser.py      # Evaluates baseline & trained student checkpoints\n"
        "|   |   \\-- aggregate.py           # Multi-seed pooling (42, 43, 44) & paired bootstrap\n"
        "|   +-- train/\n"
        "|   |   \\-- train.py               # Bi-encoder training for LaBSE & Multilingual-E5\n"
        "|   \\-- utils/\n"
        "|       \\-- common.py              # Config loading, seeding, path resolution, JSON\n"
        "+-- dashboard/\n"
        "|   +-- app.py                     # Streamlit multi-page dashboard router & sidebar\n"
        "|   +-- pilot/\n"
        "|   |   +-- bengali_hindi.py       # Main pilot dashboard: results, comparisons, curves\n"
        "|   |   \\-- laser3_co.py           # Dedicated LASER3-CO dashboard & benchmarks\n"
        "|   \\-- weeks/                     # Weekly progress logs for Kangri annotation\n"
        "+-- slurm/\n"
        "|   +-- install_laser.sbatch       # Environment installer (fairseq, laser_encoders)\n"
        "|   +-- check_laser.sbatch         # Verification script for GPU module architecture\n"
        "|   +-- train_laser_co.sbatch      # Single LASER3-CO training job\n"
        "|   +-- train_laser_seq.sbatch     # Runs all 9 LASER training runs in ONE job\n"
        "|   +-- evaluate_laser.sbatch      # Single checkpoint evaluation\n"
        "|   +-- evaluate_laser_all.sbatch  # Scores baseline + all 9 student checkpoints\n"
        "|   +-- aggregate.sbatch           # Aggregates multi-seed results & statistical tests\n"
        "|   \\-- dashboard.sbatch           # Launches Streamlit web server on GPU node\n"
        "\\-- outputs/\n"
        "    +-- checkpoints/               # Saved student model weights (best_student.pt)\n"
        "    +-- results/pilot_bn/eval/     # results.csv, seeds_summary.csv, seeds_compare.csv\n"
        "    \\-- logs/                      # SLURM stdout and stderr log files"
    )
    t_tree = Table([[Preformatted(tree_str, styles['CodeBlock'])]], colWidths=[500])
    t_tree.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#EDF2F7")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E0")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_tree)
    story.append(PageBreak())

    story.append(Paragraph("3.1 Key File Responsibilities", styles['H2']))
    file_desc_data = [
        [Paragraph("File / Module", styles['TableHead']), Paragraph("Location", styles['TableHead']), Paragraph("Functionality & Key Implementation Details", styles['TableHead'])],
        [
            Paragraph("<code>pilot_bn.yaml</code>", styles['TableCellBold']),
            Paragraph("<code>configs/</code>", styles['TableCell']),
            Paragraph("Defines datasets (MASSIVE, SIB-200, IN22), hyperparameters (batch size, queue size 4096, temp 0.05, lr 1e-5), evaluation pairs for paired testing, and aggregate runs.", styles['TableCell'])
        ],
        [
            Paragraph("<code>train_laser_co.py</code>", styles['TableCellBold']),
            Paragraph("<code>src/laser/</code>", styles['TableCell']),
            Paragraph("Implements <code>LaserSide</code> (autograd forward pass on Transformer encoder), FIFO negative <code>Queue</code>, pre-filtering (σ=0.9), and checkpoint selection on dev.", styles['TableCell'])
        ],
        [
            Paragraph("<code>evaluate_laser.py</code>", styles['TableCellBold']),
            Paragraph("<code>src/eval/</code>", styles['TableCell']),
            Paragraph("Loads baseline or trained student checkpoint into Bengali encoder, executes evaluation over test sets using <code>score_set</code>, and appends to <code>results.csv</code>.", styles['TableCell'])
        ],
        [
            Paragraph("<code>aggregate.py</code>", styles['TableCellBold']),
            Paragraph("<code>src/eval/</code>", styles['TableCell']),
            Paragraph("Aggregates 3 repeat seeds (42, 43, 44), calculates mean/sd/min/max, and conducts group-paired bootstrap testing (10,000 resamples) for statistical significance.", styles['TableCell'])
        ],
        [
            Paragraph("<code>app.py</code>", styles['TableCellBold']),
            Paragraph("<code>dashboard/</code>", styles['TableCell']),
            Paragraph("Main Streamlit application router. Offers dedicated sidebar navigation to switch between general <b>Bengali–Hindi</b> pilot and dedicated <b>LASER3-CO</b> section.", styles['TableCell'])
        ],
        [
            Paragraph("<code>laser3_co.py</code>", styles['TableCellBold']),
            Paragraph("<code>dashboard/pilot/</code>", styles['TableCell']),
            Paragraph("Full 7-tab LASER dashboard: (1) Setup, (2) Results vs LaBSE/E5, (3) Statistical Significance, (4) Error Analysis, (5) Training Curves, (6) 2D Vector Space (t-SNE maps & star query), (7) Try It Live (interactive encoding, SentencePiece tokens, 1024-D vector, & 5-NN voting).", styles['TableCell'])
        ],
        [
            Paragraph("<code>laser_utils.py</code>", styles['TableCellBold']),
            Paragraph("<code>dashboard/pilot/</code>", styles['TableCell']),
            Paragraph("Shared pipeline utilities: cached GPU loader, PyTorch 2.6+ unpickler compatibility, student checkpoint injection, SentencePiece tokenization, and language detection.", styles['TableCell'])
        ],
    ]
    t_fdesc = Table(file_desc_data, colWidths=[90, 80, 330])
    t_fdesc.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_primary),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_light_bg]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_fdesc)

    story.append(Spacer(1, 10))

    # Section 4
    story.append(Paragraph("4. Evaluation Metrics & Verification Protocol", styles['H1']))
    p_eval = (
        "All models across LaBSE, E5, and LASER families are scored using the exact same evaluation engine "
        "(<code>src/eval/evaluate.py:score_set</code>) to ensure strict, unbiased comparability."
    )
    story.append(Paragraph(p_eval, styles['Body']))

    eval_items = [
        "<b>1. Topic Accuracy (Bengali side, %):</b> 5 nearest neighbor sentences vote on the topic label. For IN22-Conv, sentences from the same conversation are masked out.",
        "<b>2. Translation Error (Bengali→Hindi, %):</b> Percentage of Bengali sentences whose closest Hindi sentence (by cosine similarity) is NOT its true translation. Lower is better.",
        "<b>3. Topic Accuracy across Languages (%):</b> Bengali test sentences queried against Hindi reference sentences.",
        "<b>4. Cluster Agreement (NMI / Normalized Mutual Information):</b> Unsupervised k-means alignment with gold labels.",
        "<b>5. 95% Bootstrap Confidence Intervals:</b> 1,000 resamples over test sentences; paired bootstrap differences across 3 repeat training seeds (42, 43, 44) to guarantee robustness."
    ]
    for it in eval_items:
        story.append(Paragraph(f"• {it}", styles['BulletText']))
    story.append(PageBreak())

    # Section 5
    story.append(Paragraph("5. Step-by-Step Execution Guide on SLURM (Server him2)", styles['H1']))
    story.append(Paragraph("All operations run strictly on GPU compute nodes via sbatch from <code>/raid/home/loitongbam/Project_NLP_2</code>.", styles['Body']))

    steps_data = [
        [Paragraph("Step", styles['TableHead']), Paragraph("SLURM Command", styles['TableHead']), Paragraph("Purpose & Expected Outcome", styles['TableHead'])],
        [
            Paragraph("<b>Step 1</b>", styles['TableCellBold']),
            Paragraph("<code>sbatch slurm/check_laser.sbatch</code>", styles['TableCell']),
            Paragraph("Inspects Bengali & Hindi encoders, downloads model files, confirms PyTorch 2.6+ unpickling compatibility.", styles['TableCell'])
        ],
        [
            Paragraph("<b>Step 2</b>", styles['TableCellBold']),
            Paragraph("<code>sbatch slurm/evaluate_laser.sbatch configs/pilot_bn.yaml</code>", styles['TableCell']),
            Paragraph("Evaluates the untouched LASER baseline across SIB-200, IN22-Gen, and IN22-Conv. Populates <code>results.csv</code>.", styles['TableCell'])
        ],
        [
            Paragraph("<b>Step 3</b>", styles['TableCellBold']),
            Paragraph("<code>sbatch slurm/train_laser_co.sbatch configs/pilot_bn.yaml co 42 --max_steps 20</code>", styles['TableCell']),
            Paragraph("Runs 20-step smoke test; verifies loss decay, autograd propagation, and checkpoint saving in <code>outputs/checkpoints/</code>.", styles['TableCell'])
        ],
        [
            Paragraph("<b>Step 4</b>", styles['TableCellBold']),
            Paragraph("<code>sbatch slurm/train_laser_seq.sbatch configs/pilot_bn.yaml \\\n  co:42 co:43 co:44 \\\n  co_filter:42 co_filter:43 co_filter:44 \\\n  co_topic:42 co_topic:43 co_topic:44</code>", styles['TableCell']),
            Paragraph("Executes all 9 full training runs sequentially in ONE batch job to respect cluster user job limits.", styles['TableCell'])
        ],
        [
            Paragraph("<b>Step 5</b>", styles['TableCellBold']),
            Paragraph("<code>sbatch slurm/evaluate_laser_all.sbatch configs/pilot_bn.yaml</code>", styles['TableCell']),
            Paragraph("Evaluates the baseline and all 9 trained student checkpoints. Logs metrics to <code>results.csv</code> and per-sentence TSVs.", styles['TableCell'])
        ],
        [
            Paragraph("<b>Step 6</b>", styles['TableCellBold']),
            Paragraph("<code>sbatch slurm/aggregate.sbatch configs/pilot_bn.yaml</code>", styles['TableCell']),
            Paragraph("Combines repeat seeds, executes paired bootstrap tests (LASER vs variants), writes <code>seeds_compare.csv</code>.", styles['TableCell'])
        ],
        [
            Paragraph("<b>Step 7</b>", styles['TableCellBold']),
            Paragraph("<code>sbatch slurm/dashboard.sbatch [port]</code>", styles['TableCell']),
            Paragraph("Launches Streamlit dashboard on compute node. Access via SSH tunnel to inspect live results, curves, and error analysis.", styles['TableCell'])
        ],
    ]
    t_steps = Table(steps_data, colWidths=[45, 230, 225])
    t_steps.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_primary),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_light_bg]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_steps)
    story.append(Spacer(1, 10))

    # Summary box
    story.append(Paragraph("6. Key Findings Expected & Comparison with Benchmarks", styles['H1']))
    p_summary = (
        "• <b>Translation Error:</b> LASER3-CO is specifically optimized for translation matching. A significant drop in translation error "
        "(e.g., from ~10% down to &lt;2% on SIB-200) is expected for <code>laser_co</code> and <code>laser_co_filter</code>.<br/>"
        "• <b>Topic Accuracy:</b> Pure contrastive translation learning is not expected to significantly move topic accuracy. Only the "
        "<code>laser_co_topic</code> variant incorporates supervised topic clustering loss (SupCon) and is expected to increase topic accuracy.<br/>"
        "• <b>Benchmark Comparison:</b> When viewing the final table in the dashboard, LASER models appear directly alongside LaBSE and "
        "Multilingual-E5, providing an end-to-end comparison of all three encoder architectures under identical test conditions."
    )
    story.append(Paragraph(p_summary, styles['Body']))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated PDF: {filename}")


if __name__ == "__main__":
    build_pdf("LASER3_CO_Project_Guide.pdf")
