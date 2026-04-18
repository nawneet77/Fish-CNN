"""
Generate Final Year Project Presentation PPT.
Smart Aquarium Fish Health Monitoring System
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import os

# ── Color Palette — warm earthy tones ─────────────────────────
BG_DARK   = RGBColor(0xF4, 0xEF, 0xE7)   # warm cream
BG_CARD   = RGBColor(0xFF, 0xFF, 0xFF)   # white cards
ACCENT    = RGBColor(0x1B, 0x4D, 0x3E)   # deep forest green
ACCENT2   = RGBColor(0x6B, 0x4E, 0x30)   # warm brown
WHITE     = RGBColor(0x2B, 0x2B, 0x2B)   # dark charcoal (text)
LIGHT     = RGBColor(0x4A, 0x4A, 0x4A)   # medium gray (body text)
GRAY      = RGBColor(0x7A, 0x73, 0x6B)   # warm gray
ORANGE    = RGBColor(0xC4, 0x70, 0x4B)   # terracotta
RED       = RGBColor(0xA3, 0x33, 0x33)   # muted brick red
GREEN     = RGBColor(0x1B, 0x4D, 0x3E)   # same forest green
YELLOW    = RGBColor(0xB8, 0x86, 0x0B)   # dark goldenrod

W = Inches(13.333)
H = Inches(7.5)


def set_slide_bg(slide, color):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_shape_fill(slide, left, top, width, height, color, alpha=None):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    if color == BG_CARD:
        # Subtle border for white cards on cream background
        shape.line.color.rgb = RGBColor(0xE0, 0xDA, 0xD2)
        shape.line.width = Pt(0.75)
    else:
        shape.line.fill.background()
    if alpha is not None:
        shape.fill.fore_color.brightness = 0
    return shape


def add_text_box(slide, left, top, width, height, text, font_size=18, color=WHITE,
                 bold=False, alignment=PP_ALIGN.LEFT, font_name="Calibri"):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = font_name
    p.alignment = alignment
    return txBox


def add_bullet_list(slide, left, top, width, height, items, font_size=14,
                    color=LIGHT, bullet_color=ACCENT, spacing=Pt(6)):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = item
        p.font.size = Pt(font_size)
        p.font.color.rgb = color
        p.font.name = "Calibri"
        p.space_after = spacing
        p.level = 0
    return txBox


def add_metric_card(slide, left, top, width, value, label, color=ACCENT):
    card = add_shape_fill(slide, left, top, width, Inches(1.1), BG_CARD)
    card.shadow.inherit = False

    add_text_box(slide, left + Inches(0.15), top + Inches(0.1),
                 width - Inches(0.3), Inches(0.6), value,
                 font_size=28, color=color, bold=True, alignment=PP_ALIGN.CENTER)
    add_text_box(slide, left + Inches(0.15), top + Inches(0.65),
                 width - Inches(0.3), Inches(0.4), label,
                 font_size=11, color=GRAY, alignment=PP_ALIGN.CENTER)


def add_accent_line(slide, left, top, width, color=ACCENT):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, Pt(3))
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()


def add_section_header(slide, top, text, color=ACCENT):
    add_text_box(slide, Inches(0.8), top, Inches(11), Inches(0.5), text,
                 font_size=13, color=color, bold=True)
    add_accent_line(slide, Inches(0.8), top + Inches(0.35), Inches(2.5), color)


# ════════════════════════════════════════════════════════════════
#  SLIDE 1: TITLE
# ════════════════════════════════════════════════════════════════
def slide_title(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    set_slide_bg(slide, BG_DARK)

    # Decorative top bar
    add_shape_fill(slide, Inches(0), Inches(0), W, Inches(0.06), ACCENT)

    # Left accent stripe
    add_shape_fill(slide, Inches(0), Inches(0), Inches(0.08), H, ACCENT)

    # Title
    add_text_box(slide, Inches(1.2), Inches(1.4), Inches(11), Inches(1.0),
                 "Smart Aquarium Fish Health",
                 font_size=42, color=WHITE, bold=True)
    add_text_box(slide, Inches(1.2), Inches(2.1), Inches(11), Inches(1.0),
                 "Monitoring System",
                 font_size=42, color=ACCENT, bold=True)

    # Subtitle
    add_text_box(slide, Inches(1.2), Inches(3.2), Inches(10), Inches(0.6),
                 "Real-time AI-powered fish health detection with IoT aquarium automation",
                 font_size=18, color=LIGHT)

    # Accent line
    add_accent_line(slide, Inches(1.2), Inches(4.0), Inches(3), ACCENT)

    # Team info
    add_text_box(slide, Inches(1.2), Inches(4.4), Inches(5), Inches(0.4),
                 "Team Members", font_size=12, color=GRAY, bold=True)
    add_text_box(slide, Inches(1.2), Inches(4.75), Inches(5), Inches(0.5),
                 "Nawneet Kumar  |  Sanket Vaidya",
                 font_size=16, color=WHITE, bold=True)

    add_text_box(slide, Inches(1.2), Inches(5.4), Inches(5), Inches(0.4),
                 "Mentor", font_size=12, color=GRAY, bold=True)
    add_text_box(slide, Inches(1.2), Inches(5.75), Inches(5), Inches(0.5),
                 "Dr. Rishikant Rajdeepak",
                 font_size=16, color=ACCENT2, bold=True)

    # Date box
    add_text_box(slide, Inches(1.2), Inches(6.5), Inches(5), Inches(0.4),
                 "April 14, 2026  |  Final Year Project Presentation",
                 font_size=12, color=GRAY)

    # Right side — decorative cards
    add_metric_card(slide, Inches(8.5), Inches(4.2), Inches(2.0),
                    "92.5%", "Detection Accuracy", ACCENT)
    add_metric_card(slide, Inches(10.8), Inches(4.2), Inches(2.0),
                    "95.3%", "Precision", GREEN)
    add_metric_card(slide, Inches(8.5), Inches(5.6), Inches(2.0),
                    "Real-time", "30 FPS (GPU)", ORANGE)
    add_metric_card(slide, Inches(10.8), Inches(5.6), Inches(2.0),
                    "7 Sensors", "IoT Integration", ACCENT2)


# ════════════════════════════════════════════════════════════════
#  SLIDE 2: PROBLEM STATEMENT
# ════════════════════════════════════════════════════════════════
def slide_problem(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, BG_DARK)
    add_shape_fill(slide, Inches(0), Inches(0), W, Inches(0.06), ACCENT)

    add_text_box(slide, Inches(0.8), Inches(0.4), Inches(5), Inches(0.5),
                 "PROBLEM STATEMENT", font_size=11, color=ACCENT, bold=True)
    add_text_box(slide, Inches(0.8), Inches(0.8), Inches(11), Inches(0.8),
                 "Why do aquarium fish die silently?",
                 font_size=32, color=WHITE, bold=True)

    # Problem cards - left column
    problems = [
        ("Late Detection", "Fish disease symptoms are subtle. By the time owners notice\n"
         "color changes or lethargy, the disease has often progressed\n"
         "beyond treatment.", RED),
        ("Manual Monitoring", "Aquarium owners must manually check water parameters,\n"
         "observe fish behavior, and maintain feeding schedules.\n"
         "Human oversight is inconsistent and error-prone.", ORANGE),
        ("No Early Warning", "Behavioral changes (swimming patterns, isolation, surface\n"
         "gasping) precede visible symptoms by 4-5 days, but humans\n"
         "cannot track individual fish behavior 24/7.", YELLOW),
    ]

    y = Inches(1.9)
    for title, desc, color in problems:
        # Card background
        add_shape_fill(slide, Inches(0.8), y, Inches(6.5), Inches(1.4), BG_CARD)
        # Color indicator
        add_shape_fill(slide, Inches(0.8), y, Inches(0.06), Inches(1.4), color)
        # Title
        add_text_box(slide, Inches(1.1), y + Inches(0.1), Inches(6), Inches(0.35),
                     title, font_size=16, color=color, bold=True)
        # Description
        add_text_box(slide, Inches(1.1), y + Inches(0.45), Inches(6), Inches(0.9),
                     desc, font_size=12, color=LIGHT)
        y += Inches(1.6)

    # Solution summary - right side
    add_shape_fill(slide, Inches(7.8), Inches(1.9), Inches(5), Inches(4.8), BG_CARD)
    add_shape_fill(slide, Inches(7.8), Inches(1.9), Inches(5), Inches(0.06), GREEN)

    add_text_box(slide, Inches(8.1), Inches(2.1), Inches(4.5), Inches(0.4),
                 "OUR SOLUTION", font_size=13, color=GREEN, bold=True)

    solution_items = [
        "YOLOv8 CNN trained on fish tank footage for real-time detection (92.5% mAP50)",
        "Per-fish adaptive baselines using Welford's algorithm — each fish compared to its OWN normal",
        "Z-score anomaly detection across 9 health indicators (behavioral + visual)",
        "ESP32-based IoT system with 7 sensors for automated water quality monitoring",
        "Automated feeding, heating, lighting, and aeration based on sensor data and schedules",
        "Live web dashboard with health scores, alerts, and tuning controls",
    ]
    add_bullet_list(slide, Inches(8.1), Inches(2.6), Inches(4.4), Inches(3.8),
                    solution_items, font_size=12, color=LIGHT, spacing=Pt(10))


# ════════════════════════════════════════════════════════════════
#  SLIDE 3: METHODOLOGY — AI/Software
# ════════════════════════════════════════════════════════════════
def slide_methodology_ai(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, BG_DARK)
    add_shape_fill(slide, Inches(0), Inches(0), W, Inches(0.06), ACCENT)

    add_text_box(slide, Inches(0.8), Inches(0.4), Inches(5), Inches(0.5),
                 "METHODOLOGY", font_size=11, color=ACCENT, bold=True)
    add_text_box(slide, Inches(0.8), Inches(0.8), Inches(11), Inches(0.7),
                 "AI Pipeline — Detection to Alert",
                 font_size=28, color=WHITE, bold=True)

    # Pipeline stages
    stages = [
        ("1", "Detection", "YOLOv8n CNN", "Custom-trained on fish\ntank footage. 92.5%\nmAP50, 95.3% precision.", ACCENT),
        ("2", "Tracking", "Enhanced SORT", "Kalman filter + appearance\nRe-ID. Hungarian algorithm\nfor optimal matching.", ACCENT2),
        ("3", "Health Analysis", "Visual + Behavioral", "5 visual features (color,\ntexture, shape) + 7 behavior\nmetrics (speed, pattern).", ORANGE),
        ("4", "Adaptive Baseline", "Welford's Algorithm", "Per-fish running mean/std.\nO(1) memory. Ready after\n150 samples (~25 sec).", GREEN),
        ("5", "Anomaly Detection", "Z-Score + Consensus", "9-metric weighted consensus.\nZ > 2.0 = concern. 73%\nbehavioral, 27% visual.", YELLOW),
        ("6", "Alert Generation", "4-Stage Filter", "Baseline wait + track\nmaturity + sustained check\n+ 60s cooldown.", RED),
    ]

    x_start = Inches(0.5)
    card_w = Inches(1.95)
    gap = Inches(0.13)
    y_top = Inches(1.7)

    for i, (num, title, subtitle, desc, color) in enumerate(stages):
        x = x_start + i * (card_w + gap)

        # Card
        add_shape_fill(slide, x, y_top, card_w, Inches(3.3), BG_CARD)
        # Top color bar
        add_shape_fill(slide, x, y_top, card_w, Inches(0.05), color)

        # Number circle
        circle = slide.shapes.add_shape(MSO_SHAPE.OVAL,
                                        x + Inches(0.65), y_top + Inches(0.2),
                                        Inches(0.6), Inches(0.6))
        circle.fill.solid()
        circle.fill.fore_color.rgb = color
        circle.line.fill.background()
        tf = circle.text_frame
        tf.paragraphs[0].text = num
        tf.paragraphs[0].font.size = Pt(20)
        tf.paragraphs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        tf.paragraphs[0].font.bold = True
        tf.paragraphs[0].alignment = PP_ALIGN.CENTER
        tf.word_wrap = False

        # Title
        add_text_box(slide, x + Inches(0.1), y_top + Inches(0.95),
                     card_w - Inches(0.2), Inches(0.35),
                     title, font_size=14, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)
        # Subtitle
        add_text_box(slide, x + Inches(0.1), y_top + Inches(1.25),
                     card_w - Inches(0.2), Inches(0.3),
                     subtitle, font_size=10, color=color, alignment=PP_ALIGN.CENTER)
        # Description
        add_text_box(slide, x + Inches(0.1), y_top + Inches(1.7),
                     card_w - Inches(0.2), Inches(1.4),
                     desc, font_size=10, color=LIGHT, alignment=PP_ALIGN.CENTER)

    # Arrow connectors between cards
    for i in range(5):
        x = x_start + (i + 1) * (card_w + gap) - gap + Inches(0.01)
        arrow = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW,
                                       x, y_top + Inches(1.0),
                                       Inches(0.12), Inches(0.25))
        arrow.fill.solid()
        arrow.fill.fore_color.rgb = GRAY
        arrow.line.fill.background()

    # Bottom section — Key algorithms
    y_bottom = Inches(5.3)
    add_section_header(slide, y_bottom, "KEY ALGORITHMS & TECHNIQUES")

    techs = [
        ("Welford's Online Algorithm", "Numerically stable single-pass variance. O(1) memory per fish per metric."),
        ("Z-Score Anomaly Detection", "Compares each fish to its OWN baseline. Z = |current - mean| / std."),
        ("EMA Smoothing (alpha=0.15)", "Exponential moving average removes noise. Responds in ~7 seconds."),
        ("Hysteresis State Machine", "Dead-band gaps prevent status flickering. 4 states: Healthy/Concern/Warning/Critical."),
    ]

    x = Inches(0.8)
    for title, desc in techs:
        add_text_box(slide, x, y_bottom + Inches(0.5), Inches(2.8), Inches(0.3),
                     title, font_size=10, color=ACCENT, bold=True)
        add_text_box(slide, x, y_bottom + Inches(0.8), Inches(2.8), Inches(0.6),
                     desc, font_size=9, color=LIGHT)
        x += Inches(3.1)


# ════════════════════════════════════════════════════════════════
#  SLIDE 4: METHODOLOGY — Hardware/IoT
# ════════════════════════════════════════════════════════════════
def slide_methodology_hw(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, BG_DARK)
    add_shape_fill(slide, Inches(0), Inches(0), W, Inches(0.06), ACCENT2)

    add_text_box(slide, Inches(0.8), Inches(0.4), Inches(5), Inches(0.5),
                 "METHODOLOGY", font_size=11, color=ACCENT2, bold=True)
    add_text_box(slide, Inches(0.8), Inches(0.8), Inches(11), Inches(0.7),
                 "IoT Hardware — ESP32 Aquarium Automation",
                 font_size=28, color=WHITE, bold=True)

    # Left column — Sensors
    add_section_header(slide, Inches(1.6), "SENSORS (INPUT)", ACCENT2)

    sensors = [
        ("DS18B20", "Temperature", "Water temp in Celsius", "GPIO 4"),
        ("PH4502C", "pH Sensor", "Acidity/alkalinity (0-14)", "GPIO 34"),
        ("SEN0189", "Turbidity", "Water clarity (NTU)", "GPIO 35"),
        ("TDS V1.0", "TDS Meter", "Dissolved solids (ppm)", "GPIO 33"),
        ("DS3231", "RTC Clock", "Time & date keeping", "I2C"),
        ("TTP223", "Touch Sensors", "User input (LED control)", "GPIO 23, 5"),
        ("Pot.", "Potentiometer", "LED brightness control", "GPIO 32"),
    ]

    y = Inches(2.1)
    for model, name, purpose, pin in sensors:
        add_shape_fill(slide, Inches(0.8), y, Inches(5.5), Inches(0.45), BG_CARD)
        add_text_box(slide, Inches(0.9), y + Inches(0.05), Inches(1.1), Inches(0.35),
                     model, font_size=9, color=ACCENT2, bold=True)
        add_text_box(slide, Inches(2.05), y + Inches(0.05), Inches(1.3), Inches(0.35),
                     name, font_size=10, color=WHITE, bold=True)
        add_text_box(slide, Inches(3.4), y + Inches(0.05), Inches(2.0), Inches(0.35),
                     purpose, font_size=9, color=LIGHT)
        add_text_box(slide, Inches(5.4), y + Inches(0.05), Inches(0.9), Inches(0.35),
                     pin, font_size=8, color=GRAY)
        y += Inches(0.5)

    # Right column — Actuators
    add_section_header(slide, Inches(1.6), "ACTUATORS (OUTPUT)", ORANGE)
    # Shift header to right
    slide.shapes[-2].left = Inches(7.2)
    slide.shapes[-1].left = Inches(7.2)

    actuators = [
        ("Relay x2", "Light + Heater", "Time/temp controlled", "GPIO 26, 27"),
        ("SG90 Servo", "Fish Feeder", "Scheduled + manual feed", "GPIO 16"),
        ("Air Pump", "Oxygenation", "Time-based aeration", "GPIO 2"),
        ("PC Fan", "Cooling", "Button-triggered airflow", "GPIO 19"),
        ("WS2812B", "LED Strip", "RGB effects, touch control", "GPIO 18"),
        ("1W LEDs", "Status LEDs", "Red=feeding, Blue=servo", "GPIO 17, 25"),
    ]

    y = Inches(2.1)
    for model, name, purpose, pin in actuators:
        add_shape_fill(slide, Inches(7.2), y, Inches(5.5), Inches(0.45), BG_CARD)
        add_text_box(slide, Inches(7.3), y + Inches(0.05), Inches(1.1), Inches(0.35),
                     model, font_size=9, color=ORANGE, bold=True)
        add_text_box(slide, Inches(8.45), y + Inches(0.05), Inches(1.3), Inches(0.35),
                     name, font_size=10, color=WHITE, bold=True)
        add_text_box(slide, Inches(9.8), y + Inches(0.05), Inches(1.8), Inches(0.35),
                     purpose, font_size=9, color=LIGHT)
        add_text_box(slide, Inches(11.7), y + Inches(0.05), Inches(1.0), Inches(0.35),
                     pin, font_size=8, color=GRAY)
        y += Inches(0.5)

    # Displays section
    y_disp = Inches(5.3)
    add_section_header(slide, y_disp, "DISPLAYS", ACCENT)
    # Shift
    slide.shapes[-2].left = Inches(7.2)
    slide.shapes[-1].left = Inches(7.2)

    displays = [
        ("SSD1306 (0x3C)", "pH, Turbidity, TDS, Water condition"),
        ("SSD1306 (0x3D)", "Time, Date, Temperature, Heater status"),
        ("MAX7219", "Animated clock, Scrolling date"),
    ]
    y = y_disp + Inches(0.4)
    for name, shows in displays:
        add_text_box(slide, Inches(7.3), y, Inches(2.0), Inches(0.3),
                     name, font_size=10, color=ACCENT, bold=True)
        add_text_box(slide, Inches(9.4), y, Inches(3.3), Inches(0.3),
                     shows, font_size=10, color=LIGHT)
        y += Inches(0.35)

    # Central controller
    add_shape_fill(slide, Inches(0.8), Inches(5.7), Inches(5.5), Inches(1.3), BG_CARD)
    add_shape_fill(slide, Inches(0.8), Inches(5.7), Inches(5.5), Inches(0.05), ACCENT)

    add_text_box(slide, Inches(1.0), Inches(5.85), Inches(5.1), Inches(0.35),
                 "ESP32 WROOM — Central Controller", font_size=14, color=ACCENT, bold=True)
    add_text_box(slide, Inches(1.0), Inches(6.25), Inches(5.1), Inches(0.7),
                 "Reads all sensor data  ->  Processes logic & schedules  ->  Controls all actuators & displays\n"
                 "TIP144/TIP142 transistors switch high-current loads (pump, fan, LEDs)",
                 font_size=11, color=LIGHT)


# ════════════════════════════════════════════════════════════════
#  SLIDE 5: RESULTS & ANALYSIS
# ════════════════════════════════════════════════════════════════
def slide_results(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, BG_DARK)
    add_shape_fill(slide, Inches(0), Inches(0), W, Inches(0.06), GREEN)

    add_text_box(slide, Inches(0.8), Inches(0.4), Inches(5), Inches(0.5),
                 "RESULTS & ANALYSIS", font_size=11, color=GREEN, bold=True)
    add_text_box(slide, Inches(0.8), Inches(0.8), Inches(11), Inches(0.7),
                 "Model Performance & System Evaluation",
                 font_size=28, color=WHITE, bold=True)

    # Top row — Key metrics cards
    metrics = [
        ("92.5%", "mAP50", "Detection accuracy\nacross all test frames", ACCENT),
        ("95.3%", "Precision", "Very few false\ndetections", GREEN),
        ("84.8%", "Recall", "Catches 84.8% of\nall fish in frame", ACCENT2),
        ("76.1%", "mAP50-95", "Strict multi-threshold\naccuracy", ORANGE),
        ("30 FPS", "GPU Speed", "Real-time inference\non CUDA devices", YELLOW),
        ("~6 MB", "Model Size", "YOLOv8n nano model\nlightweight deploy", LIGHT),
    ]

    x = Inches(0.5)
    card_w = Inches(2.0)
    for val, label, desc, color in metrics:
        add_shape_fill(slide, x, Inches(1.7), card_w, Inches(1.5), BG_CARD)
        add_shape_fill(slide, x, Inches(1.7), card_w, Inches(0.04), color)
        add_text_box(slide, x + Inches(0.1), Inches(1.85), card_w - Inches(0.2), Inches(0.5),
                     val, font_size=26, color=color, bold=True, alignment=PP_ALIGN.CENTER)
        add_text_box(slide, x + Inches(0.1), Inches(2.3), card_w - Inches(0.2), Inches(0.3),
                     label, font_size=12, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)
        add_text_box(slide, x + Inches(0.1), Inches(2.6), card_w - Inches(0.2), Inches(0.5),
                     desc, font_size=9, color=GRAY, alignment=PP_ALIGN.CENTER)
        x += card_w + Inches(0.1)

    # Health Scoring Analysis - left
    y_mid = Inches(3.5)
    add_section_header(slide, y_mid, "ADAPTIVE HEALTH SCORING", GREEN)

    health_items = [
        "Per-fish baselines converge in ~25 seconds (150 samples)",
        "Healthy fish score 70-85% (vs 45-70% with fixed thresholds)",
        "Z-score detection catches behavioral changes 4-5 days before visible symptoms",
        "Multi-indicator consensus: 2+ anomalous metrics needed to trigger concern",
        "Hysteresis state machine eliminates status flickering completely",
        "Alert latency: 5-10 seconds for genuine health concerns",
    ]
    add_bullet_list(slide, Inches(0.8), y_mid + Inches(0.45), Inches(5.8), Inches(3.0),
                    health_items, font_size=11, color=LIGHT, spacing=Pt(6))

    # Hardware Results - right
    slide.shapes[-2].left = Inches(7.2)  # move section header
    slide.shapes[-3].left = Inches(7.2)
    add_section_header(slide, y_mid, "IoT SYSTEM PERFORMANCE", ORANGE)
    slide.shapes[-2].left = Inches(7.2)
    slide.shapes[-1].left = Inches(7.2)

    hw_items = [
        "All 7 sensors reading accurately in real-time",
        "Automated feeding on schedule via servo motor",
        "Temperature-controlled heater activation",
        "Time-based lighting and aeration cycles",
        "Dual OLED + MAX7219 displays showing live data",
        "Touch-controlled RGB LED lighting effects",
    ]
    add_bullet_list(slide, Inches(7.2), y_mid + Inches(0.45), Inches(5.3), Inches(3.0),
                    hw_items, font_size=11, color=LIGHT, spacing=Pt(6))

    # Comparison table at bottom
    y_comp = Inches(6.3)
    add_shape_fill(slide, Inches(0.8), y_comp, Inches(11.7), Inches(0.55), BG_CARD)
    add_text_box(slide, Inches(1.0), y_comp + Inches(0.1), Inches(3.5), Inches(0.35),
                 "Custom model vs Generic YOLO:", font_size=11, color=WHITE, bold=True)
    add_text_box(slide, Inches(4.5), y_comp + Inches(0.1), Inches(8.0), Inches(0.35),
                 "92.5% mAP50 (custom)  vs  30-50% mAP50 (generic)  —  "
                 "No false positives on hands, faces, or decorations",
                 font_size=11, color=GREEN)


# ════════════════════════════════════════════════════════════════
#  SLIDE 6: CONCLUSION & REFERENCES
# ════════════════════════════════════════════════════════════════
def slide_conclusion(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, BG_DARK)
    add_shape_fill(slide, Inches(0), Inches(0), W, Inches(0.06), ACCENT)

    add_text_box(slide, Inches(0.8), Inches(0.4), Inches(5), Inches(0.5),
                 "CONCLUSION", font_size=11, color=ACCENT, bold=True)
    add_text_box(slide, Inches(0.8), Inches(0.8), Inches(11), Inches(0.7),
                 "Summary & Future Work",
                 font_size=28, color=WHITE, bold=True)

    # What we built
    add_shape_fill(slide, Inches(0.8), Inches(1.7), Inches(5.8), Inches(3.0), BG_CARD)
    add_shape_fill(slide, Inches(0.8), Inches(1.7), Inches(5.8), Inches(0.05), ACCENT)

    add_text_box(slide, Inches(1.0), Inches(1.85), Inches(5.4), Inches(0.35),
                 "What We Built", font_size=14, color=ACCENT, bold=True)

    conclusions = [
        "End-to-end fish health monitoring: camera to alert in real-time",
        "CNN-based detection (92.5% mAP50) + adaptive per-fish baselines",
        "Statistical anomaly detection outperforms fixed thresholds significantly",
        "Full IoT automation: feeding, heating, lighting, aeration, water monitoring",
        "Live web dashboard with health scores, alerts, and parameter tuning",
        "Behavioral analysis detects illness days before visible symptoms appear",
    ]
    add_bullet_list(slide, Inches(1.0), Inches(2.25), Inches(5.4), Inches(2.3),
                    conclusions, font_size=11, color=LIGHT, spacing=Pt(6))

    # Progress & Timeline
    add_shape_fill(slide, Inches(7.0), Inches(1.7), Inches(5.8), Inches(1.3), BG_CARD)
    add_shape_fill(slide, Inches(7.0), Inches(1.7), Inches(5.8), Inches(0.05), GREEN)

    add_text_box(slide, Inches(7.2), Inches(1.85), Inches(5.4), Inches(0.35),
                 "Project Status", font_size=14, color=GREEN, bold=True)

    add_text_box(slide, Inches(7.2), Inches(2.25), Inches(2.5), Inches(0.35),
                 "Work Completed:", font_size=12, color=WHITE, bold=True)
    add_text_box(slide, Inches(9.7), Inches(2.25), Inches(2.5), Inches(0.35),
                 "85%", font_size=22, color=GREEN, bold=True)
    add_text_box(slide, Inches(7.2), Inches(2.65), Inches(2.5), Inches(0.35),
                 "Expected Completion:", font_size=12, color=WHITE, bold=True)
    add_text_box(slide, Inches(9.7), Inches(2.65), Inches(2.5), Inches(0.35),
                 "May 2026", font_size=14, color=ACCENT2, bold=True)

    # Future work
    add_shape_fill(slide, Inches(7.0), Inches(3.2), Inches(5.8), Inches(1.5), BG_CARD)
    add_shape_fill(slide, Inches(7.0), Inches(3.2), Inches(5.8), Inches(0.05), ORANGE)

    add_text_box(slide, Inches(7.2), Inches(3.35), Inches(5.4), Inches(0.35),
                 "Remaining Work", font_size=14, color=ORANGE, bold=True)

    future = [
        "Hardware-software integration (ESP32 + dashboard link)",
        "Extended testing with multiple fish species",
        "Mobile app notifications for alerts",
    ]
    add_bullet_list(slide, Inches(7.2), Inches(3.7), Inches(5.4), Inches(1.0),
                    future, font_size=11, color=LIGHT, spacing=Pt(5))

    # References
    y_ref = Inches(5.0)
    add_section_header(slide, y_ref, "REFERENCES")

    refs = [
        "Ultralytics YOLOv8 (2023) — Real-time object detection architecture",
        "Bewley et al. (2016) — SORT: Simple Online and Realtime Tracking",
        "Welford, B.P. (1962) — Numerically stable single-pass variance computation",
        "Kalman, R.E. (1960) — Optimal state estimation for linear dynamic systems",
        "Kuhn, H.W. (1955) — The Hungarian Algorithm for assignment problems",
        "OpenCV Library — Computer vision and image processing",
        "Espressif ESP32 — IoT microcontroller platform documentation",
    ]
    add_bullet_list(slide, Inches(0.8), y_ref + Inches(0.4), Inches(11.7), Inches(2.0),
                    refs, font_size=10, color=GRAY, spacing=Pt(3))

    # Thank you
    add_text_box(slide, Inches(0.8), Inches(7.0), Inches(12), Inches(0.4),
                 "Thank You  |  Questions?",
                 font_size=14, color=ACCENT, bold=True, alignment=PP_ALIGN.CENTER)


# ════════════════════════════════════════════════════════════════
#  BUILD
# ════════════════════════════════════════════════════════════════
def main():
    prs = Presentation()
    prs.slide_width = W
    prs.slide_height = H

    slide_title(prs)
    slide_problem(prs)
    slide_methodology_ai(prs)
    slide_methodology_hw(prs)
    slide_results(prs)
    slide_conclusion(prs)

    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "Fish_Health_Monitoring_Presentation.pptx")
    prs.save(out_path)
    print(f"Presentation saved: {out_path}")
    print(f"  Slides: {len(prs.slides)}")
    print(f"  Size: Widescreen 16:9")


if __name__ == "__main__":
    main()
