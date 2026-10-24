"""
make_qa_slides.py — turn Questions.txt into a presentation-style PDF
====================================================================

16:9 slides: title, how-to-read, agenda, one divider per part (with the
questions it covers), one slide per question (long answers continue on the
next slide), glossary. Aligned / table-like lines of an answer keep their
columns in a monospaced font.

Run (from idr-project/):  python tools/make_qa_slides.py
Output: IDR_Interview_QA_Slides.pdf
"""

from __future__ import annotations

import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Frame, KeepTogether, Paragraph, Spacer, XPreformatted

ROOT = Path(__file__).resolve().parents[1]
SRC, OUT = ROOT / "Questions.txt", ROOT / "IDR_Interview_QA_Slides.pdf"
W, H = 960, 540
NAVY, ACCENT, INK, MUTED, LIGHT, CARD = (HexColor(c) for c in
                                        ("#0f2a4a", "#2563eb", "#1e293b", "#64748b", "#e8eef8", "#f5f8fd"))

FONTS = Path("C:/Windows/Fonts")
pdfmetrics.registerFont(TTFont("Body", str(FONTS / "arial.ttf")))
pdfmetrics.registerFont(TTFont("Body-Bold", str(FONTS / "arialbd.ttf")))
pdfmetrics.registerFont(TTFont("Body-Italic", str(FONTS / "ariali.ttf")))
pdfmetrics.registerFont(TTFont("Mono", str(FONTS / "consola.ttf")))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic", boldItalic="Body-Bold")

S_BODY = ParagraphStyle("body", fontName="Body", fontSize=16.5, leading=23, textColor=INK, spaceAfter=8, alignment=TA_LEFT)
S_ITEM = ParagraphStyle("item", parent=S_BODY, leftIndent=30, firstLineIndent=-24, spaceAfter=6)
S_MONO = ParagraphStyle("mono", fontName="Mono", fontSize=14, leading=18.5, textColor=INK, spaceBefore=3, spaceAfter=9,
                        backColor=CARD, borderPadding=(6, 8, 6, 8), leftIndent=8, rightIndent=8)
S_QTITLE = ParagraphStyle("qt", fontName="Body-Bold", fontSize=23, leading=28, textColor=NAVY)
S_GLOSS = ParagraphStyle("gl", fontName="Body", fontSize=11.5, leading=14.5, textColor=INK, spaceAfter=5)
S_SMALL = ParagraphStyle("sm", fontName="Body", fontSize=13, leading=18, textColor=white)


# ------------------------------------------------------------------ parsing
def parse(text: str):
    lines = [ln.rstrip("\r") for ln in text.split("\n")]
    intro, parts, glossary = [], [], []
    i, mode = 0, "intro"
    cur_part = cur_q = None
    while i < len(lines):
        ln = lines[i]
        m_part = re.match(r"^PART (\d+) — (.+)$", ln)
        if m_part and i > 0 and lines[i - 1].startswith("====="):
            mode = "parts"
            cur_part = {"num": int(m_part.group(1)), "title": m_part.group(2).strip(), "qs": []}
            parts.append(cur_part)
            i += 2  # skip the closing ===== line
            continue
        if ln.startswith("GLOSSARY") and lines[i - 1].startswith("====="):
            mode, i = "glossary", i + 2
            continue
        if mode == "intro":
            intro.append(ln)
        elif mode == "parts":
            m_q = re.match(r"^Q(\d+)\) (.+)$", ln)
            if m_q:
                q = m_q.group(2).strip()
                i += 1
                while i < len(lines) and not lines[i].startswith("A:"):
                    if lines[i].strip():
                        q += " " + lines[i].strip()
                    i += 1
                first = lines[i][2:] if i < len(lines) else ""
                # keep the column offset when the answer starts with a table header (e.g. Q67)
                cur_q = {"num": int(m_q.group(1)), "q": q, "a": [first if is_tabular(first) else first.lstrip()]}
                cur_part["qs"].append(cur_q)
            elif cur_q is not None and not ln.startswith("====="):
                cur_q["a"].append(ln)
        elif mode == "glossary":
            glossary.append(ln)
        i += 1
    return intro, parts, glossary


ITEM = re.compile(r"^\s*((?:\d+(?:-\d+)?\.)|-)\s+(.*)$")


def is_tabular(ln: str) -> bool:
    """Column-aligned line (dot leaders or 3+ inner spaces), ignoring the
    spacing after a list marker such as '7.   text'."""
    m = ITEM.match(ln)
    body = (m.group(2) if m else ln).strip()
    return bool(body) and ("...." in body or re.search(r"\S {3,}\S", body) is not None)


def scaled(style: ParagraphStyle, k: float) -> ParagraphStyle:
    return ParagraphStyle(f"{style.name}{k}", parent=style, fontSize=style.fontSize * k, leading=style.leading * k,
                          spaceAfter=style.spaceAfter * k, spaceBefore=style.spaceBefore * k)


def answer_flowables(a_lines: list[str], k: float = 1.0):
    """Paragraphs, list items (hanging indent) and monospaced blocks, with
    all font sizes scaled by `k` (shrink-to-fit)."""
    S_BODY_, S_ITEM_, S_MONO_ = scaled(S_BODY, k), scaled(S_ITEM, k), scaled(S_MONO, k)
    flows, para, kind, mono = [], [], None, []

    def flush_para():
        nonlocal para, kind
        if para:
            text = " ".join(para)
            if kind == "item":
                m = ITEM.match(text)
                flows.append(Paragraph(f"<b>{escape(m.group(1))}</b>&nbsp;&nbsp;{fmt(m.group(2))}", S_ITEM_))
            else:
                flows.append(Paragraph(fmt(text), S_BODY_))
        para, kind = [], None

    def flush_mono():
        nonlocal mono
        if mono:
            indent = min(len(x) - len(x.lstrip()) for x in mono if x.strip())
            flows.append(XPreformatted(escape("\n".join(x[indent:] for x in mono)), S_MONO_))
        mono = []

    for ln in a_lines:
        if not ln.strip():
            flush_para(); flush_mono()
            continue
        if is_tabular(ln):
            flush_para()
            mono.append(ln)
            continue
        flush_mono()
        if ITEM.match(ln):
            flush_para()
            para, kind = [ln.strip()], "item"
        else:
            para.append(ln.strip())
    flush_para(); flush_mono()
    return flows


KEEP_CASE = {"gnss": "GNSS", "ins": "INS", "ml": "ML", "2-d": "2-D", "kalman": "Kalman", "(kalman": "(Kalman",
             "ml)": "ML)", "(anchored": "(anchored"}


def sentence_case(title: str) -> str:
    """'GNSS + INS FUSION (THE KALMAN FILTER)' -> 'GNSS + INS fusion (the Kalman filter)'."""
    words = title.lower().split()
    words = [KEEP_CASE.get(w, w) for w in words]
    return (words[0][:1].upper() + words[0][1:] + " " + " ".join(words[1:])).strip()


def fmt(text: str) -> str:
    """Escape and lightly emphasise quoted terms: "term" -> bold."""
    t = escape(text).replace("-&gt;", "→")  # "->" as a real arrow
    # a quoted term starts right after the opening quote (no space), so `"m" = metres, "m/s"` bolds m and m/s
    return re.sub(r"\"([^\"\s][^\"]{0,58}?)\"", lambda m: f"\u201c<b>{m.group(1)}</b>\u201d", t)


# ------------------------------------------------------------------ drawing
class Deck:
    def __init__(self, path: Path):
        self.c = canvas.Canvas(str(path), pagesize=(W, H))
        self.c.setTitle("IDR — Interview Questions and Answers")
        self.c.setAuthor("IDR project team")
        self.c.setSubject("AI/ML-based Intelligent Dead Reckoning for GPS-denied tunnels")
        self.page = 0

    def _footer(self, dark=False):
        self.page += 1
        c = self.c
        c.setFont("Body", 9.5)
        c.setFillColor(HexColor("#9fb3cf") if dark else MUTED)
        c.drawString(40, 18, "IDR · Intelligent Dead Reckoning · Interview Q&A")
        c.drawRightString(W - 40, 18, str(self.page))

    def _band(self, label: str):
        c = self.c
        c.setFillColor(NAVY)
        c.rect(0, H - 46, W, 46, stroke=0, fill=1)
        c.setFillColor(ACCENT)
        c.rect(0, H - 50, W, 4, stroke=0, fill=1)
        c.setFillColor(HexColor("#c7d6ee"))
        c.setFont("Body-Bold", 12)
        c.drawString(40, H - 29, label.upper())

    def title_slide(self):
        c = self.c
        c.setFillColor(NAVY); c.rect(0, 0, W, H, stroke=0, fill=1)
        c.setFillColor(ACCENT); c.rect(60, 300, 90, 6, stroke=0, fill=1)
        c.setFillColor(white)
        c.setFont("Body-Bold", 46); c.drawString(60, 330, "Intelligent Dead Reckoning")
        c.setFont("Body", 22); c.drawString(60, 262, "AI/ML-based navigation for GPS-denied tunnels")
        c.setFillColor(HexColor("#c7d6ee"))
        c.setFont("Body-Bold", 26); c.drawString(60, 200, "Interview questions & answers")
        c.setFont("Body", 15)
        c.drawString(60, 120, "M.Tech 1st Semester · Machine Learning Project")
        c.drawString(60, 96, "93 questions · 18 parts · glossary of technical terms")
        self._footer(dark=True); c.showPage()

    def text_slide(self, label: str, title: str, flows: list):
        self._content_slides(label, title, flows)

    def agenda(self, parts):
        c = self.c
        self._band("Agenda")
        p = Paragraph("What this deck covers", S_QTITLE); p.wrapOn(c, W - 80, 60); p.drawOn(c, 40, H - 98)
        items = [(f"Part {p_['num']}", sentence_case(p_["title"]), f"Q{p_['qs'][0]['num']}–Q{p_['qs'][-1]['num']}")
                 for p_ in parts]
        items.append(("", "Glossary", "A–Z"))
        col_h = (len(items) + 1) // 2
        for idx, (num, title, rng) in enumerate(items):
            col, row = divmod(idx, col_h)
            x, y = 40 + col * 450, H - 140 - row * 34
            c.setFillColor(ACCENT); c.setFont("Body-Bold", 12.5); c.drawString(x, y, num)
            size = 13.5  # shrink long titles so they never reach the Q-range column
            while pdfmetrics.stringWidth(title, "Body", size) > 300 and size > 9:
                size -= 0.5
            c.setFillColor(INK); c.setFont("Body", size); c.drawString(x + 58, y, title)
            c.setFillColor(MUTED); c.setFont("Body", 11); c.drawRightString(x + 430, y, rng)
        self._footer(); c.showPage()

    def divider(self, part):
        c = self.c
        c.setFillColor(NAVY); c.rect(0, 0, W, H, stroke=0, fill=1)
        c.setFillColor(ACCENT); c.rect(60, H - 132, 70, 5, stroke=0, fill=1)
        c.setFillColor(HexColor("#9fb3cf")); c.setFont("Body-Bold", 18); c.drawString(60, H - 110, f"PART {part['num']}")
        t = Paragraph(escape(part["title"]), ParagraphStyle("dt", fontName="Body-Bold", fontSize=34, leading=40, textColor=white))
        _, th = t.wrapOn(c, W - 120, 200); t.drawOn(c, 60, H - 150 - th)
        y = H - 180 - th
        qs = part["qs"]
        size = 13 if len(qs) <= 9 else 11.5
        for q in qs:
            p = Paragraph(f"<b>Q{q['num']}</b>&nbsp;&nbsp;{escape(q['q'])}",
                          ParagraphStyle("dq", parent=S_SMALL, fontSize=size, leading=size + 4.5))
            _, ph = p.wrapOn(c, W - 120, 100)
            if y - ph < 40:
                break
            p.drawOn(c, 60, y - ph); y -= ph + 4
        self._footer(dark=True); c.showPage()

    def _content_slides(self, label: str, title: str, flows: list, cont_title: str | None = None):
        flows = list(flows)
        first = True
        while True:
            c = self.c
            self._band(label)
            t = Paragraph(escape(title if first else (cont_title or title + " (continued)")), S_QTITLE)
            _, th = t.wrapOn(c, W - 80, 120)
            t.drawOn(c, 40, H - 70 - th)
            c.setStrokeColor(LIGHT); c.setLineWidth(1.2)
            c.line(40, H - 80 - th, W - 40, H - 80 - th)
            frame = Frame(34, 34, W - 68, H - 92 - th - 34, leftPadding=6, rightPadding=6, topPadding=8, bottomPadding=0,
                          showBoundary=0)
            frame.addFromList(flows, c)
            self._footer(); c.showPage()
            first = False
            if not flows:
                break

    def question(self, part, q):
        label = f"Part {part['num']} · {part['title']}"
        title = f"Q{q['num']}.  {q['q']}"
        th = Paragraph(escape(title), S_QTITLE).wrap(W - 80, 120)[1]
        avail_w, avail_h = W - 68 - 12, H - 92 - th - 34 - 8

        def make(k):
            return [Paragraph("<font color='#2563eb'><b>ANSWER</b></font>",
                              ParagraphStyle("al", parent=S_BODY, fontSize=11, leading=14, spaceAfter=4))] \
                + answer_flowables(q["a"], k)

        def height(flows):
            return sum(f.wrap(avail_w, 10_000)[1] + f.getSpaceBefore() + f.getSpaceAfter() for f in flows)

        # shrink-to-fit (down to 80%) when an answer only slightly overflows one slide
        flows = next((f for f in (make(k) for k in (1.0, 0.93, 0.86, 0.8)) if height(f) <= avail_h * 0.97), make(1.0))
        self._content_slides(label, title, flows, cont_title=f"Q{q['num']}.  (continued)")

    def glossary(self, gl_lines):
        entries, cur = [], None
        for ln in gl_lines:
            if not ln.strip():
                continue
            if not ln.startswith(" ") and " — " in ln:
                cur = ln.split(" — ", 1)
                entries.append(cur)
            elif cur is not None:
                cur[1] += " " + ln.strip()
        flows = [Paragraph(f"<b>{escape(t)}</b> — {escape(d)}", S_GLOSS) for t, d in entries]
        while flows:
            c = self.c
            self._band("Glossary")
            p = Paragraph("Glossary of technical terms", S_QTITLE); p.wrapOn(c, W - 80, 60); p.drawOn(c, 40, H - 98)
            for col in range(2):
                f = Frame(34 + col * (W - 68) / 2, 34, (W - 68) / 2, H - 140, leftPadding=6, rightPadding=12,
                          topPadding=4, bottomPadding=0)
                f.addFromList(flows, c)
            self._footer(); c.showPage()

    def closing(self):
        c = self.c
        c.setFillColor(NAVY); c.rect(0, 0, W, H, stroke=0, fill=1)
        c.setFillColor(white); c.setFont("Body-Bold", 40); c.drawString(60, 300, "Thank you")
        c.setFillColor(HexColor("#c7d6ee")); c.setFont("Body", 17)
        c.drawString(60, 255, "Full details: README.md · PROGRESS.md · results/metrics.md")
        c.drawString(60, 228, "Demo: python main.py app  →  http://localhost:8000")
        self._footer(dark=True); c.showPage()

    def save(self):
        self.c.save()


def main():
    intro, parts, glossary = parse(SRC.read_text(encoding="utf-8"))
    deck = Deck(OUT)
    deck.title_slide()
    # "How to read this file" block from the intro
    try:
        a = intro.index("How to read this file") + 2
        b = next(k for k in range(a, len(intro)) if intro[k].startswith("Contents"))
        deck.text_slide("Introduction", "How to read this deck",
                        answer_flowables([ln.replace("this file", "this deck") for ln in intro[a:b]]))
    except (ValueError, StopIteration):
        pass
    deck.agenda(parts)
    for part in parts:
        deck.divider(part)
        for q in part["qs"]:
            deck.question(part, q)
    deck.glossary(glossary)
    deck.closing()
    deck.save()
    n_q = sum(len(p["qs"]) for p in parts)
    print(f"{OUT.name}: {deck.page} slides, {len(parts)} parts, {n_q} questions")


if __name__ == "__main__":
    main()
