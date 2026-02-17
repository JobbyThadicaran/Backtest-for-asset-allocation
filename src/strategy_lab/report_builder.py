from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle,
                                Paragraph, Spacer, Image)
from reportlab.lib.styles import getSampleStyleSheet
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt


def generate_pdf_report(
    results_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    filename: str = "backtest_report.pdf"
):
    """Generates a PDF with metrics table, cumulative-return plot, and drawdown plot."""
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # ---- Title ----
    story.append(Paragraph("Backtest Report", styles['Title']))
    story.append(Spacer(1, 12))

    # ---- Metrics Table ----
    story.append(Paragraph("Performance Metrics", styles['Heading2']))
    data = [['Metric', 'Value']]
    for idx, row in metrics_df.iterrows():
        val = row.iloc[0]
        if isinstance(val, float):
            val = f"{val:.4f}"
        data.append([idx, val])

    t = Table(data)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(t)
    story.append(Spacer(1, 24))

    # ---- Cumulative Returns Plot ----
    story.append(Paragraph("Cumulative Returns", styles['Heading2']))
    plt.figure(figsize=(6, 4))
    plt.plot((1 + results_df['Strategy']).cumprod(), label='Strategy')
    plt.title("Cumulative Returns")
    plt.legend()
    plt.grid(True)
    buf1 = io.BytesIO()
    plt.savefig(buf1, format='png')
    plt.close()
    buf1.seek(0)
    story.append(Image(buf1, width=400, height=300))
    story.append(Spacer(1, 12))

    # ---- Drawdown Plot ----
    story.append(Paragraph("Drawdown", styles['Heading2']))
    plt.figure(figsize=(6, 4))
    wealth = (1 + results_df['Strategy']).cumprod()
    dd = (wealth - wealth.cummax()) / wealth.cummax()
    plt.plot(dd, label='Drawdown', color='red')
    plt.title("Drawdown")
    plt.legend()
    plt.grid(True)
    buf2 = io.BytesIO()
    plt.savefig(buf2, format='png')
    plt.close()
    buf2.seek(0)
    story.append(Image(buf2, width=400, height=300))

    doc.build(story)
    return filename
