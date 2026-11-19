
import io
from datetime import datetime
import numpy as np
from PIL import Image as PILImage
from reportlab.platypus import Image as PDFImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


def create_pdf_image(image_array, width=75, height=55):
    """Convert a NumPy image into a PDF-compatible image."""

    if image_array is None:
        return None

    image_array = np.asarray(image_array)

    if image_array.dtype == bool:
        image_array = image_array.astype(np.uint8) * 255

    elif np.issubdtype(image_array.dtype, np.floating):
        image_array = np.nan_to_num(image_array)
        image_array = np.clip(image_array, 0, 1) * 255
        image_array = image_array.astype(np.uint8)

    else:
        image_array = np.clip(
            image_array, 0, 255
        ).astype(np.uint8)

    if image_array.ndim == 2:
        pil_image = PILImage.fromarray(image_array).convert("RGB")
    else:
        pil_image = PILImage.fromarray(image_array).convert("RGB")

    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)

    return PDFImage(buffer, width=width * mm, height=height * mm)

def generate_prediction_pdf(
    image_name,
    threshold,
    total_pixels,
    predicted_pixels,
    coverage,
    rgb_image=None,
    predicted_mask=None,
    probability_map=None,
    overlay_image=None
):
    
    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm
    )

    styles = getSampleStyleSheet()
    elements = []

    elements.append(
        Paragraph(
            "LandslideGuard AI",
            styles["Title"]
        )
    )

    elements.append(
        Paragraph(
            "Satellite Image Prediction Report",
            styles["Heading2"]
        )
    )

    elements.append(Spacer(1, 15))

    elements.append(
        Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            styles["Normal"]
        )
    )

    elements.append(Spacer(1, 15))

    report_data = [
        ["Parameter", "Result"],
        ["Input Image", str(image_name)],
        ["Model", "U-Net"],
        ["Input Channels", "14"],
        ["Image Resolution", "128 x 128"],
        ["Prediction Threshold", f"{threshold:.2f}"],
        ["Total Pixels", f"{total_pixels:,}"],
        ["Predicted Landslide Pixels", f"{predicted_pixels:,}"],
        ["Predicted Pixel Coverage", f"{coverage:.2f}%"],
    ]

    table = Table(
        report_data,
        colWidths=[75 * mm, 85 * mm]
    )

    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#14213D")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
    )

    elements.append(table)

    elements.append(Spacer(1, 20))

    elements.append(
        Paragraph(
            "<b>Interpretation:</b> Predicted pixel coverage "
            "represents the percentage of image pixels classified "
            "as landslide by the model at the selected threshold. "
            "It is not a calibrated probability of landslide risk.",
            styles["Normal"]
        )
    )

    elements.append(Spacer(1, 15))

    elements.append(
        Paragraph(
            "Disclaimer: This is an academic machine-learning "
            "prototype. Predictions should not be used as official "
            "landslide warnings or for emergency decisions.",
            styles["Italic"]
        )
    )

    document.build(elements)

    # ==========================================
    # PREDICTION VISUALIZATIONS
    # ==========================================

    elements.append(Spacer(1, 20))

    elements.append(
        Paragraph(
            "Prediction Visualizations",
            styles["Heading2"]
        )
    )

    visualizations = [
        ("Satellite RGB Image", rgb_image),
        ("Predicted Landslide Mask", predicted_mask),
        ("Probability Map", probability_map),
        ("Landslide Overlay", overlay_image)
    ]

    image_cells = []

    for title, image_array in visualizations:

        if image_array is not None:

            image_element = create_pdf_image(
                image_array
            )

            image_cells.append([
                Paragraph(title, styles["Heading3"]),
                image_element
            ])

    if image_cells:

        visualization_table = Table(
            image_cells,
            colWidths=[45 * mm, 90 * mm]
        )

        visualization_table.setStyle(
            TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("PADDING", (0, 0), (-1, -1), 8),
            ])
        )

        elements.append(visualization_table)

    else:
        elements.append(
            Paragraph(
                "No prediction visualizations were provided.",
                styles["Normal"]
            )
        )
    buffer.seek(0)

    return buffer.getvalue()
