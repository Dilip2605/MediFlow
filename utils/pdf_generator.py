from datetime import datetime

# ═══════════════════════════════════════════════════════════════
# PDF REPORT GENERATOR
# Purpose: Generate downloadable medical reports for doctors
# Library: fpdf (install with: pip install fpdf)
# ═══════════════════════════════════════════════════════════════


def generate_medical_report(
    patient_name: str,
    patient_age: int,
    patient_gender: str,
    prediction_type: str,
    result: str,
    probability: float,
    risk_level: str,
    clinical_details: dict,
    recommendations: list
) -> bytes:
    """
    Generate a professional PDF medical report.
    
    Parameters:
        patient_name: Patient's full name
        patient_age: Patient's age
        patient_gender: Male/Female/Other
        prediction_type: Type of test (Diabetes/Heart Disease/etc)
        result: Prediction result (DIABETIC/HEALTHY/etc)
        probability: Confidence probability (0.0 to 1.0)
        risk_level: LOW RISK / MODERATE RISK / HIGH RISK
        clinical_details: Dict of test values {label: value}
        recommendations: List of medical recommendation strings
    
    Returns:
        PDF as bytes — use with st.download_button()
    
    Usage:
        pdf_bytes = generate_medical_report(...)
        if pdf_bytes:
            st.download_button("Download PDF", pdf_bytes, "report.pdf")
    """
    try:
        from fpdf import FPDF

        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        # ─── HEADER ─────────────────────────────────────────
        pdf.set_font("Arial", "B", 22)
        pdf.set_text_color(31, 78, 121)
        pdf.cell(0, 12, "MediFlow AI", ln=True, align="C")

        pdf.set_font("Arial", "B", 14)
        pdf.set_text_color(46, 117, 182)
        pdf.cell(0, 8, "AI-Powered Medical Diagnostic Report", ln=True, align="C")

        pdf.set_font("Arial", size=9)
        pdf.set_text_color(128, 128, 128)
        pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%d %B %Y at %H:%M')}", ln=True, align="C")
        pdf.cell(0, 6, "CONFIDENTIAL — FOR PHYSICIAN USE ONLY", ln=True, align="C")

        # Divider line
        pdf.set_draw_color(31, 78, 121)
        pdf.set_line_width(0.5)
        pdf.line(10, pdf.get_y() + 2, 200, pdf.get_y() + 2)
        pdf.ln(8)

        # ─── PATIENT INFORMATION ────────────────────────────
        pdf.set_font("Arial", "B", 13)
        pdf.set_text_color(31, 78, 121)
        pdf.cell(0, 9, "PATIENT INFORMATION", ln=True)

        pdf.set_draw_color(200, 200, 200)
        pdf.set_line_width(0.2)

        info_items = [
            ("Patient Name", patient_name),
            ("Age", f"{patient_age} years"),
            ("Gender", patient_gender),
            ("Report Date", datetime.now().strftime("%d/%m/%Y")),
            ("Report ID", f"MF-{datetime.now().strftime('%Y%m%d%H%M%S')}"),
        ]

        pdf.set_font("Arial", size=11)
        for label, value in info_items:
            pdf.set_text_color(100, 100, 100)
            pdf.cell(60, 7, f"  {label}:", ln=False)
            pdf.set_text_color(30, 30, 30)
            pdf.set_font("Arial", "B", 11)
            pdf.cell(0, 7, value, ln=True)
            pdf.set_font("Arial", size=11)

        pdf.ln(5)

        # ─── AI PREDICTION RESULT ───────────────────────────
        pdf.set_draw_color(31, 78, 121)
        pdf.set_line_width(0.5)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)

        pdf.set_font("Arial", "B", 13)
        pdf.set_text_color(31, 78, 121)
        pdf.cell(0, 9, f"AI ANALYSIS: {prediction_type.upper()}", ln=True)

        # Result with color
        pdf.set_font("Arial", "B", 18)
        is_positive = any(word in result.lower()
                         for word in ["diabetic", "disease", "positive", "pneumonia", "tb"])
        if is_positive:
            pdf.set_text_color(200, 0, 0)
        else:
            pdf.set_text_color(0, 150, 0)
        pdf.cell(0, 12, f"Result: {result}", ln=True)

        # Probability and risk
        pdf.set_font("Arial", size=12)
        pdf.set_text_color(60, 60, 60)
        pdf.cell(0, 7, f"Confidence Level: {int(probability * 100)}%", ln=True)
        pdf.cell(0, 7, f"Risk Assessment: {risk_level}", ln=True)

        pdf.ln(5)

        # ─── CLINICAL VALUES ────────────────────────────────
        if clinical_details:
            pdf.set_draw_color(31, 78, 121)
            pdf.set_line_width(0.5)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(5)

            pdf.set_font("Arial", "B", 13)
            pdf.set_text_color(31, 78, 121)
            pdf.cell(0, 9, "CLINICAL VALUES", ln=True)

            # Table header
            pdf.set_fill_color(31, 78, 121)
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("Arial", "B", 10)
            pdf.cell(90, 8, "  Parameter", border=1, fill=True)
            pdf.cell(100, 8, "  Value", border=1, fill=True, ln=True)

            # Table rows
            pdf.set_font("Arial", size=10)
            for i, (key, val) in enumerate(clinical_details.items()):
                fill = i % 2 == 0
                pdf.set_fill_color(235, 243, 251)
                pdf.set_text_color(50, 50, 50)
                pdf.cell(90, 7, f"  {key}", border=1, fill=fill)
                pdf.cell(100, 7, f"  {val}", border=1, fill=fill, ln=True)

            pdf.ln(5)

        # ─── RECOMMENDATIONS ────────────────────────────────
        if recommendations:
            pdf.set_draw_color(31, 78, 121)
            pdf.set_line_width(0.5)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(5)

            pdf.set_font("Arial", "B", 13)
            pdf.set_text_color(31, 78, 121)
            pdf.cell(0, 9, "MEDICAL RECOMMENDATIONS", ln=True)

            pdf.set_font("Arial", size=11)
            pdf.set_text_color(50, 50, 50)
            for i, rec in enumerate(recommendations, 1):
                pdf.cell(10, 7, f"{i}.", ln=False)
                pdf.multi_cell(0, 7, rec)

            pdf.ln(3)

        # ─── DISCLAIMER ─────────────────────────────────────
        pdf.set_draw_color(200, 200, 200)
        pdf.set_line_width(0.2)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(3)

        pdf.set_font("Arial", "I", 8)
        pdf.set_text_color(150, 150, 150)
        disclaimer = (
            "DISCLAIMER: This report is generated by MediFlow AI for clinical assistance only. "
            "It should not replace professional medical diagnosis or treatment decisions. "
            "All AI predictions must be validated by a qualified physician before clinical action. "
            "MediFlow AI — Developed by Dilipkumar P, BE CSE, Anna University."
        )
        pdf.multi_cell(0, 5, disclaimer)

        # Return as bytes
        return pdf.output(dest='S').encode('latin1')

    except ImportError:
        return None
    except Exception as e:
        print(f"PDF generation error: {e}")
        return None


def generate_inventory_report(medicines: list) -> bytes:
    """
    Generate inventory status PDF report.
    
    Parameters:
        medicines: List of tuples (id, name, quantity, last_updated)
    """
    try:
        from fpdf import FPDF

        pdf = FPDF()
        pdf.add_page()

        # Header
        pdf.set_font("Arial", "B", 18)
        pdf.set_text_color(31, 78, 121)
        pdf.cell(0, 12, "MediFlow AI — Inventory Report", ln=True, align="C")

        pdf.set_font("Arial", size=10)
        pdf.set_text_color(128, 128, 128)
        pdf.cell(0, 7, f"Generated: {datetime.now().strftime('%d %B %Y at %H:%M')}", ln=True, align="C")
        pdf.ln(5)

        # Summary
        total = len(medicines)
        critical = sum(1 for m in medicines if m[2] < 30)
        low = sum(1 for m in medicines if 30 <= m[2] < 50)

        pdf.set_font("Arial", "B", 12)
        pdf.set_text_color(31, 78, 121)
        pdf.cell(0, 8, f"Summary: {total} medicines | {critical} critical | {low} low stock", ln=True)
        pdf.ln(3)

        # Table
        pdf.set_fill_color(31, 78, 121)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", "B", 10)
        pdf.cell(10, 8, "ID", border=1, fill=True)
        pdf.cell(90, 8, "Medicine Name", border=1, fill=True)
        pdf.cell(40, 8, "Quantity", border=1, fill=True)
        pdf.cell(50, 8, "Status", border=1, fill=True, ln=True)

        pdf.set_font("Arial", size=10)
        for i, med in enumerate(medicines):
            qty = med[2]
            if qty < 30:
                status = "CRITICAL"
                pdf.set_text_color(200, 0, 0)
            elif qty < 50:
                status = "LOW STOCK"
                pdf.set_text_color(180, 100, 0)
            else:
                status = "OK"
                pdf.set_text_color(0, 150, 0)

            fill = i % 2 == 0
            pdf.set_fill_color(235, 243, 251)
            pdf.set_text_color(50, 50, 50)
            pdf.cell(10, 7, str(med[0]), border=1, fill=fill)
            pdf.cell(90, 7, med[1], border=1, fill=fill)
            pdf.cell(40, 7, str(qty), border=1, fill=fill)
            if qty < 30:
                pdf.set_text_color(200, 0, 0)
            elif qty < 50:
                pdf.set_text_color(180, 100, 0)
            else:
                pdf.set_text_color(0, 150, 0)
            pdf.cell(50, 7, status, border=1, fill=fill, ln=True)

        return pdf.output(dest='S').encode('latin1')

    except ImportError:
        return None
    except Exception as e:
        print(f"PDF error: {e}")
        return None
