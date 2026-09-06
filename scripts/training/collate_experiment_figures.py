#!/usr/bin/env python3
"""Collate experiment PDFs and annotated figure panels into one report."""

import argparse
import hashlib
import io
import json
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", action="append", default=[])
    parser.add_argument("--addendum", required=True,
                        help="JSON list of panels with title, text, and optional image")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    panels = json.loads(Path(args.addendum).read_text())
    writer = PdfWriter()
    inputs = [Path(args.addendum)]
    for filename in args.pdf:
        writer.append(filename)
        inputs.append(Path(filename))
    styles = getSampleStyleSheet()
    for panel in panels:
        width, height = 900, 650
        buffer = io.BytesIO()
        canvas = Canvas(buffer, pagesize=(width, height))
        title = Paragraph(panel["title"], styles["Title"])
        _, size = title.wrap(width - 72, 90)
        title.drawOn(canvas, 36, height - 30 - size)
        paragraph = Paragraph(panel["text"], styles["BodyText"])
        _, size = paragraph.wrap(width - 72, 180)
        y = height - 90 - size
        paragraph.drawOn(canvas, 36, y)
        if panel.get("image"):
            path = Path(panel["image"])
            inputs.append(path)
            image = ImageReader(str(path))
            image_width, image_height = image.getSize()
            scale = min((width - 72) / image_width, (y - 65) / image_height)
            canvas.drawImage(image, (width - image_width * scale) / 2, 45,
                             width=image_width * scale, height=image_height * scale)
        canvas.setFont("Helvetica", 9)
        canvas.drawString(36, 20, "MHCflurry 2.3.0 candidate experiments | exploratory results, not release acceptance")
        canvas.showPage()
        canvas.save()
        buffer.seek(0)
        writer.append(PdfReader(buffer))
    out = Path(args.out)
    if out.resolve() in {path.resolve() for path in inputs}:
        raise ValueError("Output must differ from every input")
    out.parent.mkdir(parents=True, exist_ok=True)
    writer.write(out)
    manifest = {
        "arguments": vars(args), "pages": len(writer.pages),
        "inputs": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                   for path in inputs],
        "output_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    }
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Wrote %d pages to %s" % (len(writer.pages), out))


if __name__ == "__main__":
    main()
