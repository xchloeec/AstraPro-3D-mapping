"""Create an A4 vector checkerboard with a print-scale verification bar."""
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm

root = Path(__file__).resolve().parent
folder = root / 'output' / 'pdf'
folder.mkdir(parents=True, exist_ok=True)
target = folder / 'camera_checkerboard_A4_18mm.pdf'
c = canvas.Canvas(str(target), pagesize=A4)
c.setTitle('Camera calibration - A4 checkerboard - 18 mm squares')
c.setFont('Helvetica-Bold', 16)
c.drawCentredString(105*mm, 277*mm, 'CAMERA CALIBRATION TARGET')
c.setFont('Helvetica', 10)
c.drawCentredString(105*mm, 268*mm, 'A4 portrait | Print at ACTUAL SIZE / 100% | Disable Fit to page')
c.drawCentredString(105*mm, 259*mm, '10 x 7 squares | 9 x 6 inner corners | Nominal square: 18 mm')
# Vector rectangles: 180 x 126 mm, with a clear white surround.
for row in range(7):
    for col in range(10):
        if (row+col) % 2 == 0:
            c.rect((15+18*col)*mm, (110+18*row)*mm, 18*mm, 18*mm,
                   fill=1, stroke=0)
c.setFont('Helvetica', 10)
c.drawCentredString(105*mm, 96*mm, 'Mount the entire sheet flat on a rigid board. Keep the white margin.')
c.drawCentredString(105*mm, 89*mm, 'Use matte paper. Avoid folds, glare, glossy lamination and curved surfaces.')
c.setLineWidth(.6)
c.line(55*mm, 66*mm, 155*mm, 66*mm)
for x in (55,155):
    c.line(x*mm, 63*mm, x*mm, 69*mm)
c.drawCentredString(105*mm, 57*mm, 'VERIFY WITH A RULER: the distance between the end ticks is 100 mm.')
c.drawCentredString(105*mm, 49*mm, 'Also measure across 5 squares: nominal distance = 90 mm.')
c.drawCentredString(105*mm, 37*mm, 'If the print size differs, reprint at 100% or record the actual square size.')
c.drawCentredString(105*mm, 29*mm, 'Calibration software must use the measured size, not the nominal label.')
c.showPage()
c.save()
print(target)
