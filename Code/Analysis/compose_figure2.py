"""Combine the deposited Figure 2a-c layout with calculated panels d-e."""
from pathlib import Path
import argparse
import json
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def compose(base_path, panel_path, layout_path, destination):
    with Image.open(base_path) as image:
        base = image.convert('RGBA')
    with Image.open(panel_path) as image:
        panels = image.convert('RGBA')
    layout = json.loads(Path(layout_path).read_text(encoding='utf-8'))
    if list(base.size) != layout['canvas_size_px']:
        raise ValueError('Figure 2 base dimensions do not match the supplied layout.')
    x, y, width, height = layout['panels_de_box_xywh']
    if min(x, y) < 0 or min(width, height) <= 0 or x + width > base.width or y + height > base.height:
        raise ValueError('Panels d-e must lie within the base canvas.')
    canvas = Image.new('RGBA', base.size, 'white')
    canvas.alpha_composite(base)
    panels.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas.alpha_composite(panels, (x + (width - panels.width) // 2, y + (height - panels.height) // 2))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert('RGB').save(destination, dpi=(350, 350))
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='Source_Data/Figure_Plotting/Figure2_abc.png')
    parser.add_argument('--layout', default='Source_Data/Figure_Plotting/Figure2_layout.json')
    parser.add_argument('--panels', default=None, help='Default: calculate Figure2_de.png in the output directory.')
    parser.add_argument('--out', default='ReviewChecks/figures/Figure2_composed.png')
    args = parser.parse_args()
    destination = project_path(args.out)
    base = project_path(args.base)
    layout = project_path(args.layout)
    if not base.is_file() or not layout.is_file():
        raise FileNotFoundError('Prepare the review inputs to obtain Figure2_abc.png and Figure2_layout.json.')
    if args.panels is None:
        from plot_figure2_de import main as plot_panels
        plot_panels(destination.parent)
        panels = destination.parent / 'Figure2_de.png'
    else:
        panels = project_path(args.panels)
    print(compose(base, panels, layout, destination))


if __name__ == '__main__':
    main()
