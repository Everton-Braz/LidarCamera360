"""Release banner from real app captures (and optional real footage stills).

Usage:
  python promo/tools/make_banner.py --version v0.4.0 --config promo/v0.4.0/banner.json

The config names the headline, the feature chips, the UI capture and an optional
hero still (a 3DGS / point cloud frame). Outputs land next to the config:
  banner_1920x1080.jpg  (GitHub release, YouTube, master)
  banner_1200x675.jpg   (X, LinkedIn and Facebook image posts)
  banner_1080x1350.jpg  (LinkedIn / Facebook / Instagram portrait feed)
"""
import argparse
import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[2]
BG = (13, 16, 21)
ACCENT = (34, 227, 240)
TEXT = (240, 244, 248)
MUTED = (150, 165, 182)
CARD = (27, 32, 40)


def font(name, size):
    for cand in (name, name.lower()):
        try:
            return ImageFont.truetype(cand, size)
        except OSError:
            continue
    return ImageFont.load_default()


def F(weight, size):
    return font({'bold': 'segoeuib.ttf', 'semi': 'seguisb.ttf', 'reg': 'segoeui.ttf'}[weight], size)


def dot_field(w, h, seed=7):
    """Decorative scan-line dot field (deterministic). Not data; just texture."""
    img = Image.new('RGB', (w, h), BG)
    d = ImageDraw.Draw(img)
    rnd = random.Random(seed)
    cx, cy = w * 0.72, h * 1.15
    for ring in range(38):
        r = 140 + ring * ring * 1.05
        n = int(60 + ring * 9)
        for k in range(n):
            a = math.pi + (k / n) * math.pi + rnd.uniform(-0.004, 0.004)
            x, y = cx + r * math.cos(a), cy + r * math.sin(a) * 0.42
            if 0 <= x < w and 0 <= y < h:
                fade = max(0.0, 1 - ring / 38)
                c = tuple(int(BG[i] + (ACCENT[i] - BG[i]) * 0.55 * fade) for i in range(3))
                d.point((x, y), fill=c)
    return img


def cover(img, w, h):
    s = max(w / img.width, h / img.height)
    img = img.resize((round(img.width * s), round(img.height * s)), Image.Resampling.LANCZOS)
    x, y = (img.width - w) // 2, (img.height - h) // 2
    return img.crop((x, y, x + w, y + h))


def rounded(img, radius):
    mask = Image.new('L', img.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, *img.size), radius=radius, fill=255)
    out = Image.new('RGBA', img.size)
    out.paste(img, (0, 0), mask)
    return out


def shadow(canvas, box, radius, blur=40, alpha=170):
    sh = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle(box, radius=radius, fill=(0, 0, 0, alpha))
    canvas.alpha_composite(sh.filter(ImageFilter.GaussianBlur(blur)))


def background(cfg, w, h, base):
    hero = cfg.get('hero')
    if hero and (base / hero).is_file():
        img = cover(Image.open(base / hero).convert('RGB'), w, h)
        # darken the left for the type
        grad = Image.new('L', (w, 1))
        for x in range(w):
            grad.putpixel((x, 0), int(235 * max(0.0, 1 - x / (w * 0.62)) + 60))
        overlay = Image.new('RGB', (w, h), BG)
        img = Image.composite(overlay, img, grad.resize((w, h)))
        return img.convert('RGBA')
    return dot_field(w, h).convert('RGBA')


def chips(draw, x, y, items, size, max_w):
    f = F('semi', size)
    pad_x, pad_y, gap = int(size * 0.8), int(size * 0.45), int(size * 0.55)
    cx, cy = x, y
    line_h = size + 2 * pad_y + gap
    for text in items:
        tw = draw.textlength(text, font=f)
        bw = tw + 2 * pad_x
        if cx + bw > x + max_w:
            cx, cy = x, cy + line_h
        draw.rounded_rectangle((cx, cy, cx + bw, cy + size + 2 * pad_y), radius=size, fill=(22, 44, 52),
                               outline=(40, 120, 132), width=2)
        draw.text((cx + pad_x, cy + pad_y - size * 0.12), text, font=f, fill=TEXT)
        cx += bw + gap
    return cy + line_h


def landscape(cfg, version, base, w, h):
    k = w / 1920
    canvas = background(cfg, w, h, base)
    draw = ImageDraw.Draw(canvas)
    m = int(96 * k)

    logo = Image.open(ROOT / 'assets/logo.png').convert('RGBA').resize((int(92 * k),) * 2, Image.Resampling.LANCZOS)
    canvas.alpha_composite(logo, (m, int(88 * k)))
    draw.text((m + int(112 * k), int(96 * k)), 'LidarCamera360', font=F('semi', int(46 * k)), fill=TEXT)
    vf = F('bold', int(30 * k))
    vx = m + int(112 * k)
    vy = int(152 * k)
    vw = draw.textlength(version, font=vf)
    draw.rounded_rectangle((vx, vy, vx + vw + 28 * k, vy + 46 * k), radius=int(10 * k), fill=ACCENT)
    draw.text((vx + 14 * k, vy + 3 * k), version, font=vf, fill=BG)

    y = int(290 * k)
    col = int(760 * k)
    size = int(84 * k)
    while size > 20 and max(draw.textlength(t, font=F('bold', size)) for t in cfg['headline']) > col:
        size -= 2
    for line in cfg['headline']:
        draw.text((m, y), line, font=F('bold', size), fill=TEXT)
        y += int(size * 1.18)
    y += int(16 * k)
    for line in cfg['subline']:
        draw.text((m, y), line, font=F('reg', int(32 * k)), fill=MUTED)
        y += int(44 * k)
    y += int(34 * k)
    chips(draw, m, y, cfg['chips'], int(25 * k), int(820 * k))

    shot = Image.open(base / cfg['ui']).convert('RGB')
    if cfg.get('ui_crop'):
        shot = shot.crop(tuple(cfg['ui_crop']))
    sw = int(960 * k)
    sh_ = int(shot.height * sw / shot.width)
    sh_ = min(sh_, int(680 * k))
    shot = cover(shot, sw, sh_)
    sx, sy = w - sw - int(60 * k), int((h - sh_) / 2 + 40 * k)
    shadow(canvas, (sx, sy + 20 * k, sx + sw, sy + sh_ + 20 * k), int(18 * k), blur=int(40 * k))
    canvas.alpha_composite(rounded(shot, int(16 * k)), (sx, sy))
    ImageDraw.Draw(canvas).rounded_rectangle((sx, sy, sx + sw, sy + sh_), radius=int(16 * k),
                                             outline=(70, 82, 98), width=max(1, int(2 * k)))
    if cfg.get('ui_caption'):
        draw.text((sx, sy + sh_ + int(18 * k)), cfg['ui_caption'], font=F('reg', int(22 * k)), fill=MUTED)

    draw.text((m, h - int(76 * k)), cfg['footer'], font=F('semi', int(24 * k)), fill=ACCENT)
    return canvas.convert('RGB')


def portrait(cfg, version, base, w=1080, h=1350):
    canvas = background(cfg, w, h, base)
    draw = ImageDraw.Draw(canvas)
    m = 72
    logo = Image.open(ROOT / 'assets/logo.png').convert('RGBA').resize((80, 80), Image.Resampling.LANCZOS)
    canvas.alpha_composite(logo, (m, 70))
    draw.text((m + 100, 74), 'LidarCamera360', font=F('semi', 40), fill=TEXT)
    vf = F('bold', 26)
    vw = draw.textlength(version, font=vf)
    draw.rounded_rectangle((m + 100, 124, m + 100 + vw + 24, 164), radius=8, fill=ACCENT)
    draw.text((m + 112, 126), version, font=vf, fill=BG)
    y = 230
    for line in cfg['headline']:
        draw.text((m, y), line, font=F('bold', 72), fill=TEXT)
        y += 86
    y += 20
    y = chips(draw, m, y, cfg['chips'], 24, w - 2 * m) + 20
    shot = Image.open(base / cfg['ui']).convert('RGB')
    crop = cfg.get('ui_crop_portrait') or cfg.get('ui_crop')
    if crop:
        shot = shot.crop(tuple(crop))
    sw = w - 2 * m
    sh_ = min(h - y - 130, int(shot.height * sw / shot.width))
    shot = cover(shot, sw, sh_)
    shadow(canvas, (m, y + 16, m + sw, y + sh_ + 16), 16)
    canvas.alpha_composite(rounded(shot, 14), (m, y))
    draw.text((m, h - 80), cfg['footer'], font=F('semi', 24), fill=ACCENT)
    return canvas.convert('RGB')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', required=True)
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg_path = Path(args.config)
    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    base = cfg_path.parent
    out = base / 'banner'
    out.mkdir(exist_ok=True)
    master = landscape(cfg, args.version, base, 1920, 1080)
    master.save(out / 'banner_1920x1080.jpg', quality=93)
    landscape(cfg, args.version, base, 1200, 675).save(out / 'banner_1200x675.jpg', quality=93)
    portrait(cfg, args.version, base).save(out / 'banner_1080x1350.jpg', quality=93)
    for p in sorted(out.glob('banner_*.jpg')):
        print(p)


if __name__ == '__main__':
    main()
