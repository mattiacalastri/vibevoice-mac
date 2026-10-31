#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate the VibeVoice app icon: a recessed LED in a machined anthracite chassis.

Why a LED and not a waveform (sess.9757): the waveform icon this replaces failed
two of the four survival tests. At 16px seven thin bars with their gaps collapse
into a smudge, and "a sound wave on teal" describes fifty other dictation apps —
it named the category, not the product. It was also teal, which is *Astra
Digital's* colour, on an app that is not agency work.

A single point survives every reduction: a point stays a point at 16px, in
monochrome, and over a photograph. And it carries the actual claim — the pill
Mattia uses every day is rack hardware, and rack hardware means an instrument you
own and that sits on your desk, not a subscription that phones home. Whisper runs
locally in ~180ms; the metal is what says so.

Palette measured by `brand_palette_audit.py` (sess.9757): no dichromatic collapse,
LED-on to display 15.01:1.

What changed in sess.9767 — the mark is untouched, the MATERIAL is not. Next to
Obsidian in the Dock the old render read as "a dark square with a green dot":
correct, and inert. A product icon has to look like an object under a light, and
that is five passes the old build did not have:

  * supersampling ×2 — everything here is a circle; drawn straight at 1024 the
    bezel and the specular carried stair-stepping down into the 128px reduction
  * brushed metal — anisotropic streaks, amplitude ~3/255. Invisible as texture,
    visible as *material*: the plate stops being a flat fill and gains a surface
  * a machined boss — one turned ring around the well. It says the recess was
    cut, not printed, and it dies below ~32px instead of muddying it
  * light modelling — vignette, dome shading, floor bounce, bottom lip. The
    chassis now has a top, a middle and a bottom
  * a contact shadow under the body, like every macOS icon on Apple's grid —
    what makes the tile sit ON the Dock instead of hovering over it

Scars inherited from make_icon.py — do not regress:
  sess.9161  corners must be TRANSPARENT, or macOS wraps the icon in a grey rect
  sess.9203  full-bleed (margin 0); a transparent margin reads as grey padding
             in the Dock next to full-bleed neighbours — superseded by the
             BODY_RATIO note below, which is the same lesson measured properly

Usage:
  python3 make_icon_led.py --out VibeVoice_LED_1024.png
  python3 make_icon_led.py --state work --out preview_work.png
"""
from __future__ import annotations

import argparse

from PIL import Image, ImageChops, ImageDraw, ImageFilter

S = 1024
SS = 2  # supersampling factor: build at S*SS, land on S once, with LANCZOS
CORNER_RATIO = 0.2237  # Apple macOS squircle corner radius ratio

# Apple's macOS icon grid: the squircle body is 824pt inside a 1024pt canvas,
# i.e. ~80.5%. Full-bleed is NOT the safe default here (sess.9757): rendered at
# 100% the icon sits visibly LARGER than every system neighbour in the Dock,
# because they all respect this inset. The `--margin 0` in the older
# make_icon.py was a fix for a different symptom on a light, full-colour icon;
# on a dark chassis it just reads as oversized.
BODY_RATIO = 0.8047

# Chassis: lit from above, like a metal enclosure under room light.
CHASSIS_TOP = (44, 50, 68)     # #242938 lifted at the very top edge;
#              the gradient lands ON #242938 by the first third — the brand hex
#              is the plate colour, not the specular of the lamp on it
CHASSIS_BOT = (10, 12, 16)     # #0A0C10
WELL = (4, 5, 10)              # #04050A — the recess the LED sits in

# The three states the pill already speaks in. Same hex as vibevoice.py.
STATES = {
    "on":   (31, 255, 82),     # #1FFF52  listening / idle-armed
    "work": (255, 184, 13),    # #FFB80D  transcribing
    "mute": (242, 69, 69),     # #F24545  muted
}


def _squircle_mask(size: int, radius: int) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, size - 1, size - 1], radius=radius, fill=255)
    return mask


def _brushed(size: int, blur: float) -> Image.Image:
    """Horizontal brushed-metal streaks, as an L overlay centred on 128.

    Built as tall-thin noise and stretched sideways. A Gaussian blur is
    isotropic and would give grain, which reads as a JPEG artefact rather than a
    machined surface: the stretch is what makes it directional.
    """
    src = Image.effect_noise((max(2, size // 220), size), 26)
    return src.resize((size, size), Image.BILINEAR).filter(ImageFilter.GaussianBlur(blur))


def _chassis(size: int, radius: int) -> Image.Image:
    """The plate: vertical gradient, brushed surface, vignette."""
    grad = Image.new("RGBA", (size, size))
    for y in range(size):
        # ease-out so the light clings to the top third instead of ramping flat
        t = (y / size) ** 0.58
        grad.paste(
            tuple(int(a + (b - a) * t) for a, b in zip(CHASSIS_TOP, CHASSIS_BOT)) + (255,),
            (0, y, size, y + 1),
        )
    alpha = _squircle_mask(size, radius)
    rgb = grad.convert("RGB")

    # brushed metal: ±3/255 around neutral. The amplitude IS the point — one
    # step louder and it stops being a surface and starts being noise.
    tex = _brushed(size, size * 0.0009)
    rgb = ImageChops.overlay(rgb, Image.merge("RGB", (tex, tex, tex)))

    # vignette: hold the middle of the plate, let the corners fall away
    vig = _vignette(size)
    rgb = ImageChops.multiply(rgb, Image.merge("RGB", (vig, vig, vig)))

    return Image.merge("RGBA", (*rgb.split(), alpha))


def _vignette(size: int) -> Image.Image:
    """Radial falloff, bright at the centre — the light source is one lamp."""
    return Image.radial_gradient("L").resize((size, size), Image.BILINEAR).point(
        lambda v: int(255 - v * 0.17))


def _edge_fade(size: int, start: float, end: float) -> Image.Image:
    """Vertical alpha ramp used to make an outline die out along the flanks."""
    fade = Image.new("L", (size, size), 0)
    fd = ImageDraw.Draw(fade)
    span = max(1e-6, end - start)
    for y in range(size):
        t = (y / size - start) / span
        fd.line([(0, y), (size, y)], fill=int(255 * min(1.0, max(0.0, t)) ** 1.5))
    return fade


def build(state: str = "on") -> Image.Image:
    led = STATES[state]
    size = S * SS
    radius = int(size * CORNER_RATIO)

    img = _chassis(size, radius)

    cx = cy = size // 2
    r_led = int(size * 0.126)
    r_well = int(r_led * 1.28)
    r_boss = int(r_led * 1.62)

    # ── machined boss: one turned ring around the recess, proof that the well
    #    was CUT into the plate. Thin and low-contrast on purpose — it must
    #    vanish below ~32px rather than fight the point for attention. ───────
    boss = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    bo = ImageDraw.Draw(boss)
    bo.ellipse([cx - r_boss, cy - r_boss, cx + r_boss, cy + r_boss],
               outline=(0, 0, 0, 62), width=max(2, int(size * 0.0034)))
    lip_off = max(1, int(size * 0.0024))
    bo.ellipse([cx - r_boss, cy - r_boss + lip_off, cx + r_boss, cy + r_boss + lip_off],
               outline=(255, 255, 255, 15), width=max(2, int(size * 0.0026)))
    img.alpha_composite(boss.filter(ImageFilter.GaussianBlur(size * 0.0016)))

    # ── glow, deliberately tight. The earlier version reached 3× r_led and
    #    turned the whole chassis into green fog: the point stopped reading as
    #    a point. A real indicator LED bleeds barely past its own bezel, so the
    #    halo dies inside the well's shadow. ─────────────────────────────────
    glow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    for i, alpha in ((1.95, 20), (1.55, 38), (1.22, 60)):
        rr = int(r_led * i)
        gd.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=led + (alpha,))
    img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(size * 0.018)))

    # ── the recess: dark well, then ONE continuous bezel ring whose light
    #    varies around it. Two separate arcs read as two loose crescents. ────
    well = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    wd = ImageDraw.Draw(well)
    wd.ellipse([cx - r_well, cy - r_well, cx + r_well, cy + r_well], fill=WELL + (255,))
    img.alpha_composite(well.filter(ImageFilter.GaussianBlur(size * 0.004)))

    # the LED throws light back onto the floor of its own well: a faint coloured
    # ring just inside the bezel. Without it the well reads as a printed hole.
    floor = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    fl = ImageDraw.Draw(floor)
    inner = max(2, int(size * 0.004))
    fl.ellipse([cx - r_well + inner, cy - r_well + inner,
                cx + r_well - inner, cy + r_well - inner],
               outline=led + (48,), width=max(2, int(size * 0.006)))
    img.alpha_composite(floor.filter(ImageFilter.GaussianBlur(size * 0.005)))

    ring = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    rd = ImageDraw.Draw(ring)
    w_ring = max(2, int(size * 0.012))
    box = [cx - r_well, cy - r_well, cx + r_well, cy + r_well]
    # one ring, swept in short segments: shadow at the top (the lip overhangs),
    # bounced light at the bottom, and a smooth handover on the flanks
    for a0 in range(0, 360, 6):
        t = (a0 - 90) % 360
        f = (1.0 - abs(180 - t) / 180.0)          # 0 at top, 1 at bottom
        if f < 0.5:
            k = (0.5 - f) * 2.0
            col = (0, 0, 0, int(30 + 165 * k))
        else:
            k = (f - 0.5) * 2.0
            col = (255, 255, 255, int(14 + 78 * k))
        rd.arc(box, start=a0, end=a0 + 7, fill=col, width=w_ring)
    img.alpha_composite(ring.filter(ImageFilter.GaussianBlur(size * 0.0026)))

    # ── the lens: saturated colour almost all the way out, with only a modest
    #    lift near the middle. The old exponent blew the core to white and that
    #    white blob became a SECOND highlight competing with the specular. The
    #    `shade` term is new: a dome curves away from the lamp at its edge, and
    #    without that falloff the lens is a flat disc of colour. ─────────────
    lens = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ld = ImageDraw.Draw(lens)
    steps = 96
    for i in range(steps, 0, -1):
        t = i / steps
        rr = max(1, int(r_led * t))
        k = max(0.0, 1.0 - t) ** 3.6 * 0.42       # gentle, never reaches white
        shade = 1.0 - 0.20 * (t ** 3.0)
        col = tuple(min(255, int((c + (255 - c) * k) * shade)) for c in led)
        ld.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=col + (255,))
    img.alpha_composite(lens.filter(ImageFilter.GaussianBlur(size * 0.0014)))

    # dome rim: a hairline along the lower-right edge of the lens — the tell
    # that the surface is curved glass and not a disc
    rim = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    rmd = ImageDraw.Draw(rim)
    edge = max(1, int(size * 0.0016))
    rmd.arc([cx - r_led + edge, cy - r_led + edge, cx + r_led - edge, cy + r_led - edge],
            start=15, end=150, fill=(255, 255, 255, 86), width=max(2, int(size * 0.0034)))
    img.alpha_composite(rim.filter(ImageFilter.GaussianBlur(size * 0.0030)))

    # the one specular highlight — small, upper-left, unmistakably glass
    spec = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    sd = ImageDraw.Draw(spec)
    sr = int(r_led * 0.26)
    sx, sy = cx - int(r_led * 0.36), cy - int(r_led * 0.38)
    sd.ellipse([sx - sr, sy - int(sr * 0.82), sx + sr, sy + int(sr * 0.82)],
               fill=(255, 255, 255, 186))
    img.alpha_composite(spec.filter(ImageFilter.GaussianBlur(size * 0.0042)))

    # ── top bevel: a hairline of light that FADES OUT down the flanks. Cutting
    #    it flat at the equator left two visible stubs on the sides. ─────────
    inset = int(size * 0.012)
    bevel = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(bevel).rounded_rectangle(
        [inset, inset, size - 1 - inset, size - 1 - inset],
        radius=radius - inset, outline=(255, 255, 255, 44), width=max(2, int(size * 0.0038)))
    top_fade = ImageChops.invert(_edge_fade(size, 0.0, 0.55))
    bevel.putalpha(Image.composite(bevel.getchannel("A"),
                                   Image.new("L", (size, size), 0), top_fade))
    img.alpha_composite(bevel)

    # bottom lip: the plate has thickness, so its lower edge catches the bounce
    lip = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(lip).rounded_rectangle(
        [inset, inset, size - 1 - inset, size - 1 - inset],
        radius=radius - inset, outline=(255, 255, 255, 28), width=max(2, int(size * 0.0030)))
    lip.putalpha(Image.composite(lip.getchannel("A"), Image.new("L", (size, size), 0),
                                 _edge_fade(size, 0.60, 1.0)))
    img.alpha_composite(lip)

    # re-cut the squircle: every composite above painted past the rounded edge
    body = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    body.paste(img, (0, 0), _squircle_mask(size, radius))

    # sit the body on Apple's grid. Drawn at full size and scaled down once, so
    # the LED, the bevel hairline and the bezel keep their proportions instead
    # of being re-derived at a smaller radius.
    side = int(S * BODY_RATIO)
    pos = (S - side) // 2
    tile = body.resize((side, side), Image.LANCZOS)

    # contact shadow: what makes the tile sit ON the Dock instead of hovering
    # over it. Every system icon on Apple's grid carries one; ours did not, and
    # side by side that absence read as "flat PNG" rather than "app".
    out = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    shadow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    shadow.paste((0, 0, 0, 130), (pos, pos, pos + side, pos + side), tile.getchannel("A"))
    shadow = ImageChops.offset(shadow, 0, int(S * 0.013))
    out.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(S * 0.017)))
    out.alpha_composite(tile, (pos, pos))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="on", choices=sorted(STATES))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    build(a.state).save(a.out)
    print(f"wrote {a.out} (state {a.state})")
