import sys
from PIL import Image

if __name__ == "__main__":
    imgs = [Image.open(p) for p in sys.argv[1:-1]]
    out_path = sys.argv[-1]

    w = max(im.width for im in imgs)
    h = sum(im.height for im in imgs)

    out = Image.new("RGBA", (w, h))

    y = 0
    for im in imgs:
        out.paste(im, (0, y))
        y += im.height

    out.save(out_path)
    print(f"Saved to {out_path}")