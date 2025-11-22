import sys
from PIL import Image

if __name__ == "__main__":
    direction = sys.argv[1] # vertical or horizontal
    imgs = [Image.open(p) for p in sys.argv[2:-1]]
    out_path = sys.argv[-1]

    if direction == "vertical":
        w = max(im.width for im in imgs)
        h = sum(im.height for im in imgs)
    else:
        w = sum(im.width for im in imgs)
        h = max(im.height for im in imgs)

    out = Image.new("RGBA", (w, h))

    offset = 0
    for im in imgs:
        if direction == "vertical":
            out.paste(im, (0, offset))
            offset += im.height
        else:
            out.paste(im, (offset, 0))
            offset += im.width

    out.save(out_path)
    print(f"Saved to {out_path}")