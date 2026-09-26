"""
Comparison figure: depth-slice maps vs. reference study (Arango)
=================================================================

Places the already-rendered depth-slice figure of our model side by side with
the reference depth-slice maps (grayscale image) from the reference study, so
both sets of horizontal sections can be compared directly.

Inputs:
    C:\\Users\\alber\\TFG\\visualizations\\depth_slices_figure.png   (this project)
    C:\\Users\\alber\\TFG\\visualizations\\Depth_Slices_Arango.png   (reference study)

Outputs (PNG @ 400 dpi + vector PDF):
    C:\\Users\\alber\\TFG\\visualizations\\compare_depth_slices_arango_figure.png
    C:\\Users\\alber\\TFG\\visualizations\\compare_depth_slices_arango_figure.pdf
"""
import os
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

OUR_IMG = r"C:\Users\alber\TFG\visualizations\depth_slices_figure.png"
REF_IMG = r"C:\Users\alber\TFG\visualizations\Depth_Slices_Arango.png"
OUT_PNG = r"C:\Users\alber\TFG\visualizations\compare_depth_slices_arango_figure.png"
OUT_PDF = r"C:\Users\alber\TFG\visualizations\compare_depth_slices_arango_figure.pdf"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "mathtext.default": "regular"})

our_img = mpimg.imread(OUR_IMG)
ref_img = mpimg.imread(REF_IMG)

# El nostre plot ara te una disposicio horitzontal (4+3, molt mes ample que
# alt) mentre que la imatge de referencia es vertical/estreta; es reparteix
# l'amplada de cada panell segons la relacio d'aspecte real de cada imatge
# perque cap de les dues quedi esprimida ni deixi massa espai en blanc.
ref_aspect = ref_img.shape[1] / ref_img.shape[0]
our_aspect = our_img.shape[1] / our_img.shape[0]

fig, (ax_ref, ax_our) = plt.subplots(
    1, 2, figsize=(19.0, 6.6),
    gridspec_kw=dict(width_ratios=[ref_aspect, our_aspect]))

ax_ref.imshow(ref_img)
ax_ref.set_axis_off()
ax_ref.set_title("(Arango Galván 2005)", fontsize=11, fontweight="bold", pad=6)

ax_our.imshow(our_img)
ax_our.set_axis_off()
ax_our.set_title("Model DM 4", fontsize=11, fontweight="bold", pad=6)

fig.suptitle("Comparació de les seccions horitzontals amb l'estudi de referència",
             fontsize=13.5, fontweight="bold", x=0.5, y=0.98)
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.90, wspace=0.03)

for path, dpi in ((OUT_PNG, 400), (OUT_PDF, None)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print("Saved:", path)
