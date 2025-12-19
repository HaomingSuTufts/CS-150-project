import pickle
import networkx as nx
import rdkit.Chem as Chem

with open(
    "samples\QM9\gdss_qm9_training\QM9-inpaint_gdss_qm9_training_best_epoch_802_128.pkl",
    "rb",
) as f:
    data = pickle.load(f)


# convert the networkx graph to picture
print(data["graphs"][0])
print(data["smiles"][0])


def graph_to_image(
    graph,
    filename="graph.png",
    figsize=(6, 6),
    dpi=300,
    pixels=None,
    seed=42,
    show_labels=False,  # hide node-type notation by default
    node_scale=0.6,  # global node size scale (smaller than before)
):
    import matplotlib.pyplot as plt
    import numpy as np

    # If pixels specified, convert to figsize so saved PNG has exact pixel size
    if pixels is not None:
        w_px, h_px = pixels
        figsize = (w_px / dpi, h_px / dpi)
    else:
        w_px, h_px = int(figsize[0] * dpi), int(figsize[1] * dpi)

    plt.figure(figsize=figsize, dpi=dpi)
    pos = nx.spring_layout(graph, seed=seed)

    # node sizes (fallback to smaller default) and labels
    base_sizes = [
        graph.nodes[n].get("size", 150) for n in graph.nodes()
    ]  # smaller default
    # scale node sizes with image height so nodes look consistent across sizes
    scale = max(0.25, h_px / 300.0)
    sizes = [max(6, int(s * scale * node_scale)) for s in base_sizes]
    labels = {}
    if show_labels:
        labels = {n: graph.nodes[n].get("label", "") for n in graph.nodes()}

    # node colors: try numeric attribute (e.g. atomic_num), otherwise single color
    raw_colors = [
        graph.nodes[n].get("atomic_num", graph.nodes[n].get("color", 0))
        for n in graph.nodes()
    ]
    try:
        vals = np.array(raw_colors, dtype=float)
        norm = plt.Normalize(
            vmin=vals.min(),
            vmax=(vals.max() if vals.max() != vals.min() else vals.min() + 1),
        )
        cmap = plt.cm.viridis
        node_colors = cmap(norm(vals))
    except Exception:
        node_colors = "lightblue"

    # edge widths from 'weight' attribute, scaled with image size (thinner)
    edge_widths = [
        max(0.3, d.get("weight", 1.0) * (scale * 0.35))
        for (_, _, d) in graph.edges(data=True)
    ]

    ax = plt.gca()
    ax.set_facecolor("white")
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=sizes,
        node_color=node_colors,
        edgecolors="k",
        linewidths=0.4,  # thinner contours
    )
    nx.draw_networkx_edges(graph, pos, width=edge_widths, alpha=0.8, edge_color="gray")
    font_size = max(4, int(h_px / 70))  # smaller font when enabled
    if show_labels:
        nx.draw_networkx_labels(graph, pos, labels, font_size=font_size)
    ax.axis("off")
    plt.tight_layout(pad=0.1)
    plt.savefig(filename, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def smiles_to_image(smiles, filename="molecule.png", size=(400, 400), dpi=300):
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit import Chem
    from PIL import Image
    import io

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES: %r" % smiles)

    # Use MolDraw2DCairo for high-quality drawing
    w, h = size
    drawer = rdMolDraw2D.MolDraw2DCairo(w, h)
    opts = drawer.drawOptions()
    opts.addAtomIndices = False
    opts.bondLineWidth = 1.6
    # atomLabelFontSize isn't available on all RDKit builds; fall back to fontSize
    try:
        opts.atomLabelFontSize = 12
    except AttributeError:
        try:
            opts.fontSize = max(8, int(h / 40))
        except AttributeError:
            pass

    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    png_bytes = drawer.GetDrawingText()
    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    img.save(filename, dpi=(dpi, dpi))


def graph_and_smiles(
    graph, smiles, filename="combined.png", figsize=(10, 5), dpi=300, seed=42
):
    import matplotlib.pyplot as plt
    from PIL import Image
    import io, os, tempfile

    # target pixel sizes
    total_w_px = int(figsize[0] * dpi)
    total_h_px = int(figsize[1] * dpi)
    half_w_px = total_w_px // 2

    # create temp files then combine
    with tempfile.TemporaryDirectory() as tmp:
        gfile = os.path.join(tmp, "g.png")
        mfile = os.path.join(tmp, "m.png")

        # generate images with the same height and equal widths
        graph_to_image(
            graph,
            filename=gfile,
            pixels=(half_w_px, total_h_px),
            dpi=dpi,
            seed=seed,
        )
        smiles_to_image(
            smiles,
            filename=mfile,
            size=(half_w_px, total_h_px),
            dpi=dpi,
        )

        gimg = Image.open(gfile).convert("RGBA")
        mimg = Image.open(mfile).convert("RGBA")

        # combine side-by-side and vertically center each image
        total_w = gimg.width + mimg.width
        h = max(gimg.height, mimg.height)
        combined = Image.new("RGBA", (total_w, h), (255, 255, 255, 255))
        g_y = (h - gimg.height) // 2
        m_y = (h - mimg.height) // 2
        combined.paste(gimg, (0, g_y), gimg)
        combined.paste(mimg, (gimg.width, m_y), mimg)
        combined.save(filename, dpi=(dpi, dpi))


graph_and_smiles(
    data["graphs"][2],
    data["smiles"][2],
    filename="graph_and_molecule.png",
    figsize=(10, 5),
    dpi=300,
    seed=42,
)
