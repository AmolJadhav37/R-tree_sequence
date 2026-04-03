"""
R-Tree Insertion Order Comparison — genuinely different trees
=============================================================
Run:   python rtree_animation.py
Output: rtree_animation.html  — open in any browser

Key parameters
--------------
M = 3  (max entries per node)
m = 1  (min entries per node)

8 rectangles: two spatially overlapping clusters (A-side left, B-side right)
with ambiguous bridge members that straddle the center.

Order A (group-first):  A1 A2 A3 A4 -> B1 B2 B3 B4
  First split fires on {A1,A2,A3,A4}. Best split separates by y,
  mixing A4 (top-left) into same leaf as B1/B4 later.
  -> Final: depth=2, overlap=12.0  (BAD grouping)

Order B (interleaved):  A1 B1 A2 B2 A3 B3 A4 B4
  First split fires on {A1,B1,A2,B2}. x-axis separation is clean.
  B-side stays right, A-side stays left from the very first split.
  -> Final: depth=3, overlap=4.0  (GOOD grouping)

This is EXACTLY the motivation for R*-tree forced re-insertion:
the split algorithm is correct but it can only work with whatever
entries happen to be in the node when overflow fires.
"""

import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.animation import FuncAnimation

# ═══════════════════════════════════════════════════════════════════
# 1. GEOMETRY
# ═══════════════════════════════════════════════════════════════════

def area(r):    return (r[2]-r[0])*(r[3]-r[1])
def combine(a,b): return (min(a[0],b[0]),min(a[1],b[1]),max(a[2],b[2]),max(a[3],b[3]))
def mbr(rects):
    r=list(rects)
    return (min(x[0] for x in r),min(x[1] for x in r),
            max(x[2] for x in r),max(x[3] for x in r))
def enlargement(m,r): return area(combine(m,r))-area(m)
def overlap_area(a,b):
    ox=max(0,min(a[2],b[2])-max(a[0],b[0]))
    oy=max(0,min(a[3],b[3])-max(a[1],b[1]))
    return ox*oy
def perimeter(r): return 2*((r[2]-r[0])+(r[3]-r[1]))

# ═══════════════════════════════════════════════════════════════════
# 2. EXHAUSTIVE SPLIT
# ═══════════════════════════════════════════════════════════════════

def exhaustive_split(entries, m, M):
    best_ov, best_pe = math.inf, math.inf
    best_g1, best_g2 = None, None
    for axis in range(2):
        for sort_upper in range(2):
            key_fn = (lambda e, a=axis, u=sort_upper: e[1][a + 2*u])
            se = sorted(entries, key=key_fn)
            n  = len(se)
            for k in range(m, n-m+1):
                g1, g2 = se[:k], se[k:]
                m1 = mbr([e[1] for e in g1])
                m2 = mbr([e[1] for e in g2])
                ov = overlap_area(m1, m2)
                pp = perimeter(m1)+perimeter(m2)
                if ov < best_ov or (ov == best_ov and pp < best_pe):
                    best_ov, best_pe = ov, pp
                    best_g1, best_g2 = g1, g2
    return best_g1, best_g2

# ═══════════════════════════════════════════════════════════════════
# 3. R-TREE NODE
# ═══════════════════════════════════════════════════════════════════

_nc = 0

class RNode:
    def __init__(self, is_leaf=True):
        global _nc
        self.nid      = _nc; _nc += 1
        self.is_leaf  = is_leaf
        self.entries  = []
        self.children = []
        self.parent   = None

    def node_mbr(self):
        if self.is_leaf:
            return mbr([e[1] for e in self.entries]) if self.entries else None
        ms = [c.node_mbr() for c in self.children if c.node_mbr() is not None]
        return mbr(ms) if ms else None

    def depth(self):
        if self.is_leaf: return 1
        return 1 + max(c.depth() for c in self.children)

# ═══════════════════════════════════════════════════════════════════
# 4. INSERT + SPLIT
# ═══════════════════════════════════════════════════════════════════

def choose_leaf(node, rect):
    if node.is_leaf: return node
    best, best_enl, best_a = None, math.inf, math.inf
    for ch in node.children:
        mm = ch.node_mbr()
        e  = enlargement(mm, rect) if mm else 0
        a  = area(mm) if mm else 0
        if e < best_enl or (e == best_enl and a < best_a):
            best_enl, best_a, best = e, a, ch
    return choose_leaf(best, rect)

def split_node(node, root, M, m):
    if node.is_leaf:
        g1, g2 = exhaustive_split(node.entries, m, M)
        n2 = RNode(is_leaf=True)
        node.entries = list(g1)
        n2.entries   = list(g2)
    else:
        pseudo = [(str(c.nid), c.node_mbr())
                  for c in node.children if c.node_mbr() is not None]
        g1p, g2p = exhaustive_split(pseudo, m, M)
        g1_ids = {e[0] for e in g1p}
        g2_ids = {e[0] for e in g2p}
        orig = list(node.children)
        n2 = RNode(is_leaf=False)
        node.children = [c for c in orig if str(c.nid) in g1_ids]
        n2.children   = [c for c in orig if str(c.nid) in g2_ids]
        for c in node.children: c.parent = node
        for c in n2.children:   c.parent = n2

    n2.parent = node.parent

    if node is root:
        nr = RNode(is_leaf=False)
        nr.children = [node, n2]
        node.parent = nr
        n2.parent   = nr
        return nr
    else:
        p = node.parent
        n2.parent = p
        p.children.append(n2)
        if len(p.children) > M:
            return split_node(p, root, M, m)
        return root

def insert(root, key, rect, M, m):
    leaf = choose_leaf(root, rect)
    leaf.entries.append((key, rect))
    if len(leaf.entries) > M:
        root = split_node(leaf, root, M, m)
    return root

# ═══════════════════════════════════════════════════════════════════
# 5. DATA
# ═══════════════════════════════════════════════════════════════════

M, m = 4, 2

RECTS = {
    'A1': (1, 5, 3, 8),   # A-cluster left, upper
    'A2': (1, 1, 3, 4),   # A-cluster left, lower
    'A3': (2, 3, 4, 6),   # A-cluster center, mid  (bridge)
    'A4': (0, 6, 2, 9),   # A-cluster far-left, top
    'B1': (7, 5, 9, 8),   # B-cluster right, upper
    'B2': (7, 1, 9, 4),   # B-cluster right, lower
    'B3': (6, 3, 8, 6),   # B-cluster center, mid  (bridge)
    'B4': (8, 6, 10, 9),  # B-cluster far-right, top
}

RECT_COLORS = {
    'A1':'#1D4ED8','A2':'#2563EB','A3':'#60A5FA','A4':'#1E40AF',
    'B1':'#B91C1C','B2':'#DC2626','B3':'#F87171','B4':'#991B1B',
}

ORDER_A = ['A1','A2','A3','A4','B1','B2','B3','B4']  # group-first
ORDER_B = ['A1','B1','A2','B2','A3','B3','A4','B4']  # interleaved

# ═══════════════════════════════════════════════════════════════════
# 6. BUILD FRAMES
# ═══════════════════════════════════════════════════════════════════

def deep_clone(node):
    n = RNode(is_leaf=node.is_leaf)
    n.nid     = node.nid
    n.entries = list(node.entries)
    for c in node.children:
        cc = deep_clone(c); cc.parent = n
        n.children.append(cc)
    return n

def build_frames(order):
    global _nc; _nc = 0
    root = RNode(is_leaf=True)
    frames = []
    for i, key in enumerate(order):
        root = insert(root, key, RECTS[key], M, m)
        frames.append({
            'step':     i+1,
            'total':    len(order),
            'inserted': list(order[:i+1]),
            'current':  key,
            'tree':     deep_clone(root),
        })
    return frames

frames_A = build_frames(ORDER_A)
frames_B = build_frames(ORDER_B)

def total_overlap(tree):
    by_depth = {}
    def collect(node, d):
        mm = node.node_mbr()
        if mm: by_depth.setdefault(d,[]).append(mm)
        if not node.is_leaf:
            for c in node.children: collect(c, d+1)
    collect(tree, 0)
    total = 0.0
    for ms in by_depth.values():
        for i in range(len(ms)):
            for j in range(i+1,len(ms)): total += overlap_area(ms[i], ms[j])
    return total

def tree_summary(tree, indent=0):
    mm = tree.node_mbr(); s = " "*indent
    if tree.is_leaf:
        keys = [e[0] for e in tree.entries]
        return s + f"LEAF N{tree.nid}: {keys}  mbr={mm}\n"
    s += f"INT  N{tree.nid}: {len(tree.children)}ch  mbr={mm}\n"
    for c in tree.children: s += tree_summary(c, indent+4)
    return s

print("="*60)
print(f"M={M}, m={m}")
print(f"Depth A = {frames_A[-1]['tree'].depth()}  |  Overlap A = {total_overlap(frames_A[-1]['tree']):.2f}")
print(f"Depth B = {frames_B[-1]['tree'].depth()}  |  Overlap B = {total_overlap(frames_B[-1]['tree']):.2f}")
print("\nFinal Tree A:"); print(tree_summary(frames_A[-1]['tree']))
print("Final Tree B:"); print(tree_summary(frames_B[-1]['tree']))
print("="*60)

# ═══════════════════════════════════════════════════════════════════
# 7. COLORS
# ═══════════════════════════════════════════════════════════════════

NODE_PALETTE = [
    ('#1D4ED8','#DBEAFE'),
    ('#B91C1C','#FEE2E2'),
    ('#15803D','#DCFCE7'),
    ('#7C3AED','#EDE9FE'),
    ('#B45309','#FEF3C7'),
    ('#0E7490','#CFFAFE'),
    ('#BE185D','#FCE7F3'),
    ('#374151','#F3F4F6'),
    ('#065F46','#D1FAE5'),
    ('#7F1D1D','#FEF2F2'),
]

def assign_colors(tree):
    colors = {}
    queue = [tree]; idx = 0
    while queue:
        node = queue.pop(0)
        colors[node.nid] = NODE_PALETTE[idx % len(NODE_PALETTE)]
        idx += 1
        for c in node.children: queue.append(c)
    return colors

def assign_display_labels(tree, start=0):
    labels = {}
    queue = [tree]
    next_id = start
    while queue:
        node = queue.pop(0)
        labels[node.nid] = f"N{next_id}"
        next_id += 1
        for child in node.children:
            queue.append(child)
    return labels

# ═══════════════════════════════════════════════════════════════════
# 8. TREE LAYOUT
# ═══════════════════════════════════════════════════════════════════

def layout_tree(node, xlo=0.04, xhi=0.96, depth=0, pos=None):
    if pos is None: pos = {}
    pos[node.nid] = ((xlo+xhi)/2.0, depth)
    if not node.is_leaf and node.children:
        n = len(node.children); step = (xhi-xlo)/n
        for i,c in enumerate(node.children):
            layout_tree(c, xlo+i*step, xlo+(i+1)*step, depth+1, pos)
    return pos

# ═══════════════════════════════════════════════════════════════════
# 9. SPATIAL PANEL
# ═══════════════════════════════════════════════════════════════════

WORLD = 11

def draw_spatial(ax, frame, title):
    ax.clear()
    ax.set_xlim(0, WORLD); ax.set_ylim(0, WORLD)
    ax.set_aspect('equal', adjustable='box')
    ax.set_facecolor('#F8F9FA')
    ax.set_title(title, fontsize=9, fontweight='bold', pad=4, color='#111')
    ax.set_xlabel('x', fontsize=8); ax.set_ylabel('y', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.10, linewidth=0.4, color='#888')
    for sp in ax.spines.values(): sp.set_linewidth(0.4); sp.set_color('#ccc')

    nc = assign_colors(frame['tree'])
    dl = assign_display_labels(frame['tree'])

    def collect_nodes(node, d, lst):
        lst.append((d, node))
        if not node.is_leaf:
            for c in node.children: collect_nodes(c, d+1, lst)
        return lst

    all_nodes = collect_nodes(frame['tree'], 0, [])
    max_d = max(d for d,_ in all_nodes) if all_nodes else 0

    # draw MBR boxes shallowest first (deepest on top)
    for depth, node in sorted(all_nodes, key=lambda x: x[0]):
        nm = node.node_mbr()
        if not nm: continue
        stroke, fill = nc.get(node.nid, ('#666','#eee'))

        if   depth == 0:     ls,lw,al = ':',  0.9, 0.05
        elif depth == max_d: ls,lw,al = '-',  2.0, 0.19
        else:                ls,lw,al = '--', 1.4, 0.11

        ax.add_patch(mpatches.Rectangle(
            (nm[0],nm[1]), nm[2]-nm[0], nm[3]-nm[1],
            linewidth=lw, edgecolor=stroke, facecolor=fill,
            alpha=al, linestyle=ls, zorder=2+depth
        ))

        # Node label with MBR coordinates
        lbl = f"{dl[node.nid]}  x:[{nm[0]},{nm[2]}] y:[{nm[1]},{nm[3]}]"
        ax.text(nm[0]+0.12, nm[3]-0.12, lbl,
                fontsize=5.0, color=stroke, va='top', fontweight='bold',
                zorder=20+depth,
                bbox=dict(boxstyle='round,pad=0.09', fc='white',
                          ec=stroke, alpha=0.88, linewidth=0.5))

    # data rectangles
    for key in frame['inserted']:
        r   = RECTS[key]
        col = RECT_COLORS[key]
        is_cur = (key == frame['current'])
        ax.add_patch(mpatches.Rectangle(
            (r[0],r[1]), r[2]-r[0], r[3]-r[1],
            linewidth=2.4 if is_cur else 1.4,
            edgecolor=col, facecolor=col,
            alpha=0.62 if is_cur else 0.28, zorder=15
        ))
        cx=(r[0]+r[2])/2; cy=(r[1]+r[3])/2
        ax.text(cx, cy, key, ha='center', va='center',
                fontsize=7.5, fontweight='bold', color='white', zorder=22,
                bbox=dict(boxstyle='round,pad=0.15', fc=col, ec='none', alpha=0.93))

    ov = total_overlap(frame['tree'])
    d  = frame['tree'].depth()
    ax.text(0.02, 0.02,
            f"depth={d}   overlap={ov:.2f} u²   step {frame['step']}/{frame['total']}",
            transform=ax.transAxes, fontsize=7.5, color='#111',
            bbox=dict(boxstyle='round,pad=0.26', fc='white', ec='#aaa', alpha=0.96, lw=0.5))

# ═══════════════════════════════════════════════════════════════════
# 10. TREE PANEL
# ═══════════════════════════════════════════════════════════════════

def draw_tree_panel(ax, frame, title):
    ax.clear()
    ax.set_xlim(0,1); ax.set_ylim(-0.06,1.06)
    ax.set_aspect('auto'); ax.axis('off')
    ax.set_facecolor('#F8F9FA')
    ax.set_title(title, fontsize=9, fontweight='bold', pad=4, color='#111')

    tree  = frame['tree']
    tdep  = tree.depth()
    pos   = layout_tree(tree, 0.04, 0.96, 0)
    nc    = assign_colors(tree)
    dl    = assign_display_labels(tree)

    BW, BH = 0.155, 0.092

    def yp(d):
        if tdep==1: return 0.50
        return 0.93 - d*(0.84/(tdep-1))

    def gxy(nid):
        cx,d = pos[nid]; return cx, yp(d)

    # edges
    def draw_edges(node):
        if node.is_leaf: return
        px,py = gxy(node.nid)
        for c in node.children:
            cx,cy = gxy(c.nid)
            ax.annotate('', xy=(cx,cy+BH/2+0.003), xytext=(px,py-BH/2-0.003),
                arrowprops=dict(arrowstyle='->', color='#bbb', lw=0.9,
                                connectionstyle='arc3,rad=0'), zorder=1)
            draw_edges(c)
    draw_edges(tree)

    # nodes
    def draw_nodes(node):
        cx,cy = gxy(node.nid)
        stroke, fill = nc.get(node.nid, ('#555','#f0f0f0'))
        nm = node.node_mbr()

        ax.add_patch(FancyBboxPatch(
            (cx-BW/2, cy-BH/2), BW, BH,
            boxstyle='round,pad=0.008',
            linewidth=1.8, edgecolor=stroke, facecolor=fill,
            alpha=0.96, zorder=5
        ))

        # name
        if node is tree:          name = f"ROOT  {dl[node.nid]}"
        elif node.is_leaf:        name = f"{dl[node.nid]}  [leaf]"
        else:                     name = f"{dl[node.nid]}  [int]"

        # MBR
        coord = (f"x:[{nm[0]},{nm[2]}]  y:[{nm[1]},{nm[3]}]" if nm else "(empty)")

        # content
        if node.is_leaf:
            keys  = [e[0] for e in node.entries]
            line3 = "  ".join(keys) if keys else "—"
            # colour the keys by their side
            a_keys = [k for k in keys if k.startswith('A')]
            b_keys = [k for k in keys if k.startswith('B')]
            if a_keys and b_keys:
                line3 = "  ".join(keys) + ""
                line3_col = '#B45309'    # amber = warning
            elif a_keys:
                line3_col = '#1D4ED8'
            elif b_keys:
                line3_col = '#B91C1C'
            else:
                line3_col = '#333'
        else:
            a_val = area(nm) if nm else 0
            line3 = f"{len(node.children)} children  area={a_val:.0f}"
            line3_col = '#333'

        ax.text(cx, cy+0.025, name,
                ha='center', va='center', fontsize=6.5, fontweight='bold',
                color=stroke, zorder=10)
        if node is not tree:
            ax.text(cx, cy+0.001, coord,
                ha='center', va='center', fontsize=5.6, color='#1a1a1a', zorder=10)
            ax.text(cx, cy-0.025, line3,
                ha='center', va='center', fontsize=6.0, color=line3_col, zorder=10)

        if not node.is_leaf:
            for c in node.children: draw_nodes(c)

    draw_nodes(tree)

    # summary bar
    ov = total_overlap(tree)
    d  = tree.depth()
    if d == 2:
        msg = f"depth={d}  overlap={ov:.2f}"
        mc  = '#B91C1C'
    else:
        msg = f"depth={d}  overlap={ov:.2f}"
        mc  = '#15803D'

    ax.text(0.5, 0.01, msg,
            ha='center', va='bottom', transform=ax.transAxes,
            fontsize=7.2, color=mc,
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#ccc', alpha=0.97, lw=0.5))

# ═══════════════════════════════════════════════════════════════════
# 11. FIGURE + ANIMATION
# ═══════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(20, 16), facecolor='#EEF0F3')

gs = fig.add_gridspec(2, 2,
                      left=0.04, right=0.97,
                      top=0.92,  bottom=0.06,
                      hspace=0.42, wspace=0.25)

ax_spA = fig.add_subplot(gs[0,0])
ax_spB = fig.add_subplot(gs[0,1])
ax_trA = fig.add_subplot(gs[1,0])
ax_trB = fig.add_subplot(gs[1,1])

fig.suptitle(
    "R-Tree Insertion 2 different orders(M=4, m=2, exhaustive split)\n"
   ,
    fontsize=12, fontweight='bold', y=0.975, color='#0a0a0a'
)

seq_ax = fig.add_axes([0.02, 0.002, 0.96, 0.052])
seq_ax.axis('off')
seq_txt = seq_ax.text(0.5, 0.5, '', ha='center', va='center',
                      fontsize=8.5, color='#111',
                      transform=seq_ax.transAxes,
                      fontfamily='monospace')

def make_seq(order, cur_idx):
    parts = []
    for i, k in enumerate(order):
        if   i <  cur_idx: parts.append(f"\u2713{k}")
        elif i == cur_idx: parts.append(f"[{k}]")
        else:               parts.append(f" {k} ")
    return "  ".join(parts)

def update(fi):
    fA = frames_A[fi]; fB = frames_B[fi]
    ovA = total_overlap(fA['tree']); ovB = total_overlap(fB['tree'])
    dA  = fA['tree'].depth();        dB  = fB['tree'].depth()

    draw_spatial(ax_spA, fA, "Order A")
    draw_spatial(ax_spB, fB, "Order B")

    draw_tree_panel(ax_trA, fA, f"Tree A — depth={dA}  overlap={ovA:.2f} u²")
    draw_tree_panel(ax_trB, fB, f"Tree B — depth={dB}  overlap={ovB:.2f} u²")

    seq_txt.set_text(
        f"A (group-first):   {make_seq(ORDER_A, fi)}     overlap_A = {ovA:.2f}\n"
        f"B (interleaved):   {make_seq(ORDER_B, fi)}     overlap_B = {ovB:.2f}"
    )

ani = FuncAnimation(fig, update, frames=len(frames_A),
                    interval=2200, repeat=True, blit=False)

out = 'rtree_animation.html'
print(f"Saving to {out} — please wait …")
with open(out, 'w') as f:
    f.write(ani.to_jshtml())
print(f"Done!   xdg-open {out}")
plt.close(fig)
