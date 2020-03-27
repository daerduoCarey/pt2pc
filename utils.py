import os
import sys
import torch
import numpy as np
import importlib
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import linear_sum_assignment
import trimesh
from colors import colors
from torch_cluster import fps
from sklearn.metrics.pairwise import pairwise_distances


def printout(flog, strout):
    print(strout)
    flog.write(strout+'\n')

def get_model_module(model_def):
    importlib.invalidate_caches()
    return importlib.import_module(model_def)

def worker_init_fn(worker_id):
    """ The function is designed for pytorch multi-process dataloader.
        Note that we use the pytorch random generator to generate a base_seed.
        Please try to be consistent.
        References:
            https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed
    """
    base_seed = torch.IntTensor(1).random_().item()
    #print(worker_id, base_seed)
    np.random.seed(base_seed + worker_id)

def render_pc(out_fn, pc, figsize=(8, 8)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.view_init(elev=20, azim=60)
    x = pc[:, 0]
    y = pc[:, 2]
    z = pc[:, 1]
    ax.scatter(x, y, z, marker='.')
    miv = np.min([np.min(x), np.min(y), np.min(z)])  # Multiply with 0.7 to squeeze free-space.
    mav = np.max([np.max(x), np.max(y), np.max(z)])
    ax.set_xlim(miv, mav)
    ax.set_ylim(miv, mav)
    ax.set_zlim(miv, mav)
    plt.tight_layout()
    fig.savefig(out_fn, bbox_inches='tight')
    plt.close(fig)

def convert_color_to_hexcode(rgb):
    r, g, b = rgb
    return '#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255))

def render_part_pcs(pcs_list, title_list=None, out_fn=None, \
        subplotsize=(1, 1), figsize=(8, 8), azim=60, elev=20, scale=0.3):
    num_pcs = len(pcs_list)
    fig = plt.figure(figsize=figsize)
    for k in range(num_pcs):
        pcs = pcs_list[k]
        ax = fig.add_subplot(subplotsize[0], subplotsize[1], k+1, projection='3d')
        ax.view_init(elev=elev, azim=azim)
        xs = []; ys = []; zs = [];
        for i in range(pcs.shape[0]):
            x = pcs[i, :, 0]
            y = pcs[i, :, 2]
            z = pcs[i, :, 1]
            xs.append(x)
            ys.append(y)
            zs.append(z)
            if out_fn is None:
                ax.scatter(x, y, z, marker='.', s=scale, c=convert_color_to_hexcode(colors[i % len(colors)]))
            else:
                ax.scatter(x, y, z, marker='.', c=convert_color_to_hexcode(colors[i % len(colors)]))
        xs = np.concatenate(xs, axis=0)
        ys = np.concatenate(ys, axis=0)
        zs = np.concatenate(zs, axis=0)
        miv = np.min([np.min(xs), np.min(ys), np.min(zs)])
        mav = np.max([np.max(xs), np.max(ys), np.max(zs)])
        ax.set_xlim(miv, mav)
        ax.set_ylim(miv, mav)
        ax.set_zlim(miv, mav)
        if title_list is not None:
            ax.set_title(title_list[k])
    plt.tight_layout()
    if out_fn is not None:
        fig.savefig(out_fn, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def export_pc(out_fn, pc):
    with open(out_fn, 'w') as fout:
        for i in range(pc.shape[0]):
            fout.write('v %f %f %f\n' % (pc[i, 0], pc[i, 1], pc[i, 2]))

def export_part_pcs(out_dir, pcs):
    os.mkdir(out_dir)
    num_part = pcs.shape[0]
    num_point = pcs.shape[1]
    for i in range(num_part):
        with open(os.path.join(out_dir, 'part-%02d.obj' % i), 'w') as fout:
            for j in range(num_point):
                fout.write('v %f %f %f\n' % (pcs[i, j, 0], pcs[i, j, 1], pcs[i, j, 2]))

# out shape: (label_count, in shape)
def one_hot(inp, label_count):
    out = torch.zeros(label_count, inp.numel(), dtype=torch.uint8, device=inp.device)
    out[inp.view(-1), torch.arange(out.shape[1])] = 1
    out = out.view((label_count,) + inp.shape)
    return out

def collate_feats(b):
    return list(zip(*b))

# row_counts, col_counts: row and column counts of each distance matrix (assumed to be full if given)
def linear_assignment(distance_mat, row_counts=None, col_counts=None):
    batch_ind = []
    row_ind = []
    col_ind = []
    for i in range(distance_mat.shape[0]):
        # print(f'{i} / {distance_mat.shape[0]}')

        dmat = distance_mat[i, :, :]
        if row_counts is not None:
            dmat = dmat[:row_counts[i], :]
        if col_counts is not None:
            dmat = dmat[:, :col_counts[i]]

        rind, cind = linear_sum_assignment(dmat.to('cpu').numpy())
        rind = list(rind)
        cind = list(cind)

        if len(rind) > 0:
            rind, cind = zip(*sorted(zip(rind, cind)))
            rind = list(rind)
            cind = list(cind)

        # complete the assignemnt for any remaining non-active elements (in case row_count or col_count was given),
        # by assigning them randomly
        #if len(rind) < distance_mat.shape[1]:
        #    rind.extend(set(range(distance_mat.shape[1])).difference(rind))
        #    cind.extend(set(range(distance_mat.shape[1])).difference(cind))

        batch_ind += [i]*len(rind)
        row_ind += rind
        col_ind += cind

    return batch_ind, row_ind, col_ind

def load_pts(fn):
    with open(fn, 'r') as fin:
        lines = [item.rstrip() for item in fin]
        pts = np.array([[float(line.split()[0]), float(line.split()[1]), float(line.split()[2])] for line in lines], dtype=np.float32)
        return pts

def export_pts(out, v):
    with open(out, 'w') as fout:
        for i in range(v.shape[0]):
            fout.write('%f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2]))

def load_obj(fn):
    fin = open(fn, 'r')
    lines = [line.rstrip() for line in fin]
    fin.close()

    vertices = []; faces = [];
    for line in lines:
        if line.startswith('v '):
            vertices.append(np.float32(line.split()[1:4]))
        elif line.startswith('f '):
            faces.append(np.int32([item.split('/')[0] for item in line.split()[1:4]]))

    if len(faces) > 0:
        f = np.vstack(faces)
    else:
        f = None
    v = np.vstack(vertices)
    return v, f

def export_obj(out, v, f):
    with open(out, 'w') as fout:
        for i in range(v.shape[0]):
            fout.write('v %f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2]))
        for i in range(f.shape[0]):
            fout.write('f %d %d %d\n' % (f[i, 0], f[i, 1], f[i, 2]))

def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)

# pc is N x 3, feat is 10-dim
def transform_pc(pc, feat):
    num_point = pc.size(0)
    center = feat[:3]
    shape = feat[3:6]
    quat = feat[6:]
    pc = pc * shape.repeat(num_point, 1)
    pc = qrot(quat.repeat(num_point, 1), pc)
    pc = pc + center.repeat(num_point, 1)
    return pc

# pc is N x 3, feat is B x 10-dim
def transform_pc_batch(pc, feat, anchor=False):
    batch_size = feat.size(0)
    num_point = pc.size(0)
    pc = pc.repeat(batch_size, 1, 1)
    center = feat[:, :3].unsqueeze(dim=1).repeat(1, num_point, 1)
    shape = feat[:, 3:6].unsqueeze(dim=1).repeat(1, num_point, 1)
    quat = feat[:, 6:].unsqueeze(dim=1).repeat(1, num_point, 1)
    if not anchor:
        pc = pc * shape
    pc = qrot(quat.view(-1, 4), pc.view(-1, 3)).view(batch_size, num_point, 3)
    if not anchor:
        pc = pc + center
    return pc

def get_surface_reweighting(xyz, cube_num_point):
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    assert cube_num_point % 6 == 0, 'ERROR: cube_num_point %d must be dividable by 6!' % cube_num_point
    np = cube_num_point // 6
    out = torch.cat([(x*y).repeat(np*2), (y*z).repeat(np*2), (x*z).repeat(np*2)])
    out = out / (out.sum() + 1e-12)
    return out

def get_surface_reweighting_batch(xyz, cube_num_point):
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    assert cube_num_point % 6 == 0, 'ERROR: cube_num_point %d must be dividable by 6!' % cube_num_point
    np = cube_num_point // 6
    out = torch.cat([(x*y).unsqueeze(dim=1).repeat(1, np*2), \
                     (y*z).unsqueeze(dim=1).repeat(1, np*2), \
                     (x*z).unsqueeze(dim=1).repeat(1, np*2)], dim=1)
    out = out / (out.sum(dim=1).unsqueeze(dim=1) + 1e-12)
    return out

def gen_obb_mesh(obbs):
    # load cube
    cube_v, cube_f = load_obj('cube.obj')

    all_v = []; all_f = []; vid = 0;
    for pid in range(obbs.shape[0]):
        p = obbs[pid, :]
        center = p[0: 3]
        lengths = p[3: 6]
        dir_1 = p[6: 9]
        dir_2 = p[9: ]

        dir_1 = dir_1/np.linalg.norm(dir_1)
        dir_2 = dir_2/np.linalg.norm(dir_2)
        dir_3 = np.cross(dir_1, dir_2)
        dir_3 = dir_3/np.linalg.norm(dir_3)

        v = np.array(cube_v, dtype=np.float32)
        f = np.array(cube_f, dtype=np.int32)
        rot = np.vstack([dir_1, dir_2, dir_3])
        v *= lengths
        v = np.matmul(v, rot)
        v += center

        all_v.append(v)
        all_f.append(f+vid)
        vid += v.shape[0]

    all_v = np.vstack(all_v)
    all_f = np.vstack(all_f)
    return all_v, all_f

def sample_pc(v, f, n_points=2048):
    mesh = trimesh.Trimesh(vertices=v, faces=f-1)
    points, __ = trimesh.sample.sample_surface(mesh=mesh, count=n_points)
    return points


