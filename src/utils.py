import os, re, sys, traceback
import yaml, logging, logging.handlers, csv
import pandas as pd, numpy as np
import pickle, subprocess
from functools import reduce
from math import floor
from numpy import array, resize, zeros, float32, matmul, identity, shape
from numpy import ones, dot, divide, subtract, size, append, transpose, isscalar
from numpy import gradient, mean, std, outer, vstack, concatenate, savetxt
from numpy.linalg import inv, lstsq
from random import random
#from scipy import stats
from sklearn.decomposition import PCA
from time import time

logger = logging.getLogger("Utils")
logger.setLevel(logging.DEBUG)

logging.basicConfig(filename='tracker.log', 
    format='%(levelname)s %(asctime)s in %(funcName)s() ' +
        '%(filename)s-%(lineno)s: %(message)s \n', level=logging.DEBUG)

state_file = "lstm_ekf.state"
procs = []
cmd_range = None
pickle_dump_csv = True
n_entries = 3


def os_run(cmd):

    res = None
    if isinstance(cmd, list):
        for c in cmd:
           res = res + os_run(c) + "\n"
        return res

    try:
        logger.debug("Running cmd,type " + str((cmd, type(cmd))))
        proc = os.popen(cmd + " 2>>lstm_ekf.log")
        res = proc.read()
        proc.close()
        logger.debug("Ran cmd,out,proc = " + str((cmd, len(res), proc)))
    except:
        ex = traceback.format_exc()
        logger.error("Error running '"+ str(cmd) +"': " + str(ex))
    return res 


def os_run_path(path, regex, cmd):
    for r, d, f in os.walk(path):
        for file in f:
            if re.search(regex, file):
                os_run(cmd + os.path.join(path, file))


def utilization(key, states, stateinfo): 
    maxval = get(stateinfo, key, 'max')
    return states[int(key)] / maxval if maxval else states[int(key)]


def get(o, *keys, dv=None):
    for k in keys:
        if isinstance(k, list):
            o = get(o[k[0]], k[1:]) if len(k) and o else o
        elif isscalar(o) or not k in o:
            return dv # default value
        else:
            o = o[k]
    return o


def load_state():
    try:
        with open(state_file, 'r') as stream:
            try:
                return yaml.load(stream)
            except yaml.YAMLError as ex:
                logger.error("Unable to load state", ex)
    except FileNotFoundError:
        logger.error("State file not found.")
    return None


def save_state(runtime_state):
    with open(state_file, 'w') as out:
        try:
            out.write(yaml.dump(runtime_state))
        except yaml.YAMLError as ex:
            logger.error("Unable to save state", ex)
            return False
    return True


def merge_state(delta):
    state = load_state() or {}
    logger.info("state, delta = " + str((state, delta)))
    state.update(delta)
    save_state(state)
    return state


def repeat(v, n):
    if isinstance(v, type(lambda i:i)):
        return [v for i in range(0,n)]
    else:
        return [v for i in range(0,n)]


def avg(seq):
    size = len(list(seq))
    return sum(seq) / (size if size else size+1)


def flatlist(matrix):
    v = array(matrix);
    v.resize(size(matrix))
    return list(v)


def pickleconc(filename, values):
    history = pickleload(filename) or []
    pickledump(filename, history + values)


def pickleadd(filename, value):
    history = pickleload(filename) or []
    pickledump(filename, history + [value])


def pickledump(filename, value):
    try:
        #print(f"pickledump().value = {value}, to_csv = {pickle_dump_csv}")
        if pickle_dump_csv:
            #savetxt("{}.csv".format(filename), array(value))
            with open("{}.csv".format(filename), "w+") as cf:
                for r in value:
                  for i in range(len(r)):
                     cf.write("{}{}".format(r[i], "," if i<len(r)-1 else "\n"))
                cf.close()
        with open(filename, 'wb') as f:
            return pickle.dump(value, f)
    except(FileNotFoundError, pickle.PicklingError):
        logger.error(str(filename) + " erro " + str(sys.exec_info()[0]))
    return None


def pickleload(filename):
    try:
        if os.path.getsize(filename) > 0:
            with open(filename, 'rb') as f:
                return pickle.load(f)
    except Exception: #(FileNotFoundError, EOFError, pickle.UnpicklingError):
        logger.error(str(filename) + " error " + str(sys.exc_info()[0]))
    return []


def savecsv(filename, value):
            #savetxt("{}.csv".format(filename), array(value))
            with open("{}.csv".format(filename), "w+") as cf:
                for r in value:
                  for i in range(len(r)):
                     cf.write("{}{}".format(r[i], "," if i<len(r)-1 else "\n"))
                cf.close()


def replace_last(string, search, replacement):
    return string[::-1].replace(search[::-1], replacement[::-1])[::-1]


def occurrences(lst, search):
    search = search if isinstance(search, list) else [search]
    (ids, lstz) = ([], list(zip(range(len(lst)), lst)))
    for s in search:
        ids.extend(filter(lambda i: i>=0, [i if v==s else -1 for i,v in lstz]))
    return ids


def next(lst, item):
    i = occurrences(lst, item)
    return lst[i[0]+1] if len(i) and len(lst) > i[0] else None
    

def find(lst, val):
    ids = [i if v==val or not val else -1 for i,v in zip(range(len(lst)), lst)]
    return list(filter(lambda v: v>=0, ids))


def sublist(lst, ids):
    return [v for i,v in filter(lambda v:v[0] in ids, zip(range(len(lst)),lst))]


def getpca_raw(n_comp, y): 
    pca = PCA(n_components=n_comp)
    pca.fit(array(y).T)
    return pca


def getpca(n_comp, y): 
    global pca
    pca = PCA(n_components=n_comp)
    return pca.fit_transform(array(y).T) 


def getpca_inv(x): 
    #pca = PCA(n_components=n_comp)
    return pca.inverse_transform(array(x).T) 


def project(n_comp, y, m):
    return dot(getpca(n_comp, y).T, m)


# Approximates A for A*x = y
def solve_linear(x, ys, m_c = None):
    if m_c or not len(ys):
        return (m_c[0], m_c[1]) if m_c else (1, 0)
    y_pc = getpca(len(x), ys)
    logger.debug("y=" + str(shape(ys)) + ", y_pc = " + str(shape(y_pc)))
    return solve_linear_pca(x, y_pc)


def solve_linear_pca(x, y_pc):
    (m, c, _, _) = lstsq(y_pc.T, array(x).reshape(len(x),1), rcond=None)
    c = array(x).reshape(len(x),1) - dot(y_pc.T, m)
    logger.debug("solve_linear().m,m.y_pc,x) = " + str((m,dot(y_pc.T,m),x)))
    return (m, c)


def rotate_right(m):
    return transpose(concatenate([transpose(m[:,1:]), transpose(m[:,0:1])]))


def iscostconverged(costs, window = 5, tolerance = 30):
    return len(costs)>window and max(costs[-window:]) < tolerance


def isconverged(data, confidence=0.95):
    return len(list(filter(lambda c: c>=confidence, convergence(data))))>0


def convergence(data):
    if not len(data):
        return []
    (stat, stds, window, step, tolerance) = ([], [], 3, int(len(data)/10), 0.03)
    for i in range(int(len(data)/step)):
        stds.append(std(data[i*step:(i+1) * step]))
        win = stds[-window:]
        m = mean(win) 
        (l, u) = (m-tolerance, m+tolerance) 
        confidence = len(list(filter(lambda d: d>=l and d<=u, win)))/len(win)
        stat.append(confidence)
    logger.info("convergence.confidences = " + str(stat[1:]) + 
                ", data = " + str(shape(data)) + ", stds = " + str(stds))
    return stat[1:] # Ignore 1st window, its always 100% by definition



def twod(lst):
    data = array([[]])
    for row in lst:
        data = append(data, row)
    data.resize(len(lst), len(lst[0]) if len(shape(lst))>1 else 1)
    return data


# size of diag is row_count
# size of half is ( row_count^2 - row_count ) / 2
def symmetric(diag, half = None):
    (data, dmatrix, offset, rows) = ([], [], 0, len(diag))
    for i in range(rows):
        data.append([])
        dmatrix.append([])
        for j in range(rows):
            if i==j:
                dmatrix[-1].append(diag[i])
                data[-1].append(0)
            elif j > i:
                data[-1].append(half[offset] if half else 0)
                offset = offset + 1
                dmatrix[-1].append(0)
            else:
                data[-1].append(0)
                dmatrix[-1].append(0)
    upper = twod(data)
    return upper + transpose(upper) + twod(dmatrix)


def test_miscellaneous():
    print(symmetric([1,2,3], [4,5,6]))
    print(symmetric([1,2,3]))
    print(symmetric([1,2,3,4], [5,6,7,8,9,10]))
    print(os_run("wine lqns testbed.lqn"))
    print(sublist([1,2,3,4,5], [1,3]))
    print(merge_state({"abc": {"def": 1, "ghi": 2}}))
    print("occurrences = " + str(occurrences([1,2,3], [4,5,2])))
    print("test_miscellaneous() done")


def quantize(v, n = 1):
    if type(v) == list:
       return [quantize(i) for i in v]
    elif type(v) == type(array([])):
       return array([quantize(i) for i in v])
    return round(v, n)


def test_pca(ns = 100, predfn = None, dopca = True):
    (sz, n, d, lqn_p0, lqn_p1, y1s, ms) = (24, 10, 1, [], [], [], [])
    lqn_ps = [[random()*ns for i in range(3)] for y in range(2)]
    lqn_ps = [lqn_ps[floor(i/(ns/2))] for i in range(ns)]
    ys = [array([msmt(y,ns,sz,lqn_ps) for j in range(n)]) for y in range(ns)]
    return test_pca_basic(ns, predfn, dopca, lqn_ps, ys)


def test_pca_basic(ns=100, predfn=None, dopca=True, lqn_ps=[], ys=[], pre=""):
    (sz, n, d, lqn_p0, lqn_p1, y1s, ms) = (24, 10, 1, [], [], [], [])
    run_pca_tests(lqn_ps, ys, y1s, ms, lqn_p0, lqn_p1, sz, n, d, predfn, dopca)
    if not dopca:
        (lqn_p1, lqn_ps, y1s) = (lqn_p1, lqn_ps, array(y1s))
        #(lqn_p1, y1s) = (scale(lqn_p1, lqn_ps), scale(y1s, lqn_ps))
    save_pca_info(lqn_ps, ys, y1s, ms, lqn_p0, lqn_p1, d, pre, dopca)
    err = sum([sum(abs(p-p1.T[0])) for p,p1 in zip(lqn_ps,lqn_p1)])/len(lqn_ps)
    erry = sum([sum(abs(p-p1.T[0])) for p,p1 in zip(ys, y1s)])[-1]/len(ys)
    sd = np.std([sum(abs(p-p1.T[0])) for p,p1 in zip(lqn_ps,lqn_p1)])
    p = sum([sum(abs(array(p))) for p,p1 in zip(lqn_ps,lqn_p1)])/len(lqn_ps)
    print("Err.mean, Err.std, Err.y, % = "+str((err, sd, erry, 100*err/p)))
    print(f"Output in pca_*{pre}.pickle files")
    print("test_pca() done")
    return 100 * err / p


def msmt(y, ns, sz, lqn_ps):
    return [(random()*0.05+5)*lqn_ps[y][0] for i in range(sz)]



def run_pca_tests(lqn_ps, ys, y1s, ms, lqn_p0, lqn_p1, sz, n, d, predfn, dopca):
    predfn(ys[0], lqn_ps[0]) if predfn and not dopca else None
    for lqn_p, y, i in zip(lqn_ps, ys, range(len(ys))):
        (pca_y, pca_y1) = (most_sig_pca(len(lqn_p), y[-1:]), [])
        noise = array([[random()*0.001*i for i in r] for r in y]) 
        if dopca:
            y1 = predfn(y) if predfn else y + noise
            pca_y1 = most_sig_pca(len(lqn_p), y1)
        else:
            ystart = i-len(y) if i>len(y) else 0
            pstart = i-len(lqn_p) if i>len(lqn_p) else 0
            y1 = predfn(array(ys[ystart:i+1]))[0]
        y1s.append(y1)
        (m, c) = solve_linear_pca(lqn_p, pca_y1) if dopca else (0, 0) 
        if (len(ms) >= d):
            y1 = array([[y1[0]]])
            lqn_p1.append(dot(pca_y1.T,ms[-d][0])+ms[-d][1] if dopca else y1)
            lqn_p0.append(dot(pca_y.T, m) + c if dopca else y1)
        ms.append((m, c)) 


def depth(values):
    return len(array(values).shape)


def pad(y):
    return concatenate((array(y), zeros(array(y).shape)))


def most_sig_pca(ncol, y):
    return concatenate((getpca(1, pad(y).T).T, zeros([ncol-1,2*len(y)]))).T


def save_pca_info(lqn_ps, ys, y1s, ms, lqn_p0, lqn_p1, d, tag="", dopca=True):
    pickledump(f"pca_ms{tag}.pickle", [m for m,c in ms]) if dopca else None
    pickledump(f"pca_msmts{tag}.pickle", [y[-1] for y in ys])
    pickledump(f"pca_kfpriors{tag}.pickle", [y for y in y1s])
    pickledump(f"pca_lqnp{tag}.pickle", lqn_ps)
    pickledump(f"pca_lqnp0{tag}.pickle", [p0.T[0] for p0 in lqn_p0])
    pickledump(f"pca_lqn-kfprior{tag}.pickle", [p1.T[0] for p1 in lqn_p1])
    pairs = zip(lqn_p1,lqn_ps[d:])
    pickledump(f"pca_lqnerrors{tag}.pickle", [abs(p1.T[0]-p) for p1,p in pairs])


def max_min(vs):
    vs = array(vs).T[0] if len(array(vs).shape) > 2 else array(vs).T
    return ([max(c) for c in vs], [min(c) for c in vs])


def scale(src, tgt):
    (sx, sn) = max_min(src)
    (tx, tn) = max_min(tgt)
    def rng(c):
        (i, c) = c
        return (c-sn[i])/(sx[i]-sn[i])*(tx[i]-tn[i])+tn[i]
    return array([array(list(map(rng, zip(range(len(v)),v)))) for v in src])


def to_float(s):
    try:
        return float(s)
    except ValueError:
        pass
    return s


def to_size(data, width, entries = n_entries):
    input = array(data)
    if width > 0:
        input.resize(width, entries)
    else:
        input.resize(int(input.shape[1]/entries)*input.shape[0], entries)
        logger.debug("data = " + str(data) + ", input = " + str(shape(input.T)))
    input = input.T
    return input


def test_pca_csv(fname, xcol = '$uAppP', ycol = '$fGet_n', y1col = '$fGet',
                 predfn = None, dopca = True, pre = "", predictions=[]):
        (r, rows, hdrs) = (-1, {}, [])
        rows = pd.read_csv(fname)
        rows[xcol] = pd.to_numeric(rows[xcol], errors="coerce").fillna(0)
        rows[ycol] = pd.to_numeric(rows[ycol], errors="coerce").fillna(0)
        ys = list(zip(rows[ycol], rows[ycol]))
        (zs, rowids) = (zeros((10,len(ys[0]))), range(len(ys)))
        ys = [array(ys[r-10:r] if r>10 else rows[ycol][0]+zs) for r in rowids]
        xs = [[a for j in range(1)] for a in rows[xcol]][1:] + repeat([0],1)
        predfn = predfn if predfn else lambda y: ys[rows[ycol].index(y[0][0])]
        perr = test_pca_basic(len(rows[ycol]), predfn, dopca, xs, ys, pre)
        print("test_pca_csv(): % err = {}".format(perr))
        predictions.extend(xs)


if __name__ == "__main__":
    if "--testpca" in sys.argv:
        test_pca()
    elif "--testpcacsv" in sys.argv:
        #f = sys.path[0] + "/../data/2LayerVaryParam2-2LayerVaryParam1.csv"
        f = sys.path[0] + "/../data/mackey_glass_time_series.csv"
        test_pca_csv(f, 'P', 'P')
    else:
        test_miscellaneous()
