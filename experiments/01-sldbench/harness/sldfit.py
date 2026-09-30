"""
Forms found by NAI, refitted the way SLDBench asks: one functional form, its
constants fitted per group (sldbench.py).

A form is NAI's prefix notation (nai_cli --forms): x0, x1… the variables,
L… integer literals - part of the form -, C… real constants, e and pi. A
quotient of two literals (/ L18 L37) is a constant NAI snapped to a
rational; here it is a real constant again, to be refitted. Everything else
of the form stays as found.

Only the Python standard library: Levenberg–Marquardt with a numerical
Jacobian, at most a handful of constants.
"""
import math

# --- parsing ---------------------------------------------------------------

def parse(s):
    """'(+ C1.5 (* x0 L2))' → ['+', 'C1.5', ['*', 'x0', 'L2']]."""
    toks = s.replace("(", " ( ").replace(")", " ) ").split()
    pos = 0

    def node():
        nonlocal pos
        t = toks[pos]
        pos += 1
        if t != "(":
            return t
        out = []
        while toks[pos] != ")":
            out.append(node())
        pos += 1
        return out
    return node()


def free_constants(t):
    """The tree with its constants as ('c', value) leaves, in order."""
    if isinstance(t, str):
        return ("c", float(t[1:])) if t.startswith("C") else t
    if t[0] == "/" and all(isinstance(k, str) and k.startswith("L") for k in t[1:]):
        return ("c", int(t[1][1:]) / int(t[2][1:]))
    return [t[0]] + [free_constants(k) for k in t[1:]]


def skeleton(t):
    """The form without its constants' values: equal forms, equal strings."""
    if isinstance(t, tuple):
        return "C"
    if isinstance(t, str):
        return t
    return "(" + " ".join([t[0]] + [skeleton(k) for k in t[1:]]) + ")"


def constants(t, out=None):
    out = [] if out is None else out
    if isinstance(t, tuple):
        out.append(t[1])
    elif isinstance(t, list):
        for k in t[1:]:
            constants(k, out)
    return out


# --- evaluation --------------------------------------------------------------

def _pow(a, b):
    try:
        v = math.pow(a, b)
    except (OverflowError, ValueError, ZeroDivisionError):
        return math.nan
    return v


def _exp(a):
    try:
        return math.exp(a)
    except OverflowError:
        return math.inf


def _log(a):
    return math.log(a) if a > 0 else math.nan


def _sqrt(a):
    return math.sqrt(a) if a >= 0 else math.nan


def _div(a, b):
    return a / b if b != 0 else math.nan


def _fact(a):
    try:
        return math.gamma(a + 1) if a > -1 else math.nan
    except (OverflowError, ValueError):
        return math.nan


def _binom(n, k):
    if k < 0 or k > n:
        return 0.0
    try:
        return math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1))
    except (OverflowError, ValueError):
        return math.nan


def _ffact(a, b):
    if b < 0 or b != math.floor(b):
        return math.nan
    v = 1.0
    for k in range(int(b)):
        v *= a - k
    return v


def _sum(bound, body):
    if not math.isfinite(bound):
        return math.nan
    return sum(body(j) for j in range(max(0, int(bound))))


def _prod(bound, body):
    if not math.isfinite(bound):
        return math.nan
    v = 1.0
    for j in range(max(0, int(bound))):
        v *= body(j)
    return v


ENV = {"_pow": _pow, "_exp": _exp, "_log": _log, "_sqrt": _sqrt, "_div": _div, "_fact": _fact,
       "_binom": _binom, "_ffact": _ffact, "_sum": _sum, "_prod": _prod, "math": math, "nan": math.nan}


def code(t, counter):
    """Python source of the tree; constants read from c[i] in order."""
    if isinstance(t, tuple):
        i = counter[0]
        counter[0] += 1
        return "c[%d]" % i
    if isinstance(t, str):
        if t.startswith("x"):
            return t
        if t.startswith("L"):
            return repr(float(t[1:]))
        if t == "j":
            return "j"
        if t == "pi":
            return repr(math.pi)
        if t == "e":
            return repr(math.e)
        raise ValueError("unknown leaf " + t)
    op, k = t[0], [code(x, counter) for x in t[1:]]
    if op == "+":
        return "(%s + %s)" % (k[0], k[1])
    if op == "-":
        return "(%s - %s)" % (k[0], k[1])
    if op == "*":
        return "(%s * %s)" % (k[0], k[1])
    if op == "/":
        return "_div(%s, %s)" % (k[0], k[1])
    if op == "^":
        return "_pow(%s, %s)" % (k[0], k[1])
    if op == "neg":
        return "(-%s)" % k[0]
    if op in ("exp", "log", "sqrt", "fact"):
        return "_%s(%s)" % (op, k[0])
    if op in ("sin", "cos"):
        return "math.%s(%s)" % (op, k[0])
    if op in ("binom", "ffact"):
        return "_%s(%s, %s)" % (op, k[0], k[1])
    if op in ("sum", "prod"):
        return "_%s(%s, lambda j: %s)" % (op, k[0], k[1])
    raise ValueError("unknown operation " + op)


def log_inputs(t, logvars):
    """A form found on logarithms of some inputs, written on the inputs
    themselves: x_i → log(x_i) for i in logvars."""
    if isinstance(t, str):
        return ["log", t] if t.startswith("x") and int(t[1:]) in logvars else t
    return [t[0]] + [log_inputs(k, logvars) for k in t[1:]]


class Form:
    def __init__(self, text, nvars, logvars=(), logy=False):
        # As found: for a program that rebuilds the form. logy - found for
        # log y (log–log space, where a power law is linear): the form is exp(g).
        self.text, self.logvars, self.logy = text, tuple(logvars), logy
        tree = parse(text)
        if logvars:
            tree = log_inputs(tree, set(logvars))
        if logy:
            tree = ["exp", tree]
        self.tree = free_constants(tree)
        self.key = skeleton(self.tree)
        self.init = constants(self.tree)
        self.k = len(self.init)
        src = code(self.tree, [0])
        args = ", ".join("x%d" % i for i in range(nvars)) + ("," if nvars == 1 else "")
        self.f = eval("lambda R, c: [_safe(lambda: %s) for (%s) in R]" % (src, args),
                      {**ENV, "_safe": _safe})

    def __call__(self, rows, c):
        return self.f(rows, c)


def _safe(g):
    try:
        v = g()
        v = v if isinstance(v, float) else float(v) if isinstance(v, int) else math.nan
    except (OverflowError, ValueError, ZeroDivisionError, TypeError):
        return math.nan
    return v if abs(v) <= 1e150 else math.nan      # beyond any loss: a form that has left the data's world


# --- fitting -----------------------------------------------------------------

def _solve(A, b):
    """Gaussian elimination with partial pivoting; None if singular."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(M[r][i]))
        if abs(M[p][i]) < 1e-300:
            return None
        M[i], M[p] = M[p], M[i]
        for r in range(i + 1, n):
            f = M[r][i] / M[i][i]
            for cc in range(i, n + 1):
                M[r][cc] -= f * M[i][cc]
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))) / M[i][i]
    return x


def sse(form, rows, y, c):
    p = form(rows, c)
    s = 0.0
    for a, b in zip(p, y):
        if not math.isfinite(a) or abs(a) > 1e150:
            return math.inf
        s += (a - b) ** 2
    return s


def fit(form, rows, y, c0, iters=60):
    """Least squares from c0: Levenberg–Marquardt. Returns (c, sse)."""
    c = list(c0)
    e = sse(form, rows, y, c)
    if form.k == 0 or not rows:
        return c, e
    lam = 1e-3
    for _ in range(iters):
        p = form(rows, c)
        if not all(math.isfinite(v) for v in p):
            return c, math.inf
        r = [a - b for a, b in zip(p, y)]
        J = []
        for i in range(form.k):
            h = 1e-6 * max(abs(c[i]), 1e-8)
            cp = list(c)
            cp[i] += h
            q = form(rows, cp)
            J.append([(qa - pa) / h if math.isfinite(qa) else 0.0 for qa, pa in zip(q, p)])
        JTJ = [[sum(a * b for a, b in zip(J[i], J[j])) for j in range(form.k)] for i in range(form.k)]
        JTr = [sum(a * b for a, b in zip(J[i], r)) for i in range(form.k)]
        improved = False
        for _ in range(8):
            A = [[JTJ[i][j] + (lam * (JTJ[i][i] + 1e-12) if i == j else 0.0) for j in range(form.k)]
                 for i in range(form.k)]
            d = _solve(A, [-v for v in JTr])
            if d is None:
                lam *= 10
                continue
            cn = [a + b for a, b in zip(c, d)]
            en = sse(form, rows, y, cn)
            if en < e:
                rel = (e - en) / max(e, 1e-300)
                c, e, lam = cn, en, max(lam / 3, 1e-9)
                improved = True
                break
            lam *= 10
        if not improved or rel < 1e-10:
            break
    return c, e


# --- printing ----------------------------------------------------------------

def pretty(t, names, c=None, counter=None):
    counter = [0] if counter is None else counter
    if isinstance(t, tuple):
        v = c[counter[0]] if c else t[1]
        counter[0] += 1
        return "%.4g" % v
    if isinstance(t, str):
        if t.startswith("x"):
            return names[int(t[1:])]
        if t.startswith("L"):
            return t[1:]
        return t
    op, k = t[0], [pretty(x, names, c, counter) for x in t[1:]]
    if op in "+-*/^":
        return "(%s %s %s)" % (k[0], op, k[1])
    if op == "neg":
        return "-" + k[0]
    if op in ("sum", "prod"):
        return "%s[j<%s](%s)" % ("Σ" if op == "sum" else "Π", k[0], k[1])
    return "%s(%s)" % (op, ", ".join(k))
