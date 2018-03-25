from common import *
import itertools
import functools
import math
import operator


def binom(n, k):
    k = min(k, n - k)
    return functools.reduce(lambda a, b: a * (n - b) // (b + 1), range(k), 1)


@memoized
def trinary_coefficients_mod_sign(length):
    '''    
    Paramaters
    ----------
    length : number of things to take linear combinations of
    Return
    ------
    numpy.ndarray of shape ((3^n+1)/2, n), n=length, containing all possible {-1,0,1}-valued
        vectors of length n with the first nonzero entry being 1
    '''
    n = length
    s = (3 ** n + 1) // 2
    c = np.full([s, n], 32, dtype=np.int8)
    ci = np.full([n], 2,  dtype=np.int8)
    i = 0
    head = 0
    p = 0
    while i != s:
        if ci[p] == 0:
            ci[p] = 2
            p -= 1
            continue
        ci[p] = (None, 0, -1, 1)[ci[p]]
        if head == p:
            ci[p] = np.abs(ci[p])
            if ci[p] == 0:
                head += 1
        if p == n - 1:
            c[i] = ci
            i += 1
        else:
            p += 1
    return c


@memoized
def consecutive_1s(n):
    '''
    Paramaters
    ----------
    length : length of array
    Return
    ------
    numpy.ndarray of shape (1/2*n*(n+1), n), n=length, containing all possible sequences of 0/1s
        where the 1s are consecutively placed (and there is at least one 1)
    '''
    s = n * (n + 1) // 2
    c = np.zeros([s, n], dtype=np.int8)
    t = 0
    for i in range(n):
        for j in range(i + 1, n + 1):
            for k in range(i, j):
                c[t, k] = 1
            t += 1
    return c


@memoized
def one_minus_one(n):
    '''
    Paramaters
    ----------
    length : length of array
    Return
    ------
    numpy.ndarray of shape (binom(n, 2), n), n=length, containing all possible sequences of entries of a 1 and a -1
    '''
    s = n * (n - 1) // 2
    c = np.zeros([s, n], dtype=np.int8)
    t = 0
    for i in range(n):
        for j in range(i + 1, n):
            c[t, i] = 1
            c[t, j] = -1
            t += 1
    return c


@memoized
def signed_coefficients_mod_sign(n):
    if n == 0:
        return np.zeros([1, 0], dtype=np.int8)
    s = 2 ** (n - 1)
    c = np.zeros([s, n], dtype=np.int8)
    for i in range(s):
        b = np.asarray([(i >> j) % 2 for j in range(n)[::-1]])
        c[i] = 1 - b - b
    return c

@memoized
def one_minus_one_signed_sums_mod_sign(n, k):
    guys = []
    c = np.concatenate([np.zeros([1, n], dtype=np.int8), one_minus_one(n)], axis=0)
    seen = set()
    for x_list in itertools.combinations_with_replacement(c, k):
        for sc in signed_coefficients_mod_sign(k):
            guy = sum((x * b for b, x in zip(sc, x_list)), np.zeros([n], dtype=np.int8))
            if tuple(guy) not in seen and functools.reduce(math.gcd, guy) == 1:
                seen.add(tuple(guy))
                seen.add(tuple(-guy))
                guys.append(guy)
    return np.stack(guys, axis=0) if guys else np.ndarray([0, n], dtype=np.int8)


@memoized
def consecutive_coefficients_mod_sign(n, w, h):
    cc = []
    for c_list in itertools.product(range(1, h + 1), *([range(-h, h + 1)] * (w - 1))):
        if all(c_list[i] != c_list[i + 1] for i in range(w - 1)) and c_list[-1] != 0 and functools.reduce(math.gcd, c_list) == 1:
            cc.append(c_list)
    s = binom(n + 1, w + 1) * len(cc)
    c = np.zeros([s, n], dtype=np.int8)
    t = 0
    for i_list in itertools.combinations(range(n + 1), w + 1):
        i_list = i_list + (n,)
        for c_list in cc:
            c[t, :i_list[0]] = 0
            for i in range(w):
                c[t, i_list[i]:i_list[i + 1]] = c_list[i]
            c[t, i_list[-1]:] = 0
            t += 1
    return c


@memoized
def zero_sum_consecutive_coefficients_mod_sign(n, w, h):
    cc = []
    for c_list in itertools.product(range(1, h + 1), *([range(-h, h + 1)] * (w - 1))):
        if all(c_list[i] != c_list[i + 1] for i in range(w - 1)) and c_list[-1] != 0 and functools.reduce(math.gcd, c_list) == 1:
            cc.append(c_list)
    s = binom(n + 1, w + 1) * len(cc)
    guys = []
    for i_list in itertools.combinations(range(n + 1), w + 1):
        i_list = i_list + (n,)
        for c_list in cc:
            guy = np.zeros([n], dtype=np.int8)
            guy[:i_list[0]] = 0
            for i in range(w):
                guy[i_list[i]:i_list[i + 1]] = c_list[i]
            guy[i_list[-1]:] = 0
            if sum(guy) == 0:
                guys.append(guy)
    return np.stack(guys, axis=0) if guys else np.ndarray([0, n], dtype=np.int8)


@memoized
def up_to_w_consecutive_coefficients_mod_sign(n, w, h):
    guys = [consecutive_coefficients_mod_sign(n, w_, h) for w_ in range(1, w + 1)]
    return np.concatenate(guys, axis=0) if guys else np.ndarray([0, n], dtype=np.int8)


@memoized
def up_to_w_zero_sum_consecutive_coefficients_mod_sign(n, w, h):
    guys = [zero_sum_consecutive_coefficients_mod_sign(n, w_, h) for w_ in range(1, w + 1)]
    return np.concatenate(guys, axis=0) if guys else np.ndarray([0, n], dtype=np.int8)


@memoized
def consecutive_1s_signed_sums_mod_sign(n, m):
    guys = []
    c1 = np.append(consecutive_1s(n), [np.asarray([0] * n, dtype=np.int8)], axis=0)
    seen = set()
    for x_list in itertools.combinations(c1, m):
        for sc in signed_coefficients_mod_sign(m):
            guy = sum(x * b for b, x in zip(sc, x_list))
            if tuple(guy) not in seen:
                seen.add(tuple(guy))
                seen.add(tuple(-guy))
                guys.append(guy)
    if guys and not guys[-1].any():
        guys = guys[:-1]
    return np.stack(guys, axis=0) if guys else np.ndarray([0, n], dtype=np.int8)


@memoized
def consecutive_1s_signed_sums_duplicates_mod_sign(n, m):
    guys = []
    c1 = np.append(consecutive_1s(n), [np.asarray([0] * n, dtype=np.int8)], axis=0)
    for x_list in itertools.combinations(c1, m):
        for sc in signed_coefficients_mod_sign(m):
            guy = sum(x * b for b, x in zip(sc, x_list))
            guys.append(guy)
    if guys and not guys[-1].any():
        guys = guys[:-1]
    return np.stack(guys, axis=0) if guys else np.ndarray([0, n], dtype=np.int8)


@memoized
def up_to_m_signed_coefficients_mod_sign(n, m):
    guys = []
    ident = np.eye(n, dtype=np.int8)
    for k in range(1, m + 1):
        for x_list in itertools.combinations(ident, k):
            for sc in signed_coefficients_mod_sign(k):
                guy = sum(x * b for b, x in zip(sc, x_list))
                guys.append(guy)
    return np.stack(guys, axis=0) if guys else np.ndarray([0, n], dtype=np.int8)


@memoized
def zero_mean_int_coefficients_mod_sign(n, m, duplicates=True):
    guys = []
    loc = 0
    path = [0]
    while True:
        while True:
            if not path:
                return np.stack(guys, axis=0) if guys else np.ndarray([0, n], dtype=np.int8)
            path[-1] += 1
            loc += 1
            if path[-1] == 0:
                path[-1] += 1
                loc += 1
            if path[-1] > m:
                loc -= path.pop()
                continue
            if abs(loc) > m * (n - len(path)):
                continue
            if len(path) == n - 1:
                break
            path.append(-m - 1)
            loc -= m + 1
        path.append(-loc)
        if path[-1] != 0 and (duplicates or functools.reduce(math.gcd, path) == 1):
            guys.append(np.asarray(path, dtype=np.int8))
        path.pop()


@memoized
def int_coefficients_mod_sign(n, m, duplicates=True):
    guys = []
    path = [0]
    while True:
        while True:
            if not path:
                return np.stack(guys, axis=0) if guys else np.ndarray([0, n], dtype=np.int8)
            path[-1] += 1
            if path[-1] == 0:
                path[-1] += 1
            if path[-1] > m:
                path.pop()
                continue
            if len(path) == n:
                break
            path.append(-m - 1)
        if duplicates or functools.reduce(math.gcd, path) == 1:
            guys.append(np.asarray(path, dtype=np.int8))


@memoized
def up_to_m_zero_mean_int_coefficients_mod_sign(n, m, h, duplicates=True):
    guys = []
    ident = np.eye(n, dtype=np.int8)
    for k in range(1, m + 1):
        for x_list in itertools.combinations(ident, k):
            for sc in zero_mean_int_coefficients_mod_sign(k, h, duplicates):
                guy = sum(x * b for b, x in zip(sc, x_list))
                guys.append(guy)
    return np.stack(guys, axis=0) if guys else np.ndarray([0, n], dtype=np.int8)


@memoized
def up_to_m_int_coefficients_mod_sign(n, m, h, duplicates=True):
    guys = []
    ident = np.eye(n, dtype=np.int8)
    for k in range(1, m + 1):
        for x_list in itertools.combinations(ident, k):
            for sc in int_coefficients_mod_sign(k, h, duplicates):
                guy = sum(x * b for b, x in zip(sc, x_list))
                guys.append(guy)
    return np.stack(guys, axis=0)


def trico(n):
    '''
    Same as trinary_coefficients_mod_sign but excludes the 0 vector
    '''
    return trinary_coefficients_mod_sign(n)[:-1]


class name(metaclass=O):
    def index(c):
        assert len(c) <= 10
        pos, neg = [], []
        for i, a in enumerate(c):
            assert isinstance(a, np.integer) or isinstance(a, int)
            if a != 0:
                arr = pos if a > 0 else neg
                arr += [i] * abs(a)
        return ''.join(str(i) for i in pos) + '/' + ''.join(str(i) for i in neg)


class combos(metaclass=O):
    name = name
    c1co = consecutive_1s #(n)
    ksgnc1co = consecutive_1s_signed_sums_mod_sign #(n, k=<how many c1 arrays to combine>)
    ksgnc1cox = consecutive_1s_signed_sums_duplicates_mod_sign #(n, k=<how many c1 arrays to combine>)
    sgnco = signed_coefficients_mod_sign #(n)
    mmsgnco = trico #(n)
    msgnco = up_to_m_signed_coefficients_mod_sign #(n, m=<max num nonzero coefficients>)
    intco0 = lambda n, h: zero_mean_int_coefficients_mod_sign(n, h, False) #(n, <max abs value of coefficients>)
    intco0x = lambda n, h: zero_mean_int_coefficients_mod_sign(n, h, True) #(n, <max abs value of coefficients>)
    intco = lambda n, h: int_coefficients_mod_sign(n, h, False) #(n, <max abs value of coefficients>)
    intcox = lambda n, h: int_coefficients_mod_sign(n, h, True) #(n, <max abs value of coefficients>)
    mintco0 = lambda n, m, h: up_to_m_zero_mean_int_coefficients_mod_sign(n, m, h, False) #(n, m=<max num nonzer coefficients>, <max abs value of coefficients>)
    mintco0x = lambda n, m, h: up_to_m_zero_mean_int_coefficients_mod_sign(n, m, h, True) #(n, m=<max num nonzer coefficients>, <max abs value of coefficients>)
    mintco = lambda n, m, h: up_to_m_int_coefficients_mod_sign(n, m, h, False) #(n, m=<max num nonzer coefficients>, <max abs value of coefficients>)
    mintcox = lambda n, m, h: up_to_m_int_coefficients_mod_sign(n, m, h, True) #(n, m=<max num nonzer coefficients>, <max abs value of coefficients>)
    komoco = one_minus_one_signed_sums_mod_sign
    wcintco = up_to_w_consecutive_coefficients_mod_sign
    wcintco0 = up_to_w_zero_sum_consecutive_coefficients_mod_sign


if __name__ == "__main__":
    n = 1
    k = 1
    m = 1
    h = 1
    assert combos.mmsgnco(n).__len__() == 1
    assert combos.sgnco(n).__len__() == 1
    assert combos.ksgnc1co(n, k).__len__() == 1
    assert combos.ksgnc1cox(n, k).__len__() == 1
    assert combos.msgnco(n, m).__len__() == 1
    assert combos.intco0(n, h).__len__() == 0
    assert combos.intco0x(n, h).__len__() == 0
    assert combos.intco(n, h).__len__() == 1
    assert combos.intcox(n, h).__len__() == 1
    assert combos.mintco0(n, m, h).__len__() == 0
    assert combos.mintco0x(n, m, h).__len__() == 0
    assert combos.mintco(n, m, h).__len__() == 1
    assert combos.mintcox(n, m, h).__len__() == 1
    n = 2
    k = 1
    m = 1
    h = 1
    assert combos.mmsgnco(n).__len__() == 4
    assert combos.sgnco(n).__len__() == 2
    assert combos.ksgnc1co(n, k).__len__() == 3
    assert combos.ksgnc1cox(n, k).__len__() == 3
    assert combos.msgnco(n, m).__len__() == 2
    assert combos.intco0(n, h).__len__() == 1
    assert combos.intco0x(n, h).__len__() == 1
    assert combos.intco(n, h).__len__() == 2
    assert combos.intcox(n, h).__len__() == 2
    assert combos.mintco0(n, m, h).__len__() == 0
    assert combos.mintco0x(n, m, h).__len__() == 0
    assert combos.mintco(n, m, h).__len__() == 2
    assert combos.mintcox(n, m, h).__len__() == 2