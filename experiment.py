import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats
import random
import jax.scipy.stats.norm as norm
from jax import grad
import matplotlib.pyplot as plt
import sympy as sp
from scipy.integrate import quad
import random
import time
import copy


def count(l):
    c = np.zeros(10)
    for xx in l:
        c[xx] += 1
    return np.array(c)


def noise_agg(hist, sig=1):
    hist = list(hist)
    nhist = [x + np.random.normal(0, sig) for x in hist]
    return int(np.argmax(nhist))


def qhat(hist, sig=40, max_iter=5e4):
    m = len(hist)
    qs = np.array([0.] * m)
    count = 0

    def helper():
        i = noise_agg(hist, sig)
        qs[i] = qs[i] + 1

    nothing = [helper() for m in range(1, int(max_iter) + 1)]
    return [x / max_iter for x in qs]


def true_q_helper1(hist, j, sig=1):
    m = len(hist)
    x = sp.Symbol('x')

    def f(x):
        p = norm.pdf(x, hist[j], sig)
        for i in range(m):
            if i == j:
                continue
            else:
                p *= norm.cdf(x, hist[i], sig)
        return p

    return quad(f, -np.inf, np.inf)[0]


def true_q1(hist, sig=40):
    m = len(hist)
    q = [true_q_helper1(hist, j, sig) for j in range(m)]
    return np.array(q)


def true_q_helper(hist, j, sig=1):
    m = len(hist)

    def f(x):
        p = norm.pdf(x, hist[j], sig)
        for i in range(m):
            if i == j:
                continue
            else:
                p *= norm.cdf(x, hist[i], sig)
        return p

    r = jnp.arange(hist[j] - 6 * sig, hist[j] + 6 * sig, 1)
    ys = jnp.array([f(rr) for rr in r])
    return jnp.trapz(ys)


def true_q(hist, sig, root):
    m = len(hist)
    cur = jnp.array([true_q_helper(hist, i, sig) for i in range(m)])
    a = cur / jnp.linalg.norm(cur)
    b = jnp.array(root) / jnp.linalg.norm(jnp.array(root))
    return jnp.linalg.norm(a - b)


def corrupt(votes, percentage):
    corrupt_num = int(len(votes) * percentage)
    new_votes = copy.deepcopy(votes)
    indices = random.sample(range(0, len(votes)), corrupt_num)
    n = 0
    corrupt_hist = np.zeros(10)
    for i in indices:
        if n == 10:
            n = 0
        new_votes[i] = n
        corrupt_hist[n] += 1
        n += 1
    new_hist = np.array(count(new_votes))
    honest_hist = new_hist - corrupt_hist
    return new_hist, honest_hist, corrupt_hist


def gradient_descent(votes, sig, max_iter, corrupt_rate):
    hist, honest_hist, corrupt_hist = corrupt(votes, corrupt_rate)
    print("hist: ", hist)
    print("honest hist: ", honest_hist)
    import numpy as np
    qvector = np.array(qhat(hist, sig=sig, max_iter=max_iter))
    print("frequency: ", qvector)
    y = true_q1(hist, sig)
    # qvector = y
    print("true probability vector: ", y)
    import jax.numpy as np
    init = np.zeros(10)+25
#     init = 250 * np.array(qvector)
    print("init: ", init)
    x = init
    xs = [init]
    errs = [100]
    big = True
    for i in range(100):
        start = time.time()
        grads = grad(true_q)(x, sig, np.array(qvector))
        end = time.time()
        usage = end - start
        print("gradient descent time: ", usage)
        if i == 0:
            start = time.time()
            err = true_q(x, sig, np.array(qvector))
            end = time.time()
            usage = end - start
            print("calculate error time: ", usage)
        print('err: ',np.linalg.norm(x-hist))
        print('GD err: ',err)
        if err > errs[-1]:
            xs = xs[0:-1]
            break
        if err < 0.01:
            break
        errs.append(err)
        
        if big:
            lr = 10 / np.linalg.norm(grads)
            x10 = x - lr * grads
            start = time.time()
            err = true_q(x10, sig, np.array(qvector))
            end = time.time()
            usage = end - start
            print("calculate error time: ", usage)
            if err > errs[-1]:
                big = False
            else:
                print("learning rate is 10")
                x = x10
        if not big:
            lr = 1 / np.linalg.norm(grads)
            x = x - lr * grads
            start = time.time()
            err = true_q(x, sig, np.array(qvector))
            end = time.time()
            usage = end - start
            print("calculate error time: ", usage)
            if err > errs[-1] or err < 0.01:
                break
            print("learning rate is 1")
        xs.append(x)
        errs.append(err)
        print("x: ", x)
        print("GD error: ", err)
        print("hist error: ", np.linalg.norm(x-hist))
    print('final err: ', np.linalg.norm(hist-x))
    print('training process: ',errs[1:])
    return np.linalg.norm(hist-x)/len(honest_hist), errs[1:]
