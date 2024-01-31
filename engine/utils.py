import taichi as ti
import numpy as np

def rolling_gaussian_generator(velocity, frequency, node_x_locs, N=5, resolution=200):
    """
    Return a Gaussian modulated sinusoid:

        ``exp(-a t^2) exp(1j*2*pi*fc*t).``

    Parameters
    ----------
    t : ndarray
        Input array.
    fc : float, optional
        Center frequency (e.g. Hz).  Default is 1000.
    bw : float, optional
        Fractional bandwidth in frequency domain of pulse (e.g. Hz).
        Default is 0.5.
    bwr : float, optional
        Reference level at which fractional bandwidth is calculated (dB).
        Default is -6.

    Returns
    -------
    yI : ndarray
        Real part of signal.
    yQ : ndarray
        Imaginary part of signal.
    yenv : ndarray
        Envelope of signal.
    """
    time_to_center = node_x_locs / velocity

    fc = frequency
    t_width = N / fc  # half-width of time range

    t = np.linspace(-t_width, t_width, resolution, endpoint=False)
    _, _, e = gausspulse(t, fc=fc)

    t_pad = np.array([-t_width - 1e-16, t_width + 1e-16])
    e_pad = np.array([0, 0])

    t = np.concatenate([t_pad[:1], t, t_pad[1:]])
    e = np.concatenate([e_pad[:1], e, e_pad[1:]])

    xvalues = t + time_to_center[:, np.newaxis]
    fxvalues = e
    return xvalues, fxvalues

def gausspulse(t, fc=1000, bw=0.5, bwr=-6):
    """
    Return a Gaussian modulated sinusoid:

        ``exp(-a t^2) exp(1j*2*pi*fc*t).``

    Parameters
    ----------
    t : ndarray
        Input array.
    fc : float, optional
        Center frequency (e.g. Hz).  Default is 1000.
    bw : float, optional
        Fractional bandwidth in frequency domain of pulse (e.g. Hz).
        Default is 0.5.
    bwr : float, optional
        Reference level at which fractional bandwidth is calculated (dB).
        Default is -6.

    Returns
    -------
    yI : ndarray
        Real part of signal.
    yQ : ndarray
        Imaginary part of signal.
    yenv : ndarray
        Envelope of signal.
    """

    # Convert t to a jnp array if it's not already
    t = np.asarray(t)

    # exp(-a t^2) <->  sqrt(pi/a) exp(-pi^2/a * f^2)  = g(f)
    ref = np.power(10.0, bwr / 20.0)
    
    # pi^2/a * fc^2 * bw^2 /4=-log(ref)
    a = -(np.pi * fc * bw) ** 2 / (4.0 * np.log(ref))

    yenv = np.exp(-a * t * t)
    yI = yenv * np.cos(2 * np.pi * fc * t)
    yQ = yenv * np.sin(2 * np.pi * fc * t)

    return yI, yQ, yenv

def interpolate(x, xvalues, fxvalues):
    assert len(xvalues) == len(fxvalues)
    # Check if x is outside the range of xvalues
    if x < min(xvalues) or x > max(xvalues):
        raise ValueError("x is outside the range of xvalues")

    # Find the two closest x-values to x
    i = 0
    while x > xvalues[i + 1]:
        i += 1
        
    # Linear interpolation formula
    x0, x1 = xvalues[i], xvalues[i + 1]
    fx0, fx1 = fxvalues[i], fxvalues[i + 1]

    interpolated_value = fx0 + (fx1 - fx0) * (x - x0) / (x1 - x0)

    return interpolated_value