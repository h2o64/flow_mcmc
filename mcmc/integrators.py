# Integrators for HMC
# Taken from https://github.com/ColCarroll/minimc/blob/master/minimc/integrators.py

# Libraries
import math

def verlet(q, p, dVdq, potential_grad, path_len, step_size, inv_mass_matrix):
    """Second order symplectic integrator that uses the velocity verlet algorithm.
    Stolen from Pyro

    Parameters
    ----------
    q : torch.Tensor
        Initial position
    p : torch.Tensor
        Initial momentum
    dVdq : torch.Tensor
        Gradient of the potential at the initial coordinates
    potential_grad : callable
        Value and gradient of the potential
    path_len : float
        How long to integrate for
    step_size : float
        How long each integration step should be
    Returns
    -------
    q, p : torch.Tensor, torch.Tensor
        New position and momentum
    """

    for _ in range(int(path_len / step_size)):
        p -= step_size * dVdq / 2 # half step
        q += step_size * p / inv_mass_matrix # whole step
        V, dVdq = potential_grad(q)
        p -= step_size * dVdq / 2

    return q, -p, V, dVdq


def leapfrog(q, p, dVdq, potential_grad, path_len, step_size, inv_mass_matrix):
    """Leapfrog integrator for Hamiltonian Monte Carlo.

    Parameters
    ----------
    q : torch.Tensor
        Initial position
    p : torch.Tensor
        Initial momentum
    dVdq : torch.Tensor
        Gradient of the potential at the initial coordinates
    potential_grad : callable
        Value and gradient of the potential
    path_len : float
        How long to integrate for
    step_size : float
        How long each integration step should be
    Returns
    -------
    q, p : torch.Tensor, torch.Tensor
        New position and momentum
    """

    p -= step_size * dVdq / 2  # half step
    for _ in range(int(path_len / step_size) - 1):
        q += step_size * p / inv_mass_matrix # whole step
        V, dVdq = potential_grad(q)
        p -= step_size * dVdq  # whole step
    q += step_size * p / inv_mass_matrix # whole step
    V, dVdq = potential_grad(q)
    p -= step_size * dVdq / 2  # half step

    # momentum flip at end
    return q, -p, V, dVdq


def leapfrog_twostage(q, p, dVdq, potential_grad, path_len, step_size, inv_mass_matrix):
    """A second order symplectic integration scheme.

    Based on the implementation from Adrian Seyboldt in PyMC3. See
    https://github.com/pymc-devs/pymc3/pull/1758 for a discussion.

    References
    ----------
    Blanes, Sergio, Fernando Casas, and J. M. Sanz-Serna. "Numerical
    Integrators for the Hybrid Monte Carlo Method." SIAM Journal on
    Scientific Computing 36, no. 4 (January 2014): A1556-80.
    doi:10.1137/130932740.

    Mannseth, Janne, Tore Selland Kleppe, and Hans J. Skaug. "On the
    Application of Higher Order Symplectic Integrators in
    Hamiltonian Monte Carlo." arXiv:1608.07048 [Stat],
    August 25, 2016. http://arxiv.org/abs/1608.07048.

    Parameters
    ----------
    q : torch.Tensor
        Initial position
    p : torch.Tensor
        Initial momentum
    dVdq : torch.Tensor
        Gradient of the potential at the initial coordinates
    potential_grad : callable
        Value and gradient of the potential
    path_len : float
        How long to integrate for
    step_size : float
        How long each integration step should be

    Returns
    -------
    q, p : torch.Tensor, torch.Tensor
        New position and momentum
    """

    a = (3 - math.sqrt(3)) / 6

    p -= a * step_size * dVdq  # `a` momentum update
    for _ in range(int(path_len / step_size) - 1):
        q += step_size * p / (inv_mass_matrix * 2)  # half position update
        V, dVdq = potential_grad(q)
        p -= (1 - 2 * a) * step_size * dVdq  # 1 - 2a position update
        q += step_size * p / (inv_mass_matrix * 2)  # half position update
        V, dVdq = potential_grad(q)
        p -= 2 * a * step_size * dVdq  # `2a` momentum update
    q += step_size * p / (inv_mass_matrix * 2)  # half position update
    V, dVdq = potential_grad(q)
    p -= (1 - 2 * a) * step_size * dVdq  # 1 - 2a position update
    q += step_size * p / (inv_mass_matrix * 2)  # half position update
    V, dVdq = potential_grad(q)
    p -= a * step_size * dVdq  # `a` momentum update

    return q, -p, V, dVdq


def leapfrog_threestage(q, p, dVdq, potential_grad, path_len, step_size, inv_mass_matrix):
    """Perform a single step of a third order symplectic integration scheme.

    Based on the implementation from Adrian Seyboldt in PyMC3. See
    https://github.com/pymc-devs/pymc3/pull/1758 for a discussion.

    References
    ----------
    Blanes, Sergio, Fernando Casas, and J. M. Sanz-Serna. "Numerical
    Integrators for the Hybrid Monte Carlo Method." SIAM Journal on
    Scientific Computing 36, no. 4 (January 2014): A1556-80.
    doi:10.1137/130932740.

    Mannseth, Janne, Tore Selland Kleppe, and Hans J. Skaug. "On the
    Application of Higher Order Symplectic Integrators in
    Hamiltonian Monte Carlo." arXiv:1608.07048 [Stat],
    August 25, 2016. http://arxiv.org/abs/1608.07048.

    Parameters
    ----------
    q : torch.Tensor
        Initial position
    p : torch.Tensor
        Initial momentum
    dVdq : torch.Tensor
        Gradient of the potential at the initial coordinates
    potential_grad : callable
        Value and gradient of the potential
    path_len : float
        How long to integrate for
    step_size : float
        How long each integration step should be

    Returns
    -------
    q, p : ntorch.Tensor, torch.Tensor
        New position and momentum
    """

    a = 12_127_897.0 / 102_017_882
    b = 4_271_554.0 / 14_421_423

    # a step
    p -= a * step_size * dVdq

    for _ in range(int(path_len / step_size) - 1):
        # b step
        q += b * step_size * p / inv_mass_matrix
        V, dVdq = potential_grad(q)
        # (0.5 - a) step
        p -= (0.5 - a) * step_size * dVdq
        # (1 - 2b) step
        q += (1 - 2 * b) * step_size * p / inv_mass_matrix
        V, dVdq = potential_grad(q)
        # (0.5 - a) step
        p -= (0.5 - a) * step_size * dVdq
        # b step
        q += b * step_size * p / inv_mass_matrix
        V, dVdq = potential_grad(q)
        # 2a step
        p -= 2 * a * step_size * dVdq

    # b step
    q += b * step_size * p / inv_mass_matrix
    V, dVdq = potential_grad(q)
    # (0.5 - a) step
    p -= (0.5 - a) * step_size * dVdq
    # (1 - 2b) step
    q += (1 - 2 * b) * step_size * p / inv_mass_matrix
    V, dVdq = potential_grad(q)
    # (0.5 - a) step
    p -= (0.5 - a) * step_size * dVdq
    # b step
    q += b * step_size * p / inv_mass_matrix
    V, dVdq = potential_grad(q)
    # a step
    p -= a * step_size * dVdq

    return q, -p, V, dVdq