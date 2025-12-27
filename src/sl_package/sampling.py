from typing import Tuple, Union, Sequence, Any
import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.stats import gaussian_kde

def generate_distribution(
    data_points: Union[Sequence[float], NDArray],
    method: str = 'kde',
    num_points: int = 1000
) -> Tuple[NDArray, NDArray, Any]:
    """Generate a probability distribution from sparse data points.

    Args:
        data_points: The observed data points
        method: Distribution fitting method - 'kde' for Kernel Density Estimation
                or 'gaussian' for fitting normal distribution
        num_points: Number of points to generate for the distribution

    Returns:
        A tuple containing:
        - x values (array)
        - probability density values (array)
        - fitted distribution object (KDE or scipy.stats distribution)

    Examples:
        >>> data = [1, 2, 3, 4, 5]
        >>> x, density, dist = generate_distribution(data, method='kde')
        >>> print(len(x))
        1000
    """
    data_points = np.array(data_points)
    
    if method == 'kde':
        # Kernel Density Estimation
        kde = gaussian_kde(data_points)
        x_range = np.linspace(min(data_points), max(data_points), num_points)
        density = kde(x_range)
        return x_range, density, kde
        
    elif method == 'gaussian':
        # Fit normal distribution
        mu, std = stats.norm.fit(data_points)
        x_range = np.linspace(min(data_points), max(data_points), num_points)
        density = stats.norm.pdf(x_range, mu, std)
        return x_range, density, stats.norm(mu, std)
    
def generate_samples(
    distribution: Any,
    num_samples: int = 100,
    method: str = 'kde'
) -> NDArray:
    """Generate random samples from the fitted distribution.

    Args:
        distribution: KDE or scipy.stats distribution object
        num_samples: Number of samples to generate
        method: Distribution type - 'kde' or 'gaussian'

    Returns:
        Array of random samples from the distribution

    Examples:
        >>> data = [1, 2, 3, 4, 5]
        >>> x, density, dist = generate_distribution(data, method='kde')
        >>> samples = generate_samples(dist, num_samples=10, method='kde')
        >>> print(len(samples))
        10
    """
    if method == 'kde':
        return distribution.resample(num_samples)[0]
    else:
        return distribution.rvs(size=num_samples)
    
def sample_from_distribution(
    mean: float,
    std_dev: float,
    num_samples: int = 1
) -> NDArray:
    """Generate samples from a normal distribution.

    Args:
        mean: Mean of the normal distribution
        std_dev: Standard deviation of the normal distribution
        num_samples: Number of samples to generate

    Returns:
        Array of samples from the normal distribution

    Examples:
        >>> samples = sample_from_distribution(mean=0, std_dev=1, num_samples=100)
        >>> print(len(samples))
        100
    """
    samples = np.random.normal(loc=mean, scale=std_dev, size=num_samples)
    return samples