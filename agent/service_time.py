"""MLB service time normalization, used by the deterministic phase resolver."""

import pandas as pd

# Days needed for one year of MLB service time
DAYS_PER_SERVICE_YEAR = 172


def normalize_service_time(service_time):
    """
    Normalize service time from years.days format to linear scale.

    MLB service time is encoded as years.days where days is 0-172.
    For example, 2.028 means 2 years and 28 days.

    This converts to a linear scale where the fractional part represents
    the proportion of a service year (0-0.99 instead of 0-0.172).

    Examples:
        2.028 -> 2 + (28/172) = 2.163
        2.100 -> 2 + (100/172) = 2.581
        2.170 -> 2 + (170/172) = 2.988

    Args:
        service_time: Service time in years.days format

    Returns:
        Normalized service time on linear scale
    """
    if pd.isna(service_time):
        return service_time

    years = int(service_time)
    # The decimal part encodes days (e.g., 0.028 = 28 days, 0.100 = 100 days)
    days = round((service_time - years) * 1000)
    normalized = years + (days / DAYS_PER_SERVICE_YEAR)
    return normalized
