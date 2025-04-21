from scripts import metrics
from unittest import TestCase

class MetricsTestCase(TestCase):

  def test_fr(self):
    """
    target : source: jumped?
    0: 0 : No (first)
    1: - : Yes
    2: 3 : Yes
    3: 4 : No
    4: - : Yes
    5: 1 : Yes
    6: 1 : No
    7: 5 : Yes

    # c+1 = 5
    # m = 8
    # 1 - (5)/(8-1) = 0.28571428571
    """
    r = metrics.fuzzy_reordering([
      (0, 0),
      (1, 5),
      (1, 6),
      (3, 2),
      (4, 3),
      (5, 7)],
      6,
      8
    )
    self.assertAlmostEqual(.28571428571, r)

  def test_fr_identical(self):
    """Fuzzy reordering of perfectly aligned sentences is 1."""
    r = metrics.fuzzy_reordering([
      (0, 0),
      (1, 1),
      (2, 2),
      (3, 3),
      (4, 4),
      (5, 5)],
      6,
      6
    )
    self.assertAlmostEqual(1.0, r)

  def test_fr_reversed(self):
      """Fuzzy reordering of perfectly aligned sentences is 1."""
      r = metrics.fuzzy_reordering([
        (0, 5),
        (1, 4),
        (2, 3),
        (3, 2),
        (4, 1),
        (5, 0)],
        6,
        6
      )
      self.assertAlmostEqual(0.0, r)