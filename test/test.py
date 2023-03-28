from hypothesis import settings, Verbosity, given
from hypothesis import strategies as st
from sympy.ntheory import isprime
import heliotom
  
@given(s=st.integers(min_value=1, max_value=2**10))
@settings(verbosity=Verbosity.normal, max_examples=500)
def test_is_prime(s):
    assert isprime(s) == heliotom.is_prime(s)
  
@given(s=st.integers(min_value=1, max_value=2**10))
@settings(verbosity=Verbosity.normal, max_examples=500)
def test_is_even(s):
    assert (s % 2 == 0) == heliotom.is_even(s)


if __name__ == "__main__":
       test_is_prime()
       test_is_even()
