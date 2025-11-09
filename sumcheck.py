import numpy as np
import hashlib

class SumcheckProver:
    def __init__(self, poly_coeffs, field_mod=31):
        self.poly_coeffs = np.array(poly_coeffs) % field_mod
        self.field_mod = field_mod
        self.log_n = int(np.log2(len(self.poly_coeffs)))
        assert 2 ** self.log_n == len(self.poly_coeffs), "poly_coeffs length must be power of two"

    def multilinear_eval(self, r):
        # Pad r with zeros if it's shorter than log_n
        r = list(r) + [0] * (self.log_n - len(r))

        result = 0
        for i in range(len(self.poly_coeffs)):
            prod = int(self.poly_coeffs[i])
            bin_i = format(i, f'0{self.log_n}b')
            for j in range(self.log_n):
                bit = int(bin_i[j])
                # linear blending for multilinear extension (toy)
                prod = (prod * (bit * r[j] + (1 - bit) * (1 - r[j]))) % self.field_mod
            result = (result + prod) % self.field_mod
        return result

    def sumcheck_round(self, prev_r=None):
        """
        Produce a list of univariate evaluations per round.
        Each round returns [uni(0), uni(1)] where uni(b) is the sum of
        poly_coeffs over assignments with current variable = b and previous fixed.
        """
        if prev_r is None:
            prev_r = []

        unis = []
        for round_num in range(self.log_n):
            uni_coeffs = [0, 0]
            # build full r for each branch (pad remaining with zeros)
            for x in [0, 1]:
                r_full = prev_r + [x] + [0] * (self.log_n - len(prev_r) - 1)
                uni_coeffs[x] = int(self.multilinear_eval(r_full))
            unis.append(uni_coeffs)
            # Note: prover here returns all rounds; in an interactive protocol
            # prev_r would be extended with verifier's challenge each time.
        return unis

    def commit(self):
        return hashlib.sha256(str(self.poly_coeffs.tolist()).encode()).hexdigest()


class SumcheckVerifier:
    def __init__(self, claimed_sum, log_n, field_mod=31, rng=None):
        self.claimed_sum = int(claimed_sum) % field_mod
        self.log_n = log_n
        self.field_mod = field_mod
        self.rng = np.random.default_rng() if rng is None else rng
        # toy: use challenge bits {0,1}
        self.challenges = self.rng.integers(0, 2, size=log_n)

    def verify(self, unis):
        if len(unis) != self.log_n:
            return False
        current_sum = self.claimed_sum
        for round_num in range(self.log_n):
            uni = unis[round_num]
            # check uni(0)+uni(1) == current_sum (mod field)
            if (int(uni[0]) + int(uni[1])) % self.field_mod != current_sum:
                return False
            # reduce according to challenge bit
            challenge = int(self.challenges[round_num])
            current_sum = int(uni[challenge]) % self.field_mod
        return True

