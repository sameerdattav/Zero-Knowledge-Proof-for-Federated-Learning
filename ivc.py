import numpy as np
import hashlib
from sumcheck import SumcheckProver, SumcheckVerifier

FIELD_MOD = 31
LOG_N = 3  # proving over 8-element vectors (2^3)


def hash_bytes(b: bytes) -> int:
    return int(hashlib.sha256(b).hexdigest(), 16) % FIELD_MOD


class IVCProver:
    def __init__(self, prev_proof, curr_proof, prev_state, curr_state):
        self.prev_proof = prev_proof
        self.curr_proof = curr_proof
        self.prev_w, self.prev_b = prev_state
        self.curr_w, self.curr_b = curr_state
        self.field = FIELD_MOD

    def _build_vector(self):
        vec = [
            int(self.prev_w) % self.field,
            int(self.prev_b) % self.field,
            int(self.curr_w) % self.field,
            int(self.curr_b) % self.field,
        ]
        prev_hash = hash_bytes(str(self.prev_proof).encode())
        curr_hash = hash_bytes(str(self.curr_proof).encode())
        vec += [prev_hash, curr_hash]
        vec += [0] * (2**LOG_N - len(vec))
        return np.array(vec, dtype=int) % self.field

    def prove(self):
        vec = self._build_vector()
        prover = SumcheckProver(vec.tolist(), field_mod=self.field)
        unis = prover.sumcheck_round()
        self.commitment = prover.commit()
        return unis, self.commitment

    @staticmethod
    def claimed_sum(prev_state, curr_state, prev_proof, curr_proof):
        vec = IVCProver(None, None, prev_state, curr_state)._build_vector()
        return int(vec.sum()) % FIELD_MOD


class IVCVerifier:
    def __init__(self, claimed_sum):
        self.claimed_sum = int(claimed_sum) % FIELD_MOD
        self.challenges = np.random.randint(0, 2, size=LOG_N)  # bits

    def verify(self, unis):
        current = self.claimed_sum
        if len(unis) != LOG_N:
            return False
        for i, uni in enumerate(unis):
            if (int(uni[0]) + int(uni[1])) % FIELD_MOD != current:
                return False
            r = int(self.challenges[i])
            current = int(uni[r]) % FIELD_MOD
        return True
