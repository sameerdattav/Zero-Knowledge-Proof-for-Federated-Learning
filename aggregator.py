from ivc import IVCProver

class Aggregator:
    def __init__(self):
        self.prev_proof = []
        self.prev_state = (0.0, 0.0)
        self.recursive_proof = None
        self.recursive_commit = None

    def ingest_client_proof(self, client_proof, client_state):
        ivc = IVCProver(self.prev_proof, client_proof, self.prev_state, client_state)
        new_proof, commit = ivc.prove()

        self.prev_proof = client_proof
        self.prev_state = client_state
        self.recursive_proof = new_proof
        self.recursive_commit = commit

        return new_proof, commit
