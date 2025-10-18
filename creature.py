
import numpy as np 

class Creature:
    def __init__(self, structure:list[int], id:int):
        self.id = id
        self.structure = structure
        self.neural_network = self.initialisation()
        self.DNA = self.createDNA()




    def initialisation(self):
        """Initialise the neural network parameters with random weights and zero biases."""

        parametres = {}
        C = len(self.structure)

        for c in range(1, C):
            parametres['W' + str(c)] = np.random.randn(self.structure[c], self.structure[c - 1]) 
            parametres['b' + str(c)] = np.zeros((self.structure[c], 1))

        return parametres
        

    def createDNA (self): 
        """Encode the current neural network parameters into a DNA string.

        The DNA format is a compact base64 string encoding the bytes of all
        weights and biases in a deterministic order, preceded by a simple
        header that lists the layer sizes so a DNA string can be self-contained.

        Returns:
            str: DNA string
        """
        import base64

        # header: comma-separated layer sizes
        header = ','.join(map(str, self.structure))

        # collect bytes for all parameters in order W1,b1,W2,b2,...
        parts = []
        for c in range(1, len(self.structure)):
            W = self.neural_network['W' + str(c)]
            b = self.neural_network['b' + str(c)]
            parts.append(W.astype(np.float32).tobytes())
            parts.append(b.astype(np.float32).tobytes())

        body = b''.join(parts)
        body_b64 = base64.urlsafe_b64encode(body).decode('ascii')

        dna = f"{header}|{body_b64}"
        self.DNA = dna
        return dna

    def loadDNA(self, dna: str):
        """Decode a DNA string and load it into self.neural_network.

        The dna string must have the format produced by createDNA: header|body_b64
        where header is comma-separated layer sizes.
        """
        import base64

        header, body_b64 = dna.split('|', 1)
        sizes = list(map(int, header.split(',')))
        # if structure differs, update it
        if sizes != self.structure:
            self.structure = sizes

        body = base64.urlsafe_b64decode(body_b64.encode('ascii'))

        params = {}
        offset = 0
        for c in range(1, len(self.structure)):
            rows = self.structure[c]
            cols = self.structure[c - 1]
            w_bytes = rows * cols * 4  # float32
            b_bytes = rows * 1 * 4
            W = np.frombuffer(body[offset:offset + w_bytes], dtype=np.float32).reshape((rows, cols))
            offset += w_bytes
            b = np.frombuffer(body[offset:offset + b_bytes], dtype=np.float32).reshape((rows, 1))
            offset += b_bytes
            params['W' + str(c)] = W.astype(np.float64)
            params['b' + str(c)] = b.astype(np.float64)

        self.neural_network = params
        self.DNA = dna
        return params

