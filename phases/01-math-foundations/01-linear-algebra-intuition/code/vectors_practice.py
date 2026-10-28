class Vector:
    def __init__(self, components):
        self.components = list(components)
        self.dim = len(self.components)

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.components, other.components)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.components, other.components)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.components])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.components, other.components))

    def magnitude(self):
        return sum(x**2 for x in self.components) ** 0.5

    def normalize(self):
        mag = self.magnitude()
        return Vector([x / mag for x in self.components])

    def cosine_similarity(self, other):
        return self.dot(other) / (self.magnitude() * other.magnitude())

    def angle_between(self, other):
        import math

        cos_theta = self.cosine_similarity(other)
        cos_theta = max(-1.0, min(1.0, cos_theta))
        return math.degrees(math.acos(cos_theta))

    def project_onto(self, other):
        scalar = self.dot(other) / other.dot(other)
        return Vector([scalar * x for x in other.components])

    def __repr__(self):
        return f"Vector({self.components})"


def is_independent(vectors):
    n = len(vectors)
    if n == 0:
        return True
    dim = vectors[0].dim
    rows = [v.components[:] for v in vectors]
    rank = 0
    for col in range(dim):
        pivot = None
        for row in range(rank, len(rows)):
            if abs(rows[row][col]) > 1e-10:
                pivot = row
                break
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        scale = rows[rank][col]
        rows[rank] = [x / scale for x in rows[rank]]
        for row in range(len(rows)):
            if row != rank and abs(rows[row][col]) > 1e-10:
                factor = rows[row][col]
                rows[row] = [rows[row][j] - factor * rows[rank][j] for j in range(dim)]
        rank += 1
    return rank == n


def gram_schmidt(vectors):
    orthonormal = []
    for v in vectors:
        w = v
        for u in orthonormal:
            proj = w.project_onto(u)
            w = w - proj
        if w.magnitude() < 1e-10:
            continue
        orthonormal.append(w.normalize())
    return orthonormal


class Matrix:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]
        self.shape = (len(self.rows), len(self.rows[0]))

    def __matmul__(self, other):
        if isinstance(other, Vector):
            return Vector(
                [
                    sum(
                        self.rows[i][j] * other.components[j]
                        for j in range(self.shape[1])
                    )
                    for i in range(self.shape[0])
                ]
            )
        rows = []
        for i in range(self.shape[0]):
            row = []
            for j in range(other.shape[1]):
                row.append(
                    sum(
                        self.rows[i][k] * other.rows[k][j] for k in range(self.shape[1])
                    )
                )
            rows.append(row)
        return Matrix(rows)

    def transpose(self):
        return Matrix(
            [
                [self.rows[j][i] for j in range(self.shape[0])]
                for i in range(self.shape[1])
            ]
        )

    def rank(self):
        rows = [row[:] for row in self.rows]
        m, n = self.shape
        r = 0
        for col in range(n):
            pivot = None
            for row in range(r, m):
                if abs(rows[row][col]) > 1e-10:
                    pivot = row
                    break
            if pivot is None:
                continue
            rows[r], rows[pivot] = rows[pivot], rows[r]
            scale = rows[r][col]
            rows[r] = [x / scale for x in rows[r]]
            for row in range(m):
                if row != r and abs(rows[row][col]) > 1e-10:
                    factor = rows[row][col]
                    rows[row] = [rows[row][j] - factor * rows[r][j] for j in range(n)]
            r += 1
        return r

    def __repr__(self):
        return f"Matrix({self.rows})"


if __name__ == "__main__":
    # Exercise 1:
    v1 = Vector([1, 0])
    v2 = Vector([0, 1])
    print(v1.angle_between(v2))  # 90.0 degrees
    print(v1.angle_between(v1))  # 0.0

    # Exercise 2:
    # Scaling matrix: doubles x (diag 0 = 2), triples y (diag 1 = 3)
    S = Matrix([[2, 0], [0, 3]])
    v = Vector([1, 1])
    result = S @ v
    print(result)  # Vector([2, 3])

    # Exercise 3:
    import random

    random.seed(42)
    # 5 random vectors in 50 dimensions
    vectors = [Vector([random.gauss(0, 1) for _ in range(50)]) for _ in range(5)]
    best_sim = -2.0
    best_pair = None
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            sim = vectors[i].cosine_similarity(vectors[j])
            if sim > best_sim:
                best_sim = sim
                best_pair = (i, j)
    print(
        f"Most similar pair: Vector {best_pair[0]} and Vector {best_pair[1]} (similarity: {best_sim:.4f})"
    )

    # Exercise 4:
    u1 = Vector([1, 1, 0])
    u2 = Vector([1, 0, 1])
    u3 = Vector([0, 1, 1])
    basis = gram_schmidt([u1, u2, u3])

    # 1. Check dot product is ~0 for all distinct pairs (orthogonal)
    for i in range(len(basis)):
        for j in range(i + 1, len(basis)):
            dot_prod = basis[i].dot(basis[j])
            assert abs(dot_prod) < 1e-9, f"Pair ({i},{j}) not orthogonal: {dot_prod}"

    # 2. Check magnitude is ~1 for each vector (normalized)
    for i, vec in enumerate(basis):
        assert (
            abs(vec.magnitude() - 1.0) < 1e-9
        ), f"Vector {i} not unit length: {vec.magnitude()}"

    print("All vectors are verified orthonormal within 1e-9 tolerance!")

    # Exercise 5:
    # Make row 3 a linear combination of rows 1 and 2 (e.g. row 1 + row 2)
    M = Matrix([[1, 0, 1], [0, 1, 1], [1, 1, 2]])
    print(f"Matrix rank: {M.rank()}")  # Outputs 2

    # Exercise 6:
    a = Vector([1, 2, 3])
    b = Vector([1, 1, 1])
    proj = a.project_onto(b)
    residual = a - proj
    print(f"Projection: {proj}")  # Vector([2.0, 2.0, 2.0])
    print(f"Residual: {residual}")  # Vector([-1.0, 0.0, 1.0])
    print(f"Residual · b: {residual.dot(b)}")  # 0.0
