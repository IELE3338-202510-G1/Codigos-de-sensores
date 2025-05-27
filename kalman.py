import numpy as np

class KalmanFilter:
    def __init__(self, dt, L=1.0):
        self.dt = dt      # Paso de tiempo
        self.L = L        # Longitud entre ejes (para modelo tipo Ackermann)

        # Estado: [x, y, theta]
        self.x = np.zeros((3, 1))  # Estimación del estado
        self.P = np.eye(3) * 1     # Covarianza del estado

        # Matrices del modelo
        self.F = np.eye(3)         # Matriz de transición
        self.B = np.zeros((3, 2))  # Matriz de control [v, delta]
        self.H = np.eye(3)         # Matriz de observación

        # Ruido
        self.Q = np.eye(3) * 0.01  # Ruido del proceso
        self.R = np.eye(3) * 0.1   # Ruido de mediciones

        # Variables intermedias
        self.y = np.zeros((3, 1))  # Innovación (residual)
        self.S = np.zeros((3, 3))  # Covarianza de la innovación
        self.K = np.zeros((3, 3))  # Ganancia de Kalman

    def prediction_step(self, u):
        v, delta = u
        theta = self.x[2, 0]



        # Modelo cinemático (Ackermann simplificado)
        self.B = np.array([
            [np.cos(theta) * self.dt, 0],
            [np.sin(theta) * self.dt, 0],
            [np.tan(delta) * self.dt / self.L, 0]
        ])

        self.F = np.eye(3)  # Lineal en el filtro estándar

        # Predicción del estado
        self.x = self.F @ self.x + self.B @ np.array([[v], [delta]])
        self.P = self.F @ self.P @ self.F.T + self.Q

    def observation_step(self, z):
        """
        Cálculo de la innovación (residual) y su covarianza.
        """
        z = z.reshape((3, 1))
        self.y = z - self.H @ self.x
        self.S = self.H @ self.P @ self.H.T + self.R

    def update_step(self):
        """
        Corrección del estado con la observación.
        """
        self.K = self.P @ self.H.T @ np.linalg.inv(self.S)
        self.x = self.x + self.K @ self.y
        self.P = (np.eye(3) - self.K @ self.H) @ self.P

    def get_state(self):
        return self.x.flatten()

kf = KalmanFilter(dt=0.1)

# Paso 1: predicción con entrada de control [velocidad, ángulo de giro]
kf.prediction_step([1.0, 0.1])

# Paso 2: observación (mediciones ruidosas de [x, y, theta])
kf.observation_step(np.array([0.95, 0.12, 0.03]))

# Paso 3: actualización del estado
kf.update_step()

# Estado final estimado
print("Estado estimado:", kf.get_state())
