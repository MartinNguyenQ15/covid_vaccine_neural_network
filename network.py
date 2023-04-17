class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None
        self.err = 0

    def add (self, layer):
        self.layers.append(layer)

    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input_data):
        samples = len(input_data)
        result = []

        for i in range(samples):
            forward_data = input_data[i]
            for layer in self.layers:
                forward_data = layer.forward_propagation(forward_data)
            result.append(forward_data)
        
        return result

    def fit(self, x, y, steps, learning_rate):
        samples_per_step = len(x)

        for _ in range(steps):
            for j in range(samples_per_step):
                forward = x[j]
                for layer in self.layers:
                    forward = layer.forward_propagation(forward)
                self.err += self.loss(y[j], forward)
                backwards = self.loss_prime(y[j], forward)
                for layer in reversed(self.layers):
                    backwards = layer.backwards_propagation(backwards, learning_rate)
            self.err /= samples_per_step
