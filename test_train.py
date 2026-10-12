import unittest

from train import train_model


class TestStudentPlacementModel(unittest.TestCase):

    def test_model_training(self):
        model, accuracy = train_model()

        self.assertIsNotNone(model)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
