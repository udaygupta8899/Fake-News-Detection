import unittest
from flask_app.app import app

class FakeNewsAppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_home_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<title>Fake News Detector</title>', response.data)

    def test_predict_page_real_or_fake(self):
        response = self.client.post('/predict', data=dict(
            title="NASA discovers new habitable planet",
            text="Scientists from NASA have discovered a new Earth-like planet in the habitable zone..."
        ))
        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            b'Real News' in response.data or b'Fake News' in response.data,
            "Response should contain either 'Real News' or 'Fake News'"
        )

if __name__ == '__main__':
    unittest.main()
