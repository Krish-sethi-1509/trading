import os
import unittest
from unittest.mock import patch

from chat_service import ChatServiceError, Source, answer_query, search_market_sources, search_recent_context


class ChatServiceConfigurationTests(unittest.TestCase):
    def test_search_provider_rejects_placeholder_key_and_uses_public_fallback(self):
        environment = {
            "SEARCH_PROVIDER": "tavily",
            "TAVILY_API_KEY": "your_tavily_key",
        }
        source = Source("Gold headline", "https://example.test", "example.test", None, "Excerpt")
        with patch.dict(os.environ, environment, clear=True), patch(
            "chat_service._request_json", side_effect=AssertionError("placeholder key sent")
        ), patch("chat_service.search_public_rss", return_value=[source]) as rss:
            self.assertEqual(search_recent_context("What is moving gold?"), [source])
        rss.assert_called_once()

    def test_greeting_does_not_require_external_provider_credentials(self):
        with patch("chat_service.search_recent_context", side_effect=AssertionError("greetings need no search")):
            result = answer_query("hi")
        self.assertTrue(result["answer"])
        self.assertEqual(result["sources"], [])

    def test_missing_search_key_uses_public_rss_fallback(self):
        source = Source("Gold and yields", "https://example.test/gold", "example.test", None, "Headline excerpt")
        with patch.dict(os.environ, {"SEARCH_PROVIDER": "tavily"}, clear=True), patch(
            "chat_service.search_public_rss", return_value=[source]
        ) as rss:
            result = search_market_sources("news")
        self.assertEqual(result, [source])
        rss.assert_called_once()

    def test_missing_language_model_key_returns_local_source_grounded_answer(self):
        source = Source("Gold and yields", "https://example.test/gold", "example.test", None, "Headline excerpt")
        with patch.dict(os.environ, {}, clear=True), patch(
            "chat_service.search_recent_context", return_value=[source]
        ):
            result = answer_query("What recent macro news is affecting gold?")
        self.assertEqual(result["mode"], "local")
        self.assertIn("[1]", result["answer"])
        self.assertEqual(result["sources"][0]["url"], source.url)

    def test_no_search_results_still_returns_clear_educational_response(self):
        with patch("chat_service.search_recent_context", return_value=[]), patch.dict(os.environ, {}, clear=True):
            result = answer_query("What is moving gold today?")
        self.assertEqual(result["mode"], "local")
        self.assertIn("couldn’t retrieve current headlines", result["answer"])
        self.assertEqual(result["sources"], [])


if __name__ == "__main__":
    unittest.main()
