"""The optional TTL cache: off by default, keyed on everything that matters."""

from __future__ import annotations

from llm_client import LLMClient
from llm_client.cache import ResponseCache


class TestDefaults:
    def test_caching_is_off_unless_asked_for(self, fake_openai):
        client = LLMClient("openai")
        client.complete("hi")
        client.complete("hi")
        assert fake_openai.call_count == 2
        assert client.cache_stats() == {
            "enabled": False,
            "current_size": 0,
            "max_size": 0,
            "ttl_seconds": 0.0,
        }


class TestHits:
    def test_an_identical_call_is_served_from_the_cache(self, fake_openai):
        client = LLMClient("openai", cache=True)
        first = client.complete("hi")
        second = client.complete("hi")
        assert fake_openai.call_count == 1
        assert first.text == second.text

    def test_a_cache_hit_reports_no_usage(self, fake_openai):
        client = LLMClient("openai", cache=True)
        client.complete("hi")
        cached = client.complete("hi")
        assert cached.usage.total_tokens is None
        assert cached.raw is None
        assert (cached.provider, cached.model) == ("openai", "gpt-4o")

    def test_different_prompts_do_not_collide(self, fake_openai):
        client = LLMClient("openai", cache=True)
        client.complete("one")
        client.complete("two")
        assert fake_openai.call_count == 2
        assert client.cache_stats()["current_size"] == 2

    def test_parameters_are_part_of_the_key(self, fake_openai):
        client = LLMClient("openai", cache=True)
        client.complete("hi")
        client.complete("hi", temperature=0.9)
        client.complete("hi", json_mode=True)
        client.complete("hi", system="be terse")
        assert fake_openai.call_count == 4

    def test_two_clients_do_not_share_a_cache(self, fake_openai):
        LLMClient("openai", cache=True).complete("hi")
        LLMClient("openai", cache=True).complete("hi")
        assert fake_openai.call_count == 2

    def test_clear_empties_the_cache(self, fake_openai):
        client = LLMClient("openai", cache=True)
        client.complete("hi")
        assert client.cache_stats()["current_size"] == 1
        client.clear_cache()
        assert client.cache_stats()["current_size"] == 0
        client.complete("hi")
        assert fake_openai.call_count == 2

    def test_maxsize_is_honoured(self, fake_openai):
        client = LLMClient("openai", cache=True, cache_maxsize=2)
        for prompt in ("a", "b", "c"):
            client.complete(prompt)
        assert client.cache_stats()["current_size"] == 2

    async def test_async_calls_share_the_cache(self, fake_openai):
        client = LLMClient("openai", cache=True)
        client.complete("hi")
        assert (await client.acomplete("hi")).text == "hi"
        assert fake_openai.call_count == 1


class TestKey:
    def test_the_key_ignores_dict_ordering(self):
        turns = [{"role": "user", "content": "hi"}]
        one = ResponseCache.key("p", "m", turns, None, {"a": 1, "b": 2})
        two = ResponseCache.key("p", "m", turns, None, {"b": 2, "a": 1})
        assert one == two

    def test_provider_and_model_are_in_the_key(self):
        turns = [{"role": "user", "content": "hi"}]
        assert ResponseCache.key("a", "m", turns, None, {}) != ResponseCache.key(
            "b", "m", turns, None, {}
        )
        assert ResponseCache.key("a", "m1", turns, None, {}) != ResponseCache.key(
            "a", "m2", turns, None, {}
        )
