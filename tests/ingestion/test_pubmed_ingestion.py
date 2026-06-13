import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from llm_data_pretraining.ingestion.pubmed_ingestion import (
    INTERVENTION_PATTERNS,
    IngestionResult,
    PubmedIngestion,
    PubmedIngestionConfig,
)


@pytest.fixture
def tmp_jsonl():
    records = [
        {"id": "1", "text": "This is a clinical trial about drug efficacy.", "title": "Trial 1"},
        {"id": "2", "text": "A study about the history of mathematics.", "title": "Math 1"},
        {"id": "3", "text": "Randomized placebo-controlled study on dosage.", "title": ""},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
        path = Path(f.name)
    yield path
    path.unlink()


@pytest.fixture
def config(tmp_jsonl):
    return PubmedIngestionConfig(
        jsonl_path=tmp_jsonl,
        deepdive_url="http://localhost:8000",
        batch_size=2,
        max_records=None,
        filter_interventions=True,
        concurrency=4,
        extract_title=True,
    )


@pytest.fixture
def config_no_filter(tmp_jsonl):
    return PubmedIngestionConfig(
        jsonl_path=tmp_jsonl,
        filter_interventions=False,
    )


class TestPubmedIngestionConfig:
    def test_missing_file_raises(self):
        with pytest.raises(ValueError, match="JSONL file not found"):
            PubmedIngestionConfig(jsonl_path=Path("/nonexistent/file.jsonl"))

    def test_batch_size_out_of_range_low(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl") as f:
            with pytest.raises(Exception):
                PubmedIngestionConfig(jsonl_path=Path(f.name), batch_size=0)

    def test_batch_size_out_of_range_high(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl") as f:
            with pytest.raises(Exception):
                PubmedIngestionConfig(jsonl_path=Path(f.name), batch_size=257)

    def test_concurrency_out_of_range(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl") as f:
            with pytest.raises(Exception):
                PubmedIngestionConfig(jsonl_path=Path(f.name), concurrency=0)

    def test_defaults(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl") as f:
            cfg = PubmedIngestionConfig(jsonl_path=Path(f.name))
        assert cfg.deepdive_url == "http://localhost:8000"
        assert cfg.batch_size == 32
        assert cfg.max_records is None
        assert cfg.filter_interventions is True
        assert cfg.concurrency == 4
        assert cfg.extract_title is True


class TestInterventionPatterns:
    @pytest.mark.parametrize(
        "text",
        [
            "This clinical trial shows promising results.",
            "The randomized study was conducted.",
            "Patients received a placebo.",
            "The dosage was increased gradually.",
            "We measured the efficacy of the treatment.",
            "Adverse events were recorded.",
            "The treatment group improved significantly.",
            "A new drug was tested.",
            "Gene therapy is advancing.",
            "Phase III results were published.",
            "This meta-analysis combines studies.",
            "A systematic review was performed.",
            "The cohort study followed patients.",
            "Double-blind trial results.",
            "Dose escalation was performed.",
            "Intravenous administration was used.",
        ],
    )
    def test_matches_intervention(self, text):
        assert any(p.search(text) for p in INTERVENTION_PATTERNS)

    @pytest.mark.parametrize(
        "text",
        [
            "The weather was sunny today.",
            "Python is a programming language.",
            "The cat sat on the mat.",
            "Quantum mechanics is fascinating.",
        ],
    )
    def test_no_match_non_medical(self, text):
        assert not any(p.search(text) for p in INTERVENTION_PATTERNS)


class TestPubmedIngestion:
    def test_init(self, config):
        ingestion = PubmedIngestion(config)
        assert ingestion.config is config
        assert ingestion._session is None

    def test_session_not_initialized(self, config):
        ingestion = PubmedIngestion(config)
        with pytest.raises(RuntimeError, match="Session not initialised"):
            _ = ingestion.session

    def test_load_records(self, config):
        ingestion = PubmedIngestion(config)
        records = ingestion.load_records()
        assert len(records) == 3
        assert records[0]["id"] == "1"
        assert records[1]["id"] == "2"
        assert records[2]["id"] == "3"

    def test_load_records_empty_lines(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"id": "1", "text": "hello"}\n')
            f.write("\n")
            f.write('{"id": "2", "text": "world"}\n')
            f.write("   \n")
            path = Path(f.name)
        try:
            cfg = PubmedIngestionConfig(jsonl_path=path)
            ingestion = PubmedIngestion(cfg)
            records = ingestion.load_records()
            assert len(records) == 2
        finally:
            path.unlink()

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("clinical trial on drug efficacy", True),
            ("randomized placebo study", True),
            ("the weather is nice", False),
            ("", False),
            (None, False),
        ],
    )
    def test_is_medical_intervention(self, config, text, expected):
        ingestion = PubmedIngestion(config)
        assert ingestion.is_medical_intervention(text) is expected

    def test_filter_records_with_filter(self, config):
        ingestion = PubmedIngestion(config)
        records = ingestion.load_records()
        filtered = ingestion.filter_records(records)
        assert len(filtered) == 2
        ids = {r["id"] for r in filtered}
        assert "1" in ids
        assert "3" in ids
        assert "2" not in ids

    def test_filter_records_no_filter(self, config_no_filter):
        ingestion = PubmedIngestion(config_no_filter)
        records = ingestion.load_records()
        filtered = ingestion.filter_records(records)
        assert len(filtered) == 3

    def test_filter_records_fallback_to_abstract_text(self, config):
        ingestion = PubmedIngestion(config)
        records = [{"abstract_text": "clinical trial results"}]
        filtered = ingestion.filter_records(records)
        assert len(filtered) == 1

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("The study was conclusive. It showed results.", "The study was conclusive."),
            ("No punctuation here", "No punctuation here"),
            ("First sentence. Second sentence. Third.", "First sentence."),
            ("Short", "Short"),
        ],
    )
    def test_extract_title(self, config, text, expected):
        ingestion = PubmedIngestion(config)
        assert ingestion._extract_title(text) == expected

    def test_extract_title_truncates_200(self, config):
        ingestion = PubmedIngestion(config)
        long_text = "A" * 300
        result = ingestion._extract_title(long_text)
        assert len(result) <= 200

    def test_prepare_payloads(self, config):
        ingestion = PubmedIngestion(config)
        records = [
            {"id": "100", "text": "Abstract text here.", "title": "My Title"},
            {"pmid": "200", "abstract_text": "Another abstract.", "title": ""},
        ]
        payloads = ingestion.prepare_payloads(records)
        assert len(payloads) == 2
        assert payloads[0] == {"pmid": "100", "title": "My Title", "abstract": "Abstract text here."}
        assert payloads[1]["pmid"] == "200"
        assert payloads[1]["abstract"] == "Another abstract."

    def test_prepare_payloads_skips_missing_pmid(self, config):
        ingestion = PubmedIngestion(config)
        records = [{"text": "No id here"}]
        payloads = ingestion.prepare_payloads(records)
        assert len(payloads) == 0

    def test_prepare_payloads_skips_empty_abstract(self, config):
        ingestion = PubmedIngestion(config)
        records = [{"id": "1", "text": ""}]
        payloads = ingestion.prepare_payloads(records)
        assert len(payloads) == 0

    def test_prepare_payloads_extracts_title_when_missing(self, config):
        ingestion = PubmedIngestion(config)
        records = [{"id": "1", "text": "First sentence. Rest of abstract.", "title": ""}]
        payloads = ingestion.prepare_payloads(records)
        assert payloads[0]["title"] == "First sentence."

    def test_prepare_payloads_no_title_extraction(self, tmp_jsonl):
        cfg = PubmedIngestionConfig(jsonl_path=tmp_jsonl, extract_title=False)
        ingestion = PubmedIngestion(cfg)
        records = [{"id": "1", "text": "First sentence. Rest.", "title": ""}]
        payloads = ingestion.prepare_payloads(records)
        assert payloads[0]["title"] == ""

    @pytest.mark.asyncio
    async def test_async_context_manager(self, config):
        async with PubmedIngestion(config) as ingestion:
            assert ingestion._session is not None
        assert ingestion._session is None

    @pytest.mark.asyncio
    async def test_send_batch_success(self, config):
        class MockResponse:
            status = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        mock_session = MagicMock()
        mock_session.post.return_value = MockResponse()

        ingestion = PubmedIngestion(config)
        ingestion._session = mock_session

        payloads = [{"pmid": "1", "title": "T", "abstract": "A"}]
        sent, errors = await ingestion.send_batch(payloads)
        assert sent == 1
        assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_send_batch_http_error(self, config):
        class MockResponse:
            status = 500

            async def text(self):
                return "Internal Server Error"

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        mock_session = MagicMock()
        mock_session.post.return_value = MockResponse()

        ingestion = PubmedIngestion(config)
        ingestion._session = mock_session

        payloads = [{"pmid": "1", "title": "T", "abstract": "A"}]
        sent, errors = await ingestion.send_batch(payloads)
        assert sent == 0
        assert len(errors) == 1
        assert "HTTP 500" in errors[0]

    @pytest.mark.asyncio
    async def test_send_batch_timeout(self, config):
        mock_session = MagicMock()
        mock_session.post.side_effect = asyncio.TimeoutError

        ingestion = PubmedIngestion(config)
        ingestion._session = mock_session

        payloads = [{"pmid": "1", "title": "T", "abstract": "A"}]
        sent, errors = await ingestion.send_batch(payloads)
        assert sent == 0
        assert len(errors) == 1
        assert "timeout" in errors[0]

    @pytest.mark.asyncio
    async def test_send_batch_client_error(self, config):
        import aiohttp

        mock_session = MagicMock()
        mock_session.post.side_effect = aiohttp.ClientError("Connection refused")

        ingestion = PubmedIngestion(config)
        ingestion._session = mock_session

        payloads = [{"pmid": "1", "title": "T", "abstract": "A"}]
        sent, errors = await ingestion.send_batch(payloads)
        assert sent == 0
        assert len(errors) == 1
        assert "Connection refused" in errors[0]

    @pytest.mark.asyncio
    async def test_send_batch_concurrency(self, config):
        class MockResponse:
            status = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        mock_session = MagicMock()
        mock_session.post.return_value = MockResponse()

        ingestion = PubmedIngestion(config)
        ingestion._session = mock_session

        payloads = [{"pmid": str(i), "title": "T", "abstract": "A"} for i in range(8)]
        sent, errors = await ingestion.send_batch(payloads)
        assert sent == 8
        assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_run_empty_filtered(self, tmp_jsonl):
        cfg = PubmedIngestionConfig(
            jsonl_path=tmp_jsonl,
            filter_interventions=True,
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"id": "1", "text": "Not medical at all"}) + "\n")
            path = Path(f.name)
        try:
            cfg = PubmedIngestionConfig(jsonl_path=path, filter_interventions=True)
            async with PubmedIngestion(cfg) as ingestion:
                result = await ingestion.run()
            assert result.success is True
            assert result.total_records == 1
            assert result.filtered_records == 0
            assert result.sent_records == 0
        finally:
            path.unlink()

    @pytest.mark.asyncio
    async def test_run_with_max_records(self, tmp_jsonl):
        cfg = PubmedIngestionConfig(
            jsonl_path=tmp_jsonl,
            filter_interventions=False,
            max_records=2,
        )

        class MockResponse:
            status = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        mock_session = MagicMock()
        mock_session.post.return_value = MockResponse()
        mock_session.close = AsyncMock()

        ingestion = PubmedIngestion(cfg)
        ingestion._session = mock_session

        result = await ingestion.run()

        assert result.total_records == 3
        assert result.filtered_records == 2

    def test_ingestion_result_defaults(self):
        result = IngestionResult(
            success=True,
            total_records=10,
            filtered_records=5,
            sent_records=5,
            failed_records=0,
        )
        assert result.errors == []

    def test_ingestion_result_with_errors(self):
        result = IngestionResult(
            success=False,
            total_records=10,
            filtered_records=5,
            sent_records=3,
            failed_records=2,
            errors=["error1", "error2"],
        )
        assert result.success is False
        assert len(result.errors) == 2


class TestIterRecords:
    def test_yields_all_records(self, config):
        ingestion = PubmedIngestion(config)
        records = list(ingestion.iter_records())
        assert len(records) == 3
        assert records[0]["id"] == "1"
        assert records[1]["id"] == "2"
        assert records[2]["id"] == "3"

    def test_skips_empty_lines(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"id": "1", "text": "hello"}\n')
            f.write("\n")
            f.write('{"id": "2", "text": "world"}\n')
            f.write("   \n")
            path = Path(f.name)
        try:
            cfg = PubmedIngestionConfig(jsonl_path=path)
            ingestion = PubmedIngestion(cfg)
            records = list(ingestion.iter_records())
            assert len(records) == 2
        finally:
            path.unlink()

    def test_stops_on_incomplete_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"id": "1", "text": "hello"}\n')
            f.write('{"id": "2", "text": "partial\n')
            f.write('{"id": "3", "text": "should not appear"}\n')
            path = Path(f.name)
        try:
            cfg = PubmedIngestionConfig(jsonl_path=path)
            ingestion = PubmedIngestion(cfg)
            records = list(ingestion.iter_records())
            assert len(records) == 1
            assert records[0]["id"] == "1"
        finally:
            path.unlink()

    def test_is_generator(self, config):
        import types
        ingestion = PubmedIngestion(config)
        assert isinstance(ingestion.iter_records(), types.GeneratorType)


class TestIterPayloadBatches:
    def test_yields_batches_of_correct_size(      @       `@             $               @             @ @                                  $  �  D  H         � @     @  @�   @                                                      �                 @ � H                A            @                               @  �   @   � 0       ` �    �    @                             �  !  @�                                                  �        �               @   @    @              @
 	                         @     T               T          �       
     @   �                     
     � @   "                     @ �  @               �              @       �          @         @    L    "@                    @      �           �    ��                  �          �        �               �                                 A         @                    @     @     �   �               �@��������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������