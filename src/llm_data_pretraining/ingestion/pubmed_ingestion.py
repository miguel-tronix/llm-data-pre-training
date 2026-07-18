import asyncio
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import aiohttp
from pydantic import BaseModel, Field, field_validator

from llm_data_pretraining.utils.pipeline_logger import get_pipeline_logger

logger = get_pipeline_logger()

HTTP_OK = 200

INTERVENTION_PATTERNS: list[re.Pattern] = [
    re.compile(r, re.IGNORECASE)
    for r in [
        r"\bclinical trial\b",
        r"\brandomized\b",
        r"\bplacebo\b",
        r"\bdosage\b",
        r"\bdose[- ]?(?:finding|response|dependent|escalation)?\b",
        r"\bpharmacokinetic",
        r"\befficacy\b",
        r"\bsafety\b",
        r"\badverse event",
        r"\bside effect",
        r"\btreatment group\b",
        r"\bdrug\b",
        r"\bmedication\b",
        r"\btherapeutic\b",
        r"\bchemotherapy\b",
        r"\bantiviral\b",
        r"\bantibiotic\b",
        r"\bantihypertensive\b",
        r"\bantidepressant\b",
        r"\bantipsychotic\b",
        r"\bimmunosuppressant\b",
        r"\badjuvant\b",
        r"\bneoadjuvant\b",
        r"\bsurgery\b",
        r"\bsurgical\b",
        r"\boperation\b",
        r"\bresection\b",
        r"\btransplant\b",
        r"\bimplant\b",
        r"\bbypass\b",
        r"\blaparoscopic\b",
        r"\bendoscopic\b",
        r"\bexcision\b",
        r"\bablation\b",
        r"\barthroplasty\b",
        r"\bradiation\b",
        r"\bradiotherapy\b",
        r"\bphysical therapy\b",
        r"\bgene therapy\b",
        r"\bimmunotherapy\b",
        r"\bvaccin\w+\b",
        r"\bdialysis\b",
        r"\bventilation\b",
        r"\bintubation\b",
        r"\bcatheter",
        r"\bstent\b",
        r"\bpacemaker\b",
        r"\bprosthes[ise]",
        r"\bdefibrillator\b",
        r"\bintervention\b",
        r"\btreatment\b",
        r"\btherapy\b",
        r"\bmanagement of\b",
        r"\badministered\b",
        r"\binjection\b",
        r"\binfusion\b",
        r"\btransfusion\b",
        r"\boutcome\b",
        r"\bsurvival\b",
        r"\bmortality\b",
        r"\bremission\b",
        r"\brecovery\b",
        r"\bprognos\w+\b",
        r"\bphase [ivxl]+",
        r"\bphase \d",
        r"\bmeta-analysis\b",
        r"\bsystematic review\b",
        r"\bcohort study\b",
        r"\bprospective\b",
        r"\bretrospective\b",
        r"\bcardiopulmonary resuscitation\b",
        r"\bmechanical ventilation\b",
        r"\bextracorporeal\b",
        r"\bhemodialysis\b",
        r"\bpercutaneous\b",
        r"\bendarterectomy\b",
        r"\bthrombectomy\b",
        r"\bembolization\b",
        r"\bradiofrequency ablation\b",
        r"\bcryoablation\b",
        r"\bphotocoagulation\b",
        r"\blaser therapy\b",
        r"\bphotodynamic therapy\b",
        r"\bhormone therapy\b",
        r"\btargeted therapy\b",
        r"\bbiological therapy\b",
        r"\bstem cell\b",
        r"\bplatelet-rich plasma\b",
        r"\boccupational therapy\b",
        r"\bspeech therapy\b",
        r"\bcognitive behavioral\b",
        r"\bcontrolled trial\b",
        r"\bdouble-blind\b",
        r"\bdose escalation\b",
        r"\bdose[- ]?response\b",
        r"\bdose limiting\b",
        r"\bmaximum tolerated\b",
        r"\brecommended phase\b",
        r"\boral administration\b",
        r"\bintravenous\b",
        r"\bsubcutaneous\b",
    ]
]


class PubmedIngestionConfig(BaseModel):
    jsonl_path: Path = Field(..., description="Path to cleaned PubMed JSONL file")
    deepdive_url: str = Field(
        default="http://localhost:8000",
        description="DeepDive API base URL",
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        le=256,
        description="Records per batch",
    )
    max_records: int | None = Field(
        default=None,
        description="Max records to process (None = all)",
    )
    filter_interventions: bool = Field(
        default=True,
        description="Filter for medical intervention abstracts",
    )
    concurrency: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Max concurrent HTTP requests",
    )
    extract_title: bool = Field(
        default=True,
        description="Extract title from first sentence if absent",
    )

    model_config = {"arbitrary_types_allowed": False}

    @field_validator("jsonl_path")
    @classmethod
    def validate_jsonl_path(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"JSONL file not found: {v}")
        return v


class IngestionResult(BaseModel):
    success: bool
    total_records: int
    filtered_records: int
    sent_records: int
    failed_records: int
    errors: list[str] = []


class PubmedIngestion:
    def __init__(self, config: PubmedIngestionConfig):
        self.config = config
        self._session: aiohttp.ClientSession | None = None

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None:
            raise RuntimeError("Session not initialised. Use async with or call run()")
        return self._session

    async def __aenter__(self) -> "PubmedIngestion":
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    def load_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with open(self.config.jsonl_path, encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    records.append(json.loads(stripped))
        return records

    def iter_records(self) -> Iterator[dict[str, Any]]:
        with open(self.config.jsonl_path, encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    yield json.loads(stripped)
                except json.JSONDecodeError:
                    logger.warning(
                        f"Skipping incomplete line at EOF: {stripped[:120]}..."
                    )
                    break

    def iter_payload_batches(
        self,
    ) -> Iterator[tuple[list[dict[str, str]], int]]:
        batch: list[dict[str, str]] = []
        segment_read = 0
        cumulative_read = 0
        yielded = False

        for rec in self.iter_records():
            if (
                self.config.max_records is not None
                and cumulative_read >= self.config.max_records
            ):
                break

            cumulative_read += 1
            segment_read += 1

            if self.config.filter_interventions:
                text = rec.get("text") or rec.get("abstract_text", "")
                if not self.is_medical_intervention(text):
                    continue

            abstract = rec.get("text") or rec.get("abstract_text", "")
            pmid = rec.get("id") or rec.get("pmid", "")
            title = rec.get("title", "")

            if not title and self.config.extract_title:
                title = self._extract_title(abstract)

            if pmid and abstract:
                batch.append(
                    {
                        "pmid": str(pmid),
                        "title": title,
                        "abstract": abstract,
                    }
                )

            if len(batch) >= self.config.batch_size:
                yield batch, segment_read
                yielded = True
                batch = []
                segment_read = 0

        if batch:
            yield batch, segment_read
        elif cumulative_read > 0 and not yielded:
            yield batch, segment_read

    def is_medical_intervention(self, text: str) -> bool:
        if not text:
            return False
        return any(pattern.search(text) for pattern in INTERVENTION_PATTERNS)

    def filter_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.config.filter_interventions:
            return records

        filtered = []
        for rec in records:
            text = rec.get("text") or rec.get("abstract_text", "")
            if self.is_medical_intervention(text):
                filtered.append(rec)
        logger.info(
            f"Filtered {len(records)} → {len(filtered)} intervention-related records"
        )
        return filtered

    def _extract_title(self, text: str) -> str:
        match = re.match(r"^(.+?[.!?])\s", text)
        if match:
            title = match.group(1)
        else:
            title = text.split(".", maxsplit=1)[0] if "." in text else text[:200]
        return title.strip()[:200]

    def prepare_payloads(self, records: list[dict[str, Any]]) -> list[dict[str, str]]:
        payloads = []
        for rec in records:
            abstract = rec.get("text") or rec.get("abstract_text", "")
            pmid = rec.get("id") or rec.get("pmid", "")
            title = rec.get("title", "")

            if not title and self.config.extract_title:
                title = self._extract_title(abstract)

            if pmid and abstract:
                payloads.append(
                    {
                        "pmid": str(pmid),
                        "title": title,
                        "abstract": abstract,
                    }
                )
        return payloads

    async def send_batch(self, payloads: list[dict[str, str]]) -> tuple[int, list[str]]:
        semaphore = asyncio.Semaphore(self.config.concurrency)
        errors: list[str] = []
        sent = 0

        async def send_one(payload: dict[str, str]) -> None:
            nonlocal sent
            async with semaphore:
                try:
                    async with self.session.post(
                        f"{self.config.deepdive_url}/api/embeddings",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as resp:
                        if resp.status == HTTP_OK:
                            sent += 1
                        else:
                            body = await resp.text()
                            errors.append(
                                f"PMID {payload['pmid']}: "
                                f"HTTP {resp.status} - {body[:200]}"
                            )
                except asyncio.TimeoutError:
                    errors.append(f"PMID {payload['pmid']}: timeout")
                except aiohttp.ClientError as e:
                    errors.append(f"PMID {payload['pmid']}: {e}")

        tasks = [send_one(p) for p in payloads]
        await asyncio.gather(*tasks)
        return sent, errors

    async def run(self) -> IngestionResult:
        logger.info(f"Loading records from {self.config.jsonl_path}")

        all_records = self.load_records()
        total = len(all_records)
        logger.info(f"Loaded {total} records")

        filtered = self.filter_records(all_records)
        if self.config.max_records:
            filtered = filtered[: self.config.max_records]

        if not filtered:
            return IngestionResult(
                success=True,
                total_records=total,
                filtered_records=0,
                sent_records=0,
                failed_records=0,
            )

        payloads = self.prepare_payloads(filtered)
        logger.info(f"Prepared {len(payloads)} API payloads")

        total_sent = 0
        all_errors: list[str] = []

        async with aiohttp.ClientSession() as self._session:
            for i in range(0, len(payloads), self.config.batch_size):
                batch = payloads[i : i + self.config.batch_size]
                sent, errors = await self.send_batch(batch)
                total_sent += sent
                all_errors.extend(errors)
                batch_num = i // self.config.batch_size + 1
                total_batches = (
                    len(payloads) + self.config.batch_size - 1
                ) // self.config.batch_size
                logger.info(
                    f"Batch {batch_num}/{total_batches}: "
                    f"{sent}/{len(batch)} sent, {len(errors)} errors"
                )

        result = IngestionResult(
            success=len(all_errors) == 0,
            total_records=total,
            filtered_records=len(payloads),
            sent_records=total_sent,
            failed_records=len(all_errors),
            errors=all_errors[:20],
        )

        logger.info(f"Ingestion complete: {total_sent} sent, {len(all_errors)} failed")
        return result

    async def run_streaming(self) -> IngestionResult:
        logger.info(f"Streaming records from {self.config.jsonl_path}")

        total_sent = 0
        total_filtered = 0
        total_read = 0
        all_errors: list[str] = []
        batch_num = 0

        owns_session = self._session is None

        if owns_session:
            async with aiohttp.ClientSession() as self._session:
                for batch, read_count in self.iter_payload_batches():
                    batch_num += 1
                    total_read += read_count
                    total_filtered += len(batch)
                    sent, errors = await self.send_batch(batch)
                    total_sent += sent
                    all_errors.extend(errors)
                    logger.info(
                        f"Batch {batch_num}: "
                        f"{sent}/{len(batch)} sent, {len(errors)} errors"
                    )
        else:
            for batch, read_count in self.iter_payload_batches():
                batch_num += 1
                total_read += read_count
                total_filtered += len(batch)
                sent, errors = await self.send_batch(batch)
                total_sent += sent
                all_errors.extend(errors)
                logger.info(
                    f"Batch {batch_num}: {sent}/{len(batch)} sent, {len(errors)} errors"
                )

        result = IngestionResult(
            success=len(all_errors) == 0,
            total_records=total_read,
            filtered_records=total_filtered,
            sent_records=total_sent,
            failed_records=len(all_errors),
            errors=all_errors[:20],
        )

        logger.info(
            f"Streaming ingestion complete: {total_sent} sent, {len(all_errors)} failed"
        )
        return result
