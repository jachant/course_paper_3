"""CodeQL CLI client — creates databases and runs QL queries via subprocess."""

from __future__ import annotations

import csv
import io
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


class CodeQLError(Exception):
    """Raised when a CodeQL command fails."""


class CodeQLClient:
    """Low-level interface to the CodeQL CLI.

    Responsibilities:
    - Create a CodeQL database for a source tree.
    - Execute arbitrary .ql query files against the database.
    - Decode BQRS results into Python-friendly CSV rows.

    Pack resolution strategy
    ------------------------
    CodeQL needs to find ``codeql/python-all`` to resolve ``import python``.
    The client tries, in order:

    1. ``--search-path`` provided explicitly via config.
    2. Auto-detected pack cache at ``~/.codeql/packages``.
    3. ``codeql pack download codeql/python-all`` to fetch packs if absent.

    A temporary "query pack" directory with ``qlpack.yml`` is created once
    and reused for all inline queries.
    """

    # qlpack.yml declaring the dependency on codeql/python-all.
    _QLPACK_YML = (
        "name: vuln-triage/queries\n"
        "version: 0.0.1\n"
        "dependencies:\n"
        '  codeql/python-all: "*"\n'
    )

    def __init__(self, codeql_path: str = "codeql", search_path: str | None = None):
        self._codeql = self._find_binary(codeql_path)
        self._explicit_search_path = search_path
        self._resolved_search_path: str | None = None
        self._db_path: Path | None = None
        self._query_pack_dir: Path | None = None

    # ------------------------------------------------------------------
    # Binary discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _find_binary(name_or_path: str) -> str:
        candidate = Path(name_or_path)
        if candidate.exists():
            return str(candidate.resolve())
        found = shutil.which(name_or_path)
        if found:
            return found
        raise CodeQLError(
            f"CodeQL binary not found: '{name_or_path}'. "
            "Install CodeQL CLI or set codeql.codeql_path in config."
        )

    # ------------------------------------------------------------------
    # Pack resolution
    # ------------------------------------------------------------------

    def _get_search_path(self) -> str | None:
        """Return the search-path string for ``codeql query run``.

        Resolves once, then caches.
        """
        if self._resolved_search_path is not None:
            return self._resolved_search_path or None

        # 1. Explicit config value
        if self._explicit_search_path:
            self._resolved_search_path = self._explicit_search_path
            logger.debug("Using explicit search path: %s", self._resolved_search_path)
            return self._resolved_search_path

        # 2. Auto-detect: ask CodeQL where it can find python packs
        found = self._detect_pack_search_path()
        if found:
            self._resolved_search_path = found
            logger.debug("Auto-detected search path: %s", found)
            return found

        # 3. Try downloading packs
        logger.info("Python QL packs not found — attempting codeql pack download …")
        self._download_packs()
        found = self._detect_pack_search_path()
        if found:
            self._resolved_search_path = found
            return found

        # Nothing worked
        self._resolved_search_path = ""
        logger.warning("Could not locate codeql/python-all packs")
        return None

    def _detect_pack_search_path(self) -> str | None:
        """Find where codeql/python-all lives and return a search path.

        Parses the output of ``codeql resolve qlpacks`` which looks like::

            codeql/python-all (/some/path/python/ql/lib)

        We need to return a path high enough that CodeQL can find ALL
        packs (python-all, python-queries, etc.), so we walk up from
        the resolved path to the repository root.
        """
        res = self._run(
            [self._codeql, "resolve", "qlpacks"], timeout=30
        )
        if res.returncode != 0:
            return None

        # Parse: "codeql/python-all (/path/to/python/ql/lib)"
        for line in res.stdout.splitlines():
            if "codeql/python-all" in line and "(" in line:
                # Extract the path between parentheses
                start = line.index("(") + 1
                end = line.index(")")
                pack_path = Path(line[start:end].strip())

                # Walk up to find a reasonable search root.
                # codeql-repo/python/ql/lib → codeql-repo
                # We need the root that contains the full QL pack tree.
                search_root = pack_path
                for _ in range(5):
                    parent = search_root.parent
                    if parent == search_root:
                        break
                    search_root = parent

                logger.info(
                    "Found codeql/python-all at %s, search root: %s",
                    pack_path, search_root,
                )
                return str(search_root)

        # Fallback: check common locations
        candidates = [
            Path.home() / ".codeql" / "packages",
            Path(self._codeql).parent / "qlpacks",
            Path(self._codeql).parent.parent / "qlpacks",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

        return None

    def _download_packs(self) -> None:
        """Download codeql/python-all via ``codeql pack download``."""
        res = self._run(
            [self._codeql, "pack", "download", "codeql/python-all"],
            timeout=300,
        )
        if res.returncode != 0:
            logger.warning(
                "codeql pack download failed: %s", res.stderr[:500]
            )

    # ------------------------------------------------------------------
    # Database management
    # ------------------------------------------------------------------

    @property
    def db_path(self) -> Path | None:
        return self._db_path

    def create_database(
        self,
        source_root: Path,
        language: str = "python",
        *,
        timeout: int = 600,
        overwrite: bool = False,
    ) -> Path:
        """Create a CodeQL database for *source_root*."""
        if self._db_path and self._db_path.exists() and not overwrite:
            logger.info("Reusing existing CodeQL database: %s", self._db_path)
            return self._db_path

        db_dir = Path(tempfile.mkdtemp(prefix="codeql_db_"))
        logger.info("Creating CodeQL database for %s …", source_root)

        cmd = [
            self._codeql, "database", "create", str(db_dir),
            f"--language={language}",
            f"--source-root={source_root.resolve()}",
            "--overwrite",
        ]

        result = self._run(cmd, timeout=timeout)
        if result.returncode != 0:
            raise CodeQLError(
                f"codeql database create failed (exit {result.returncode}):\n"
                f"{result.stderr[:1000]}"
            )

        self._db_path = db_dir
        logger.info("CodeQL database created: %s", db_dir)
        return db_dir

    # ------------------------------------------------------------------
    # Query execution
    # ------------------------------------------------------------------

    def run_query(
        self,
        query_path: Path,
        *,
        timeout: int = 300,
    ) -> list[dict[str, str]]:
        """Execute a ``.ql`` file and return decoded rows."""
        if not self._db_path:
            raise CodeQLError("No database. Call create_database() first.")

        bqrs_path = Path(tempfile.mktemp(suffix=".bqrs"))

        try:
            run_cmd = [
                self._codeql, "query", "run",
                str(query_path),
                f"--database={self._db_path}",
                f"--output={bqrs_path}",
                "--threads=0",
            ]

            search = self._get_search_path()
            if search:
                run_cmd.append(f"--search-path={search}")

            res = self._run(run_cmd, timeout=timeout)
            if res.returncode != 0:
                raise CodeQLError(
                    f"codeql query run failed for {query_path.name}:\n"
                    f"{res.stderr[:1000]}"
                )

            return self._decode_bqrs(bqrs_path)
        finally:
            bqrs_path.unlink(missing_ok=True)

    def _ensure_query_pack(self) -> Path:
        """Create (once) a temp directory with ``qlpack.yml``.

        Also runs ``codeql pack install`` so that the lock file is
        generated and CodeQL can resolve imports.
        """
        if self._query_pack_dir and self._query_pack_dir.exists():
            return self._query_pack_dir

        pack_dir = Path(tempfile.mkdtemp(prefix="codeql_pack_"))
        (pack_dir / "qlpack.yml").write_text(self._QLPACK_YML, encoding="utf-8")

        # Try to install dependencies (creates codeql-pack.lock.yml)
        install_cmd = [self._codeql, "pack", "install", str(pack_dir)]
        search = self._get_search_path()
        if search:
            install_cmd.append(f"--search-path={search}")

        res = self._run(install_cmd, timeout=120)
        if res.returncode != 0:
            logger.warning(
                "codeql pack install had issues (will try --search-path fallback): %s",
                res.stderr[:500],
            )

        self._query_pack_dir = pack_dir
        logger.debug("Query pack directory: %s", pack_dir)
        return pack_dir

    def run_query_text(
        self,
        ql_text: str,
        *,
        timeout: int = 300,
    ) -> list[dict[str, str]]:
        """Execute an inline QL snippet.

        Written into a query-pack directory with ``qlpack.yml``
        so that ``import python`` resolves correctly.
        """
        pack_dir = self._ensure_query_pack()
        tmp = pack_dir / f"q_{id(ql_text) & 0xFFFFFFFF:08x}.ql"
        try:
            tmp.write_text(ql_text, encoding="utf-8")
            return self.run_query(tmp, timeout=timeout)
        finally:
            tmp.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # BQRS → Python
    # ------------------------------------------------------------------

    def _decode_bqrs(self, bqrs_path: Path) -> list[dict[str, str]]:
        """Decode a .bqrs file into a list of dicts via CSV."""
        csv_path = Path(tempfile.mktemp(suffix=".csv"))
        try:
            cmd = [
                self._codeql, "bqrs", "decode",
                str(bqrs_path),
                f"--output={csv_path}",
                "--format=csv",
            ]
            res = self._run(cmd, timeout=60)
            if res.returncode != 0:
                logger.warning("BQRS decode failed: %s", res.stderr[:500])
                return []

            text = csv_path.read_text(encoding="utf-8")
            if not text.strip():
                return []

            reader = csv.DictReader(io.StringIO(text))
            return list(reader)
        finally:
            csv_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _run(
        cmd: list[str], *, timeout: int = 300
    ) -> subprocess.CompletedProcess[str]:
        logger.debug("Running: %s", " ".join(cmd))
        try:
            return subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired as exc:
            raise CodeQLError(
                f"Command timed out after {timeout}s: {' '.join(cmd[:4])}"
            ) from exc

    def cleanup(self) -> None:
        """Remove temporary database and query-pack directories."""
        if self._db_path and self._db_path.exists():
            shutil.rmtree(self._db_path, ignore_errors=True)
            logger.debug("Cleaned up CodeQL database: %s", self._db_path)
            self._db_path = None
        if self._query_pack_dir and self._query_pack_dir.exists():
            shutil.rmtree(self._query_pack_dir, ignore_errors=True)
            logger.debug("Cleaned up query pack: %s", self._query_pack_dir)
            self._query_pack_dir = None
