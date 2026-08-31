"""Structured per-run timing statistics for the prs pipeline.

Every pipeline command (ocr / evaluate / extract / compile / run-all)
accumulates timestamps and durations into ``<output_dir>/logs/stats.json``.
Inspect the accumulated runs with ``prs stats`` (rendered with rich).

File layout::

    {"runs": [
        {
            "id": "20260831095315",
            "command": "evaluate",
            "model": "Qwen/Qwen2.5-1.5B-Instruct",
            "started_at": "2026-08-31T09:53:15",
            "duration_sec": 12.3,
            "steps": [
                {
                    "name": "step: evaluate",
                    "started_at": "2026-08-31T09:53:15",
                    "duration_sec": 11.0,
                    "papers": [
                        {"stem": "1801.05086v1", "started_at": "...",
                         "duration_sec": 9.1, "status": "ok"},
                    ],
                }
            ]
        }
    ]}
"""
import json
import time
import threading
from datetime import datetime
from pathlib import Path


class StatsRecorder:
    def __init__(self):
        self._lock = threading.Lock()
        self._path: Path | None = None
        self._run: dict | None = None
        self._stack: list[dict] = []

    def configure(self, path: Path) -> None:
        self._path = Path(path)

    def start_run(self, command: str, **meta) -> None:
        with self._lock:
            now = datetime.now()
            self._run = {
                "id": now.strftime("%Y%m%d%H%M%S"),
                "command": command,
                "started_at": now.isoformat(timespec="seconds"),
                "duration_sec": None,
                "_t0": time.monotonic(),
                "steps": [],
            }
            self._run.update(meta)
            self._stack = [self._run]

    def end_run(self) -> None:
        with self._lock:
            if self._run is None:
                return
            t0 = self._run.pop("_t0", time.monotonic())
            self._run["duration_sec"] = round(time.monotonic() - t0, 3)
            self._stack = []
            if self._path is not None:
                self._flush_locked(self._run)
            self._run = None

    def start_step(self, name: str) -> None:
        with self._lock:
            if self._run is None:
                return
            entry = {
                "name": name,
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "_t0": time.monotonic(),
                "papers": [],
            }
            self._run["steps"].append(entry)
            self._stack.append(entry)

    def end_step(self) -> None:
        with self._lock:
            if len(self._stack) > 1 and "papers" in self._stack[-1]:
                entry = self._stack.pop()
                entry["duration_sec"] = round(
                    time.monotonic() - entry.pop("_t0", time.monotonic()), 3
                )

    def start_paper(self, stem: str) -> None:
        with self._lock:
            step = self._current_step()
            if step is None:
                return
            entry = {
                "stem": stem,
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "_t0": time.monotonic(),
                "status": "ok",
            }
            step["papers"].append(entry)
            self._stack.append(entry)

    def end_paper(self, status: str | None = None) -> None:
        with self._lock:
            if not self._stack or "stem" not in self._stack[-1]:
                return
            entry = self._stack.pop()
            entry["duration_sec"] = round(
                time.monotonic() - entry.pop("_t0", time.monotonic()), 3
            )
            if status:
                entry["status"] = status

    def note_paper(self, stem: str, status: str) -> None:
        """Record a paper that was skipped/cached (no measurable duration)."""
        with self._lock:
            step = self._current_step()
            if step is None:
                return
            step["papers"].append(
                {"stem": stem, "status": status, "duration_sec": 0}
            )

    def _current_step(self) -> dict | None:
        for entry in reversed(self._stack):
            if "papers" in entry:
                return entry
        return None

    def _flush_locked(self, run: dict) -> None:
        assert self._path is not None
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = load_stats(self._path)
        data["runs"].append(run)
        with open(self._path, "w") as f:
            json.dump(data, f, indent=2)


recorder = StatsRecorder()

def configure_stats(path) -> None:
    recorder.configure(path)


def begin_run(command: str, **meta) -> None:
    recorder.start_run(command, **meta)


def end_run() -> None:
    recorder.end_run()


def note_paper(stem: str, status: str) -> None:
    recorder.note_paper(stem, status)


def load_stats(path: Path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"runs": []}


def get_stats_path() -> Path | None:
    return recorder._path


def _fmt(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {secs}s"


def show_stats(path: Path, limit: int = 0) -> None:
    """Render accumulated runs as rich tables. `limit` caps the runs shown (0 = all)."""
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    runs = sorted(load_stats(path)["runs"], key=lambda r: r.get("started_at", ""))
    if not runs:
        console.print(
            "[yellow]No runs recorded yet. Run `prs ocr`, `prs evaluate`, "
            "`prs extract`, or `prs compile` first.[/yellow]"
        )
        return

    if limit > 0:
        runs = runs[-limit:]

    header = f"prs pipeline stats — {len(runs)} session(s) in {stats_path_short(path)}"
    console.print(Panel(header, style="bold cyan"))

    for run in runs:
        steps = run.get("steps", [])
        console.print()

        head = f"command: [bold]{run.get('command')}[/bold]"
        if run.get("model"):
            model = run["model"]
            head += f" · model: [magenta]{model if len(model) <= 42 else model[:39] + '...'}[/magenta]"
        if run.get("force"):
            head += " · force"
        head += f" · started: [cyan]{run.get('started_at', '?')[11:]}[/cyan]"
        head += f" · total: [bold green]{_fmt(run.get('duration_sec', 0))}[/bold green]"
        console.print(Panel(head, box=box.SIMPLE, expand=False))

        table = Table(box=box.SIMPLE_HEAD, pad_edge=False)
        table.add_column("Step", style="cyan", no_wrap=True)
        table.add_column("#Papers", justify="right")
        table.add_column("Step time", justify="right")
        table.add_column("Model time", justify="right")

        for step in steps:
            papers = step.get("papers", [])
            model_time = sum(p.get("duration_sec", 0) for p in papers)
            table.add_row(
                step.get("name", "?"),
                str(len(papers)),
                _fmt(step.get("duration_sec", 0)),
                _fmt(model_time),
            )
        console.print(table)

        if run is runs[-1] and any(s.get("papers") for s in steps):
            detail = Table(
                title=f"latest run detail — {run.get('command')}",
                box=box.SIMPLE_HEAD,
                pad_edge=False,
            )
            detail.add_column("Step", style="cyan", no_wrap=True)
            detail.add_column("Paper", no_wrap=True)
            detail.add_column("Status")
            detail.add_column("Duration", justify="right")
            for step in steps:
                for p in step.get("papers", []):
                    detail.add_row(
                        step.get("name", "?"),
                        p.get("stem", "?"),
                        p.get("status", ""),
                        _fmt(p.get("duration_sec", 0)),
                    )
            console.print()
            console.print(detail)

    total_runs = load_stats(path)["runs"]
    if total_runs:
        total_time = sum(r.get("duration_sec", 0) for r in total_runs)
        avg = total_time / len(total_runs)
        summary = Table.grid(padding=(0, 1))
        summary.add_row(
            "total sessions:", str(len(total_runs)),
            "· total GPU time:", _fmt(total_time),
            "· avg/session:", _fmt(avg),
        )
        console.print()
        console.print(summary)


def stats_path_short(path: Path) -> str:
    text = str(path)
    try:
        text = text.replace(str(Path.cwd()) + "/", "")
    except OSError:
        pass
    return text