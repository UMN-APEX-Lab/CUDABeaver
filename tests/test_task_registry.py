from pathlib import Path
import json

from harness.benchmark import load_reference
from harness.task_registry import (
    discover_task_stems,
    resolve_task_paths,
    task_error_category,
)


def test_resolve_benchmark_task_paths_from_data_root():
    paths = resolve_task_paths("cuda_110__logic_error__957be815", Path("data"))
    assert paths.input_path.name == "input.json"
    assert paths.prompt_path.name == "prompt.txt"
    assert paths.testbench_path.name == "testbench"
    assert "cuda_110__logic_error__" in str(paths.input_path)


def test_resolve_added_memory_variant_from_benchmark_task_list():
    stem = "cuda_137__memory_crash__9b14b4ef"
    paths = resolve_task_paths(stem, Path("data"))
    assert paths.input_path.exists()
    assert paths.testbench_path.exists()
    assert task_error_category(stem, Path("data")) == "memory_crash"


def test_discover_data_root_uses_benchmark_task_list():
    stems = discover_task_stems(Path("data"))
    assert "cuda_137__memory_crash__9b14b4ef" in stems
    assert "cuda_110__logic_error__957be815" in stems
    task_list = json.loads(Path("benchmark/task_list.json").read_text())
    assert len(stems) == task_list["n_total"]


def test_load_reference_can_materialize_from_task_registry():
    ref = load_reference("cuda_137__memory_crash__9b14b4ef", dataset_root="data")
    assert ref.task_id == "cuda_137__memory_crash__9b14b4ef"
    assert ref.timing_mode.type == "cuda_event_stdout"
    assert ref.reference_solution.type == "file"


def test_release_corpus_uses_benchmark_references():
    from harness.benchmark import REFS_DIR, _refs_dir
    assert _refs_dir("data") == REFS_DIR
    task_list = json.loads(Path("benchmark/task_list.json").read_text())
    for e in task_list["entries"]:
        ref = load_reference(e["task_id"], dataset_root="data")
        assert ref.task_id == e["task_id"]
