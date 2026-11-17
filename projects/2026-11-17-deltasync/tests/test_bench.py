from deltasync import bench


def test_small_benchmark_all_methods_reconstruct_and_rank_sensibly():
    report = bench.run(size=40_000, seeds=(1,),
                       workloads=["identical", "prepend-1b", "move-32k-section", "full-rewrite"])
    assert report["all_correct"]
    wl = report["workloads"]
    # a one-byte shift defeats offset-aligned and fixed-chunk methods but not rsync or CDC
    shift = wl["prepend-1b"]
    assert shift["rsync-delta"]["mean_total"] < 0.2 * shift["whole-file"]["mean_total"]
    assert shift["aligned-blocks"]["mean_total"] > shift["whole-file"]["mean_total"]
    assert shift["cdc-dedupe"]["mean_total"] < 0.5 * shift["fixed-dedupe"]["mean_total"]
    # on an unrelated rewrite nothing beats sending the file
    rw = wl["full-rewrite"]
    assert min(v["mean_total"] for v in rw.values()) == rw["whole-file"]["mean_total"]


def test_markdown_render_contains_every_workload():
    report = bench.run(size=20_000, seeds=(1,), workloads=["identical", "append-4k"])
    md = bench.to_markdown(report)
    assert "| identical |" in md and "| append-4k |" in md
    assert "byte-identical: **True**" in md
