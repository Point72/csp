import argparse
import time
from datetime import datetime

import requests

from csp.profiler import ProfilerInfo


def diff_mem(new_meminfo, old_meminfo):
    res = {}
    for obj, (new_count, new_mem) in new_meminfo.items():
        old_count, old_mem = old_meminfo.get(obj, (0, 0))
        res_count = new_count - old_count
        res_mem = new_mem - old_mem

        if res_count != 0 or res_mem != 0:
            res[obj] = (res_count, res_mem)

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", dest="host", action="store", required=True)
    parser.add_argument("--port", dest="port", action="store", required=True)
    parser.add_argument(
        "--interval",
        dest="interval",
        action="store",
        required=False,
        default="5",
        help="interval in seconds",
    )
    parser.add_argument("--include_mem", dest="include_mem", action="store_true")

    args = parser.parse_args()
    url = f"http://{args.host}:{args.port}?format=json"
    if args.include_mem:
        url += "&snap_memory=true"
    last_profile = None
    last_mem = None
    while True:
        result = requests.get(url)
        profile = ProfilerInfo.from_dict(result.json()["profiling_data"])
        if last_profile is not None:
            diff = profile - last_profile
            print("=" * 80, "\n")
            print(datetime.now().isoformat(), ":")
            diff.print_stats()

            if args.include_mem:
                new_mem = result.json()["memory_data"]
                diff = diff_mem(new_mem, last_mem)
                print("Memory diffs:")
                print("%-20s %-10s %-10s" % ("TYPE", "COUNT", "SIZE"))
                for obj, (count, mem) in diff.items():
                    print("%-20s %-10s %-10s" % (obj, count, mem))

        last_profile = profile
        if args.include_mem:
            last_mem = result.json()["memory_data"]
        time.sleep(float(args.interval))
