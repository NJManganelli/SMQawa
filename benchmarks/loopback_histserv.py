import argparse
import asyncio
import concurrent.futures
import ipaddress
import json
import os
import signal

import grpc
from histserv.protos import hist_pb2_grpc
from histserv.service import Histogrammer

DESCRIPTION = """A histserv 0.2.1 gRPC server bound to a loopback address only, for tests and benchmarks.

`python -m histserv` and `histserv.Server` always bind [::] (every interface) and have no host
option. This server builds the same grpc.aio server around the upstream servicer, but binds the
given loopback address. It uses two names outside histserv.__all__: histserv.service.Histogrammer
and histserv.protos.hist_pb2_grpc.add_HistogrammerServiceServicer_to_server.

It differs from histserv.Server in that:
  - it binds --host (default 127.0.0.1), which must be a loopback address;
  - the asyncio default executor (Snapshot and Flush run in it) has --executor-threads threads;
  - it starts no prune and no stats task, so no histogram is pruned while it runs;
  - once it serves, it writes {"address", "pid"} atomically to --ready-file.

SIGTERM and SIGINT stop it, with --grace seconds for the running RPCs.
"""


def loopback_host(host):
    if not ipaddress.ip_address(host).is_loopback:
        raise argparse.ArgumentTypeError(f"refusing to bind the non-loopback address {host!r}")
    return host


def host_port(host, port):
    return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"


async def serve(args):
    loop = asyncio.get_running_loop()
    loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=args.executor_threads))
    server = grpc.aio.server(
        compression=grpc.Compression.NoCompression,
        options=[
            ("grpc.max_receive_message_length", 1 << 29),
            ("grpc.max_send_message_length", -1),
            ("grpc.so_reuseport", 0),
        ],
    )
    hist_pb2_grpc.add_HistogrammerServiceServicer_to_server(Histogrammer(), server)
    bind = host_port(args.host, args.port)
    port = server.add_insecure_port(bind)
    if port == 0:
        raise SystemExit(f"failed to bind {bind}")
    await server.start()
    address = host_port(args.host, port)
    stop = asyncio.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    if args.ready_file:
        tmp = args.ready_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"address": address, "pid": os.getpid()}, f)
        os.replace(tmp, args.ready_file)
    print(f"histserv (loopback) serving at {address}, pid {os.getpid()}", flush=True)
    await stop.wait()
    await server.stop(args.grace)
    print("histserv (loopback) stopped", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", type=loopback_host, default="127.0.0.1", help="loopback address to bind")
    p.add_argument("--port", type=int, default=0, help="port to bind; 0 picks a free port")
    p.add_argument("--ready-file", help="JSON file written once the server serves")
    p.add_argument("--executor-threads", type=int, default=1, help="threads of the asyncio default executor")
    p.add_argument("--grace", type=float, default=5.0, help="seconds the running RPCs get at shutdown")
    asyncio.run(serve(p.parse_args(argv)))


if __name__ == "__main__":
    main()
