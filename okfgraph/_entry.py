"""Console-script entry for ``okf``: guards the package import itself.

``okfgraph.cli`` imports the router and every component at module load,
so an import-time failure (broken install, half-edited editable checkout,
a native DLL that fails to load) would otherwise die as a bare Python
traceback before the CLI's §4 catch-all exists. This shim imports the CLI
inside a guard, writes a crash report, and ends with one ``[ERROR]`` line
naming it.
"""
import sys


def okf() -> None:
    try:
        from okfgraph.cli import main
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:  # noqa: BLE001 — start-up guard
        import traceback
        from okfgraph.crash import write_crash_report
        traceback.print_exc()
        path = write_crash_report(exc, command="")
        # ASCII only: this runs before cli forces UTF-8 streams (cp1252).
        where = f" - full traceback: {path}" if path else ""
        print(f"[ERROR] INTERNAL: okf failed to start "
              f"({type(exc).__name__}: {exc}){where}", file=sys.stderr)
        sys.exit(1)
    main()
