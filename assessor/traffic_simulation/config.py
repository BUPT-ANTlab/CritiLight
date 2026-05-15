
import os
import sys


def setup_sumo_environment():
    """Configure SUMO from the caller environment."""
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        raise EnvironmentError(
            "Please set the 'SUMO_HOME' environment variable to your SUMO installation path."
        )

    tools_path = os.path.join(sumo_home, "tools")
    bin_path = os.path.join(sumo_home, "bin")

    if tools_path not in sys.path:
        sys.path.append(tools_path)

    current_path = os.environ.get("PATH", "")
    path_parts = current_path.split(os.pathsep) if current_path else []
    if bin_path not in path_parts:
        os.environ["PATH"] = current_path + os.pathsep + bin_path if current_path else bin_path
