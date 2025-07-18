import torch
from lerobot.scripts.server.configs import PolicyServerConfig
from lerobot.scripts.server.policy_server import serve


def start_lerobot_policy_server(use_gpu: bool = False, host: str = "0.0.0.0", port: int = 8080):
    """
    Start the LeRobot policy server.

    Args:
        use_gpu: If True, use CUDA ('cuda'), else use CPU.
        host: Host to bind the gRPC server on.
        port: Port to expose the server on.
    """
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

    config = PolicyServerConfig(
        host=host,
        port=port,
        policy_device=device,
    )

    serve(config)



if __name__ == "__main__":
    # Example usage
    start_lerobot_policy_server(use_gpu=True)