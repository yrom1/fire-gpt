import numpy as np


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


# def softmax(x):
#     exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
#     return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


...


def main(
    prompt: str,
    n_tokens_to_generate: int = 40,
    model_size: str = "124M",
    model_dir: str = "models",
):
    ...


if __name__ == "__main__":
    import fire

    fire.Fire(main)
