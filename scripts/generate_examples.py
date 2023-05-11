from src.pyxtend import struct


def generate_examples():
    examples = [
        {
            "name": "Example 1",
            "description": "A simple list of integers",
            "input": [1, 2, 3, 4, 5],
        },
        {
            "name": "Example 2",
            "description": "A list of lists",
            "input": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        },
    ]

    return examples


def main():
    examples = generate_examples()

    with open("EXAMPLES.md", "w") as f:
        for example in examples:
            f.write(f"## {example['name']}\n")
            f.write(f"{example['description']}\n\n")
            f.write("```python\n")
            f.write(f"Input: {example['input']}\n")
            f.write(f"Output (without examples): {struct(example['input'])}\n")
            f.write(f"Output (with examples): {struct(example['input'], examples=True)}\n")
            f.write("```\n\n")


if __name__ == "__main__":
    main()
