# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# Standing on the shoulders of giants
from typing import List
import fire
import requests

from llama import Llama

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: int = None,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir: The directory containing checkpoint files for the pretrained model
        tokenizer_path: The path to the tokenizer model file
        temperature: Controls the "creativity" of the model. Higher values make the output more random. Lower values make the output more focused.
        top_p: Controls diversity via nucleus sampling: 0.5 means half of all likelihood-weighted options are considered.
        max_seq_len: Maximum sequence length for input to the model
        max_batch_size: Maximum batch size for input to the model
        max_gen_len: Maximum length of the generated text sequence. If None, it defaults to max_seq_len - 1.
    """
    if max_gen_len is None:
        max_gen_len = max_seq_len - 1

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    instructions: List[str] = [
        # For these prompts, the expected answer is the natural language form
        "I am going to Paris, what should I see?",

        """Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:

            1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.
            2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.
            3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.

            These are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.""",
        "What is Netflix's business model?",

        """Netflix's business model is based on a subscription-based streaming service that provides customers with unlimited access to a vast library of TV shows, movies, and documentaries. The company generates revenue primarily through monthly subscription fees from its members.""",
        "Who is the founder of Facebook?",

        "Mark Zuckerberg",
        "What is the capital of France?",

        "Paris",
        "If you were to look at a star chart, what types of things would you see stars near to?",

        """On a star chart, you would typically see stars near to the following celestial objects:
        
        1. Planets: Stars can appear near planets in the night sky.
        2. The Moon: Stars can appear near the Moon.
        3. Constellations: Stars that are part of a constellation will appear near each other.
        4. The Milky Way: In a star chart, you might see stars near the path of the Milky Way.

        Star charts are a tool used by astronomers and stargazers to locate celestial objects in the night sky.""",
        "Is 22 a perfect number?",

        "No, 22 is not a perfect number. A perfect number is a positive integer that is equal to the sum of its proper divisors, excluding the number itself. The first few perfect numbers are 6, 28, 496, and 8128.",
        "Write a python program to generate 10 random numbers.",

        '''```python
import random

def generate_random_numbers(n):
    return [random.randint(1, 100) for _ in range(n)]

print(generate_random_numbers(10))
```''',
    ]

    results = generator.chat_completion(
        [
            [{"role": "user", "content": instruction}]
            for instruction in instructions
        ],  # format as list of messages
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    for instruction, result in zip(instructions, results):
        out = result["generation"]["content"]
        print(f"> {instruction}")
        print(f"{out}")
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
