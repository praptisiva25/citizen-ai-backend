from transformers import pipeline

intent_pipe = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-1.5B-Instruct"
)

def extract_intent(
    title,
    description
):

    prompt = f"""
    Title:
    {title}

    Description:
    {description}

    Return only one word.

    Example:
    pothole
    garbage
    pipe_leak
    """

    response = intent_pipe(
        prompt,
        max_new_tokens=5
    )

    output = response[0][
        "generated_text"
    ].lower()

    if "pothole" in output:
        return "pothole"

    if "garbage" in output:
        return "garbage"

    if "leak" in output:
        return "pipe_leak"

    return "unknown"