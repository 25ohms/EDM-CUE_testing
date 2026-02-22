import os
import subprocess
from openai import OpenAI

# Safety / cost guardrail
MAX_DIFF_CHARS = 180_000  # adjust as needed


def sh(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Common case for forked PRs where secrets are unavailable
        with open("ai_review.md", "w", encoding="utf-8") as f:
            f.write(
                "### AI Review\n"
                "OPENAI_API_KEY is not available for this run "
                "(this often happens on forked PRs). "
                "Skipping AI review.\n"
            )
        return

    base_ref = os.getenv("BASE_REF", "main")
    repo = os.getenv("REPO", "")
    pr_number = os.getenv("PR_NUMBER", "")
    head_sha = os.getenv("HEAD_SHA", "")

    # Ensure base branch exists locally
    subprocess.check_call(
        ["git", "fetch", "origin", base_ref, "--depth=1"]
    )

    diff = sh(
        ["git", "diff", f"origin/{base_ref}...HEAD", "--unified=3"]
    )

    if not diff:
        with open("ai_review.md", "w", encoding="utf-8") as f:
            f.write("### AI Review\nNo code changes detected in diff.\n")
        return

    if len(diff) > MAX_DIFF_CHARS:
        diff = diff[:MAX_DIFF_CHARS] + "\n\n[diff truncated]\n"

    prompt = f"""
        You are an expert software engineer performing a pull request review.
        
        Repository: {repo}
        Pull Request: {pr_number}
        Head SHA: {head_sha}
        
        Review goals:
        - Identify correctness bugs, edge cases, and security issues
        - Flag breaking changes and missing tests
        - Call out performance concerns
        - Suggest concrete, actionable improvements
        - Be concise and professional
        
        Output format (Markdown):
        1. Summary (3–6 bullets)
        2. High-risk issues (if any) with file hints
        3. Medium / low-risk issues
        4. Tests to add or verify
        5. Nitpicks (optional)
        
        Unified diff:
        ```diff
        {diff}
        """.strip()

    client = OpenAI()  # uses OPENAI_API_KEY from env
    
    response = client.responses.create(
        model="gpt-5.2-codex",
        input=prompt,
    )
    
    review_md = response.output_text.strip()
    if not review_md:
        review_md = "### AI Review\n(No output returned by model.)\n"
    
    with open("ai_review.md", "w", encoding="utf-8") as f:
        f.write(review_md + "\n")
    
if __name__ == "__main__":
    main()
