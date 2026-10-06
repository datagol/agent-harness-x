"""Built-in ask_user tool: let the agent ask the person it works for.

A model given an ambiguous request picks a reading and commits to it. Asked to
"research the Jev model", it researched the Jevons paradox: nine searches and a
report on the wrong subject, when one question would have settled it. Without a
way to ask, guessing is the only move it has.

The tool does not know how to reach a person; the application does. It supplies
``ask``: a coroutine that puts the question in front of someone and returns the
answer -- a web prompt, a terminal ``input()``, a Slack message. When nobody can
answer, ``ask`` returns None and the model is told to proceed on a stated
assumption rather than wait.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Awaitable, Callable

from harnessx.types import PermissionLevel
from ._registration import select_tools

if TYPE_CHECKING:
    from harnessx.tools import ToolRegistry

# Waiting on a person, not a machine: the default tool timeout would cut a
# question off while it is still being read.
ASK_TIMEOUT_SECONDS = 3600.0

NO_ONE_TO_ASK = (
    "No one is available to answer right now. Proceed with the most likely "
    "interpretation, and state that assumption plainly at the start of your reply."
)

AskCallback = Callable[[str, list[str]], Awaitable[str | None]]


def register_ask_user_tool(
    registry: ToolRegistry,
    ask: AskCallback,
    *,
    permission: PermissionLevel | None = PermissionLevel.ALLOW,
    timeout_seconds: float = ASK_TIMEOUT_SECONDS,
    replace: bool = False,
) -> list[str]:
    """Register ``ask_user``, which puts a question to the user through ``ask``.

    ``ask(question, choices)`` returns the user's answer, or None when no one
    can answer. ``choices`` is empty for a free-text question.
    """
    if not select_tools(registry, ["ask_user"], None, None, replace=replace):
        return []

    async def ask_user(question: str, choices: list[str] | None = None) -> str:
        """Ask the user a question and wait for the answer.

        Ask BEFORE doing substantial work when the request is ambiguous in a way
        that changes what you would do: a name, acronym or term you do not
        recognize or that has several plausible meanings, a missing detail you
        cannot sensibly default, or two readings that lead to different work.
        Researching, building or writing the wrong thing costs far more than one
        question.

        Do not ask what you can find out yourself with your tools, what has an
        obvious default, or for permission to continue. Ask one focused
        question; offer choices when the plausible answers are few.

        Args:
            question: The question, with enough context to answer it.
            choices: Optional short answers to offer, such as two readings of a term.
        """
        options = [str(c).strip() for c in (choices or []) if str(c).strip()]
        answer = await ask(question.strip(), options)
        if answer is None or not str(answer).strip():
            return NO_ONE_TO_ASK
        return f"The user answered: {str(answer).strip()}"

    registry.register_tool(
        ask_user, name="ask_user", permission=permission, concurrent=False,
        replay_policy="safe", timeout_seconds=timeout_seconds, replace=replace,
    )
    return ["ask_user"]
