from heir.interfaces import CompilationResult, EncValue
import os
import pathlib
from pathlib import Path


def find_above(dirname: str) -> Path | None:
  path = pathlib.Path(__file__).resolve()
  matching = [p for p in path.parents if (p / dirname).exists()]
  return matching[-1] if matching else None


def get_repo_root() -> Path | None:
  default = find_above("bazel-bin")
  found = os.getenv("HEIR_REPO_ROOT_MARKER")
  return Path(found) if found else default


def strip_and_verify_eval_arg_consistency(
    compilation_result: CompilationResult, *args, **kwargs
):
  stripped_args = []
  for i, arg in enumerate(args):
    if i in compilation_result.secret_args:
      if not isinstance(arg, EncValue):
        raise ValueError(f"Expected EncValue for argument {i}, got {type(arg)}")
      # check that the name matches:
      if not arg.identifier == compilation_result.arg_names[i]:
        raise ValueError(
            "Expected EncValue for identifier"
            f" {compilation_result.arg_names[i]}, got EncValue for"
            f" {arg.identifier}"
        )
      # strip the identifier
      stripped_args.append(arg.value)
    else:
      if isinstance(arg, EncValue):
        raise ValueError(
            f"Expected non-EncValue for argument {i}, "
            f"got EncValue for {arg.identifier}"
        )
      stripped_args.append(arg)

  # How to deal with kwargs?
  if kwargs:
    raise NotImplementedError(
        "HEIR's Python Frontend currently doesn't support passing values as"
        " keyword arguments."
    )

  return stripped_args, kwargs
