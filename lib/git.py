import git
from pathlib import Path

GIT_REPO = git.Repo(Path(__file__).parent, search_parent_directories=True)
GIT_ROOT = Path(GIT_REPO.git.rev_parse("--show-toplevel"))


def get_rev_short():
    sha = GIT_REPO.head.commit.hexsha
    return GIT_REPO.git.rev_parse(sha, short=7)


def get_rev():
    sha = GIT_REPO.head.commit.hexsha
    return GIT_REPO.git.rev_parse(sha)
