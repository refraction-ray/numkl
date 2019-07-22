import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--all",
        action="store_true",
        default=False,
        help="enable all tests that may fail",
    )


def pytest_collection_modifyitems(config, items):
    """
    if -all not specified, the test functions makerd by issue is omitted
    """
    if config.getoption("--all"):
        return
    skip = pytest.mark.skip(reason="need --all option to run")
    for item in items:
        if "issue" in item.keywords:
            item.add_marker(skip)
