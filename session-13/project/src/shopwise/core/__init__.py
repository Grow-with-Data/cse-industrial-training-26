"""What every ShopWise feature depends on and no feature owns. New in v9.

    core/config.py          settings: everything the assistant can be told that is not code
    core/constants.py       fixed facts of the domain: the specialists, the return window
    core/exceptions.py      ShopWiseError, and (v10) the HTTP status each error maps to
    core/logging_config.py  v10: log format, with the request id on every line
    core/deps.py            v10: FastAPI dependencies, how a route gets a service
"""
