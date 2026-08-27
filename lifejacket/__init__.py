import logging

# Per https://docs.python.org/3/howto/logging.html#configuring-logging-for-a-library:
# a library should not configure handlers or levels itself (that's the
# application's call); attaching a NullHandler here just silences "No
# handlers could be found" for callers who use lifejacket without configuring
# logging themselves. The `lifejacket` CLI (post_deployment_analysis.cli)
# configures real logging output for its own use.
logging.getLogger(__name__).addHandler(logging.NullHandler())
