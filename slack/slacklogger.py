
def __init__(self, api_key, channel, stack_trace=True):
    self.client = SlackClient(api_key)
    self.channel = channel 
    if not self.channel.startswith('#'):
        self.channel = '#' + self.channel


def emit(self, record):
    message = self.build_msg(record)

    if self.stack_trace:
        trace = self.build_trace(record, fallback=message)
        attachments = json.dumps([trace])
    else:
        attachments = None

    try:
        self.
