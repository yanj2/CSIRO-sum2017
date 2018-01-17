
'xoxp-289274891636-289747038498-289228558673-2ed610c4e7455bf8630fc75ce61d1276'

def __init__(self, api_key, channel, stack_trace=True):

    Handler.__init__(self)
    self.stack_trace = stack_trace
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
