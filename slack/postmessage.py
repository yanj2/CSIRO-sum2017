from slackclient import SlackClient

def main():

    slack_token = "xoxp-289274891636-289747038498-289228558673-2ed610c4e7455bf8630fc75ce61d1276"
    sc = SlackClient(slack_token)

    sc.api_call(
        "chat.postMessage",
        channel="#test",
        text="...",
        attachments=[
            {
                "text": "wallop",
                "fallback": "button doesn't work",
                "callback_id": "ok",
                "color": "good",
                "attachment_type": "default",
                "actions": [
                    {
                        "name": "slack",
                        "text": "chess",
                        "type": "button",
                        "value": "chess"
                    }
                ]
            }
            ]
    )

    # Use threads to handle all the content we don't want

if __name__ == "__main__":
    main()
