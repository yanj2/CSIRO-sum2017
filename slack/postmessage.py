from slackclient import SlackClient

def main():

    slack_token = "xoxp-289274891636-288579619664-289382437285-b428e9fdbe674508ebcec1b4dcb7caae"
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
